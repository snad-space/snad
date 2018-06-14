from functools import partial
from pprint import pformat

import numpy as np
from multistate_kernel import MultiStateKernel
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor


def _tri_matrix_to_flat(matrix):
    return matrix[np.tril_indices_from(matrix)]


class FitFailedError(RuntimeError):
    pass


class GPInterpolator(object):
    """Interpolate light curve using multi-state Gaussian Process

    Parameters
    ----------
    curve: MultiStateData
    kernels: tuple of sklearn.gaussian_process kernels, size is len(curve)
    constant_matrix: matrix, shape is (len(curves), len(curves))
    constant_matrix_bounds: pair of matrices
    optimize_method: str or None, optional
        Optimize method name, should be valid `scipy.optimize.minimize` method
        with a support of constraints and hessian update strategy. The default
        is None, the default value of `optimizer` argument of
        `sklearn.gaussian_process.GaussianProcessRegressor` will be used
    n_restarts_optimizer: int, optional
    add_err: float, optional
        Additional error for all the data, percent of flux
    raise_on_bounds: bool, optional
        Raise or not if fitted parameters are near bounds
    random_state: int or RandomState or None, optional

    Raises
    ------
    FitFailedError
    """
    def __init__(self, curve,
                 kernels, constant_matrix, constant_matrix_bounds,
                 normalize_y=True,
                 optimize_method=None, n_restarts_optimizer=0,
                 random_state=None, add_err=0, raise_on_bounds=True):
        self.msd = curve
        self.optimizer = optimize_method

        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.kernel = MultiStateKernel(kernels, constant_matrix, constant_matrix_bounds)
        alpha = self.msd.arrays.err**2 + self.msd.arrays.y**2*(add_err/100)**2
        self.regressor = GaussianProcessRegressor(self.kernel,
                                                  alpha=alpha,
                                                  optimizer=self.get_optimizer(optimize_method),
                                                  n_restarts_optimizer=self.n_restarts_optimizer,
                                                  normalize_y=normalize_y, random_state=self.random_state)
        self.regressor.fit(self.msd.arrays.x, self.msd.arrays.y)
        self.near_bounds = self.is_near_bounds(self.regressor.kernel_)
        if self.near_bounds and raise_on_bounds:
            raise FitFailedError(
                '''Fit was not succeed, some of the values are near bounds. Resulted kernel is
                {}'''.format(pformat(self.regressor.kernel_.get_params()))
            )


    def __call__(self, x, compute_err=True):
        """Produce median and std of GP realizations

        Parameters
        ----------
        x: array-like, shape = (n,)
        compute_err: bool, optional

        Returns
        -------
        MultiStateData
        """
        x_ = self.msd.sample(x)
        if compute_err:
            y, err = self.regressor.predict(x_, return_std=True)
        else:
            y = self.regressor.predict(x_)
            err = np.full_like(y, np.nan)
        msd = self.msd.convert_arrays(x_, y, err)
        return msd

    def y_samples(self, x, samples=1, random_state=None):
        """Generate GP realizations

        Parameters
        ----------
        x: array-like, shape = (n,)
        samples: int, optional
            Number of samples to generate. If larger than 0, additional tuple
            of samples will be returned
        random_state: int or RandomState or None, optional

        Returns
        -------
        tuple[MultiStateData]
        """
        x_ = self.msd.sample(x)
        if random_state is None:
            random_state = self.random_state
        y_samples = self.regressor.sample_y(x_, n_samples=samples, random_state=random_state)
        msd_ = tuple(self.msd.convert_arrays(x_, y_, np.full_like(y_, np.nan)) for y_ in y_samples)
        return msd_

    @staticmethod
    def _optimizer(obj_func, initial_theta, bounds, method):
            constraints = [optimize.LinearConstraint(np.eye(initial_theta.shape[0]), bounds[:, 0], bounds[:, 1])]
            res = optimize.minimize(lambda theta: obj_func(theta=theta, eval_gradient=False),
                                    initial_theta,
                                    constraints=constraints,
                                    method=method,
                                    jac=lambda theta: obj_func(theta=theta, eval_gradient=True)[1],
                                    hess=optimize.BFGS(),
                                    options={'gtol': 1e-6})
            return res.x, res.fun

    @staticmethod
    def get_optimizer(method='trust-constr'):
        if method is None:
            return 'fmin_l_bfgs_b'

        return partial(GPInterpolator._optimizer, method=method)

    @staticmethod
    def is_near_bounds(kernel, rtol=1e-4):
        params = kernel.get_params()
        bounds_sufix = '_bounds'
        bounds = (k for k in params if k.endswith(bounds_sufix) if str(params[k]) != 'fixed')
        for b in bounds:
            param = b[:-len(bounds_sufix)]
            value = params[param]
            lower_upper = params[b]
            if param == 'scale':
                value = _tri_matrix_to_flat(value)
                lower_upper = np.array([_tri_matrix_to_flat(m) for m in lower_upper])
                value[0] = np.log(value[0])
                lower_upper[:,0] = np.log(lower_upper[:,0])
            else:
                value = np.log(value)
                lower_upper = np.log(lower_upper)
            atol = (lower_upper[1] - lower_upper[0]) * rtol
            if np.any(np.abs(value - lower_upper) < atol):
                return True
        return False
