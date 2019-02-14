from functools import partial
from pprint import pformat

import numpy as np
from multistate_kernel import MultiStateKernel
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor, kernels


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
        x: array-like, shape = (n,) or (n, 2)
        compute_err: bool, optional

        Returns
        -------
        MultiStateData
        """
        x_ = x
        if x.ndim == 1:
            x_ = self.msd.sample(x_)
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

    @staticmethod
    def default_initial_kernels(curve, kernel=kernels.RBF, min_length_limits=(0, np.inf)):
        """A variant of default kernels

        Parameters
        ----------
        curve: MultiStateData
            Light curves to use
        kernel: sklearn.gaussian_process.kernels.Kernel
            Kernel to use, default is RBF, should support `length_scale_bounds`
            keyword argument
        min_length_limits: (float, float), optional
            Minimum value of kernel length is set to maximum step between
            neighbour observations. This value clamps the minimum length to
            provided interval. Good choice for RBF kernel is
            `(bin_size, 0.5*approximation_range)`

        Returns
        -------
        Tuple[sklearn.gaussian_process.kernels.Kernel]
        """
        diff_ = (np.max(np.diff(lc.x)) if lc.x.size > 1 else 0 for lc in curve.odict.values())
        min_length_ = [max(min_length_limits[0], min(diff, min_length_limits[1])) for diff in diff_]
        kernels_ = tuple(kernel(length_scale_bounds=(min_length, 1e4)) for min_length in min_length_)
        return kernels_

    @staticmethod
    def default_constant_matrix(n):
        """A variant of initial constant matrix and its bounds

        Parameters
        ----------
        n: int
            Matrix dimension

        Returns
        -------
        initial_constant_matrix: array, shape = (n_kernels, n_kernels)
        constant_matrix_bounds: (array, array), shape = (n_kernels, n_kernels)

        Examples
        --------
        For `n = 1`:
        >>> from thesnisright import GPInterpolator
        >>> m, m_bounds = GPInterpolator.default_constant_matrix(1)
        >>> print(m)
        [[1.]]
        >>> print(m_bounds)
        (array([[0.0001]]), array([[10000.]]))

        For `n = 3`:

        >>> from thesnisright import GPInterpolator
        >>> m, m_bounds = GPInterpolator.default_constant_matrix(3)
        >>> print(m)
        [[1.  0.  0. ]
         [0.5 1.  0. ]
         [0.5 0.5 1. ]]
        >>> print(m_bounds[0])
        [[ 1.e-04  0.e+00  0.e+00]
         [-1.e+03 -1.e+03  0.e+00]
         [-1.e+03 -1.e+03 -1.e+03]]
        >>> print(m_bounds[1])
        [[10000.     0.     0.]
         [ 1000.  1000.     0.]
         [ 1000.  1000.  1000.]]

        """
        m = np.eye(n)
        m[np.tril_indices_from(m, -1)] = 0.5

        m_lower_bound = np.zeros_like(m)
        m_lower_bound[np.tril_indices_from(m_lower_bound)] = -1e3
        m_lower_bound[0, 0] = 1e-4

        m_upper_bound = np.zeros_like(m)
        m_upper_bound[np.tril_indices_from(m_upper_bound)] = 1e3
        m_upper_bound[0, 0] = 1e4

        return m, (m_lower_bound, m_upper_bound)
