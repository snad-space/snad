from pprint import pformat

import numpy as np
from multistate_kernel import MultiStateKernel
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor


def optimizer(method='trust-constr'):
    def f(obj_func, initial_theta, bounds):
        constraints = [optimize.LinearConstraint(np.eye(initial_theta.shape[0]), bounds[:, 0], bounds[:, 1])]
        res = optimize.minimize(lambda theta: obj_func(theta=theta, eval_gradient=False),
                                initial_theta,
                                constraints=constraints,
                                method=method,
                                jac=lambda theta: obj_func(theta=theta, eval_gradient=True)[1],
                                hess=optimize.BFGS(),
                                options={'gtol': 1e-6})
        return res.x, res.fun
    return f


def _matrix_to_flat(matrix):
    return matrix[np.tril_indices_from(matrix)]


def is_near_bounds(kernel, rtol=1e-4):
    params = kernel.get_params()
    bounds_sufix = '_bounds'
    bounds = (k for k in params if k.endswith(bounds_sufix) and params[k] != 'fixed')
    for b in bounds:
        param = b[:-len(bounds_sufix)]
        value = params[param]
        lower_upper = params[b]
        if param == 'scale':
            value = _matrix_to_flat(value)
            lower_upper = np.array([_matrix_to_flat(m) for m in lower_upper])
        else:
            value = np.log(value)
            lower_upper = np.log(lower_upper)
        atol = (lower_upper[1] - lower_upper[0]) * rtol
        if np.any(np.abs(value - lower_upper) < atol):
            return True
    return False


class FitFailedError(RuntimeError):
    pass


def interpolate(curve, x,
                kernels, constant_matrix, constant_matrix_bounds,
                samples=0,
                optimize_method=None, n_restarts_optimizer=0,
                random_state=None):
    """Interpolate light curve using multi-state Gaussian Process

    Parameters
    ----------
    curve: MultiStateData
    x: array-like, shape = (n,)
    kernels: tuple of sklearn.gaussian_process kernels, size is len(curve)
    constant_matrix: matrix, shape is (len(curves), len(curves))
    constant_matrix_bounds: pair of matrices
    samples: int, optional
        Number of samples to generate. If larger than 0, additional tuple of
        samples will be returned
    optimize_method: str or None, optional
        Optimize method name, should be valid `scipy.optimize.minimize` method
        with a support of constraints and hessian update strategy. The default
        is None, the default value of `optimizer` argument of
        `sklearn.gaussian_process.GaussianProcessRegressor` will be used
    n_restarts_optimizer: int, optional
    random_state: int or None, optional

    Returns
    -------
    MultiStateData
    tuple[MultiStateDate], optional

    Raises:
    FitFailedError
    """
    kernel = MultiStateKernel(kernels, constant_matrix, constant_matrix_bounds)
    if optimize_method is None:
        optimize_method = 'fmin_l_bfgs_b'  # the default for scikit-learn 0.19
    else:
        optimize_method = optimizer(optimize_method)
    regressor = GaussianProcessRegressor(kernel, alpha=curve.arrays.err**2,
                                         optimizer=optimize_method, n_restarts_optimizer=n_restarts_optimizer,
                                         normalize_y=False, random_state=random_state)
    regressor.fit(curve.arrays.x, curve.arrays.y)
    if is_near_bounds(regressor.kernel_):
        raise FitFailedError(
            '''Fit was not succeed, some of the values are near bounds. Resulted kernel is
            {}'''.format(pformat(regressor.kernel_.get_params()))
        )
    x = curve.sample(x)
    y, err = regressor.predict(x, return_std=True)
    msd = curve.convert_arrays(x, y, err)
    if samples == 0:
        return msd
    y_samples = regressor.sample_y(x, n_samples=samples, random_state=random_state)
    msd_samples = tuple(curve.convert_arrays(x, y_, np.full_like(y_, np.nan)) for y_ in y_samples)
    return msd, msd_samples


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from curves import OSCCurve
    from sklearn.gaussian_process import kernels

    sn_name = 'SDSS-II SN 1609'
    bands = "g',r',i'".split(',')

    k1 = kernels.RBF(length_scale_bounds=(1e-4, 1e4))
    k2 = kernels.RBF(length_scale_bounds=(1e-4, 1e4))
    # k2 = kernels.WhiteKernel()
    # k3 = kernels.ConstantKernel(constant_value_bounds='fixed')
    k3 = kernels.WhiteKernel()

    m = np.zeros((3,3)); m[0, 0] = 1
    m_bounds = (np.array([[1e-4, 0, 0],
                          [-1e2, -1e3, 0],
                          [-1e2, -1e2, -1e3]]),
                np.array([[1e4, 0, 0],
                          [1e2, 1e3, 0],
                          [1e2, 1e2, 1e3]]))

    colors = {"g'": 'g', "r'": 'r', "i'": 'brown'}

    curve = OSCCurve.from_name(sn_name, bands=bands).binned(bin_width=1).filtered(sort='filtered')
    x_ = np.linspace(curve.X[:,1].min(), curve.X[:,1].max(), 101)
    msd = interpolate(
        curve, x_, (k1, k2, k3), m, m_bounds,
        optimize_method='trust-constr',
        n_restarts_optimizer=0,
        random_state=0
    )
    for i, band in enumerate(bands):
        plt.subplot(2, 2, i+1)
        blc = curve[band]
        plt.errorbar(blc['x'], blc['y'], blc['err'], marker='x', ls='', color=colors[band])
        plt.plot(msd.odict[band].x, msd.odict[band].y, color=colors[band], label=band)
        plt.grid()
        plt.legend()
    plt.show()
