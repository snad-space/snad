from pprint import pformat

import numpy as np
from multistate_kernel import MultiStateKernel
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd

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
    random_state: int or RandomState or None, optional

    Raises
    ------
    FitFailedError
    """
    def __init__(self, curve,
                 kernels, constant_matrix, constant_matrix_bounds,
                 optimize_method=None, n_restarts_optimizer=0,
                 random_state=None, add_err=0):
        self.curve = curve
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.kernel = MultiStateKernel(kernels, constant_matrix, constant_matrix_bounds)
        if optimize_method is None:
            self.optimizer = 'fmin_l_bfgs_b'  # the default for scikit-learn 0.19
        else:
            self.optimizer = self.optimizer(optimize_method)
        self.regressor = GaussianProcessRegressor(self.kernel, alpha=curve.arrays.err**2 + curve.arrays.y**2 * (add_err/100)**2,
                                                  optimizer=self.optimizer,
                                                  n_restarts_optimizer=self.n_restarts_optimizer,
                                                  normalize_y=True, random_state=self.random_state)
        self.regressor.fit(curve.arrays.x, curve.arrays.y)
#        if self.is_near_bounds(self.regressor.kernel_):
#            raise FitFailedError(
#                '''Fit was not succeed, some of the values are near bounds. Resulted kernel is
#                {}'''.format(pformat(self.regressor.kernel_.get_params()))
#            )

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
        x = curve.sample(x)
        if compute_err:
            y, err = self.regressor.predict(x, return_std=True)
        else:
            y = self.regressor.predict(x)
            err = np.full_like(y, np.nan)
        return curve.convert_arrays(x, y, err)

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
        if random_state is None:
            random_state = self.random_state
        y_samples = self.regressor.sample_y(x, n_samples=samples, random_state=random_state)
        return tuple(curve.convert_arrays(x, y_, np.full_like(y_, np.nan)) for y_ in y_samples)

    @staticmethod
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

    @staticmethod
    def is_near_bounds(kernel, rtol=1e-4):
        params = kernel.get_params()
        bounds_sufix = '_bounds'
        bounds = (k for k in params if k.endswith(bounds_sufix) and params[k] != 'fixed')
        for b in bounds:
            param = b[:-len(bounds_sufix)]
            value = params[param]
            lower_upper = params[b]
            if param == 'scale':
                value = _tri_matrix_to_flat(value)
                lower_upper = np.array([_tri_matrix_to_flat(m) for m in lower_upper])
            else:
                value = np.log(value)
                lower_upper = np.log(lower_upper)
            atol = (lower_upper[1] - lower_upper[0]) * rtol
            if np.any(np.abs(value - lower_upper) < atol):
                return True
        return False

def get_kernels_from_names(*names):
    #k1 = kernels.RBF(length_scale_bounds=(1e-4, 1e4))
    #k2 = kernels.RBF(length_scale_bounds=(1e-4, 1e4))
    # k2 = kernels.WhiteKernel()
    # k3 = kernels.ConstantKernel(constant_value_bounds='fixed')
    #k3 = kernels.WhiteKernel()
    for n in names:
        if n == 'rbf':
            yield kernels.RBF(length_scale_bounds=(1e-4, 1e4))
        elif n == 'white':
            yield kernels.WhiteKernel()
        elif n == 'const':
            yield kernels.ConstantKernel(constant_value_bounds='fixed')
        else:
            raise ValueError(n)

def get_theta_from_params(params):
    ks = list(params['kernels'])
    scale = params['scale']
    flat_scale = scale[np.tril_indices_from(scale)]
    length_scale = np.zeros((3,))
    for i,k in enumerate(ks):
        if isinstance(k, kernels.RBF):
            length_scale[i] = params['s{}__length_scale'.format(i)]
        elif isinstance(k, kernels.WhiteKernel):
            length_scale[i] = 0
        elif isinstance(k, kernels.ConstantKernel):
            length_scale[i] = 1e38
        else:
            raise ValueError(k)
    ret = np.concatenate((flat_scale,length_scale), axis=0)
    return ret

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from curves import OSCCurve
    from sklearn.gaussian_process import kernels

    bands = "g',r',i'".split(',')

    m = np.array([
        [1.0,0.0,0.0],
        [0.5,1.0,0.0],
        [0.5,0.5,1.0],
    ])
    m_bounds = (np.array([[1e-4, 0, 0],
                          [-1e2, -1e3, 0],
                          [-1e2, -1e2, -1e3]]),
                np.array([[1e4, 0, 0],
                          [1e2, 1e3, 0],
                          [1e2, 1e2, 1e3]]))

    colors = {"g'": 'g', "r'": 'r', "i'": 'brown'}

    df = pd.read_csv('gri_pr.csv')
    df = df[['SN','ker1','ker2','ker3','add_err']]

    name_list = []
    theta_list = []

    for index, row in df.iterrows():
        print(row['SN'])
        sn_name = row['SN']
        add_err = row['add_err']

        try:
            k1,k2,k3 = get_kernels_from_names(*[row[i] for i in ['ker1', 'ker2', 'ker3']])

            curve = OSCCurve.from_name(sn_name, bands=bands).binned(bin_width=1, discrete_time=True).filtered(sort='filtered')
            x_ = np.linspace(curve.X[:,1].min(), curve.X[:,1].max(), 101)
            interpolator = GPInterpolator(
                curve, (k1, k2, k3), m, m_bounds,
                optimize_method=None,
                n_restarts_optimizer=0,
                random_state=0,
                add_err=add_err
            )
            msd = interpolator(x_)
            theta_list.append(get_theta_from_params(interpolator.regressor.kernel_.get_params()))
            name_list.append(sn_name)
        except Exception as e:
            print("An error occured: " + str(e))

    sn_name_pd = pd.DataFrame(data={'SN': name_list})
    theta_pd = pd.DataFrame(np.array(theta_list))
    df_c = pd.concat([sn_name_pd.reset_index(drop=True), theta_pd], axis=1)
    df_c.to_csv("theta.csv")
