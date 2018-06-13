from pprint import pformat

import numpy as np
from multistate_kernel import MultiStateKernel
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
from interpolate import GPInterpolator

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

    df = pd.read_csv('../gri_pr.csv')
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
    df_c.to_csv("../theta.csv")
