"""
Licensed under the GNU General Public License, Version 3.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/gpl-3.0.html.en

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright: (c) 2021, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""

import numba
import numpy as np
from typing import *
from arch_opt_exp.surrogates.sklearn_models.distance_base import *
from arch_opt_exp.surrogates.sklearn_models.mixed_int_dist import hamming_grad

__all__ = ['ArcDistance', 'IndefiniteConditionalDistance', 'ImputationDistance', 'WedgeDistance']


class ArcDistance(WeightedDistance):
    """
    Arc distance, based on:
    Hutter 2013, "A Kernel for Hierarchical Parameter Spaces"
    Zaefferer 2018, "A First Analysis of Kernels for Kriging-Based Optimization in Hierarchical Search Spaces"

    Used kernel: exponential (Matern nu=.5)
    """

    has_gradient = True

    def __init__(self, rho0=1., **kwargs):
        self.rho0 = rho0
        self.rho = None
        super(ArcDistance, self).__init__(**kwargs)

        self._n_dis_values = None

    def _process_samples(self, x: np.ndarray, y: np.ndarray):
        self._n_dis_values = np.max(x, axis=0)+1

        self.rho = np.ones((self.xt.shape[1],))*self.rho0

    def _call(self, u: np.ndarray, v: np.ndarray, u_is_active: np.ndarray, v_is_active: np.ndarray,
              eval_gradient=False) -> Union[float, Tuple[float, Sequence[float]]]:
        d_dim = _arc(u, v, u_is_active, v_is_active, self.is_cont_mask, self.is_discrete_mask, self._n_dis_values,
                     self.theta, self.rho)
        d = np.sum(d_dim)
        if eval_gradient:
            theta_grad = None if self.fix_theta else d_dim
            rho_grad = _arc_dim_grad_rho(
                u, v, u_is_active, v_is_active, self.is_cont_mask, self.is_discrete_mask, self._n_dis_values,
                self.theta, self.rho)

            grad = np.concatenate([theta_grad, rho_grad]) if theta_grad is not None else rho_grad
            return d, grad
        return d

    def kernel(self, **kwargs):
        return CustomDistanceKernel(self, length_scale_bounds='fixed', **kwargs)

    def hyperparameters(self) -> Optional[List[Hyperparameter]]:
        hp = super(ArcDistance, self).hyperparameters()
        hp += [Hyperparameter('rho', 'numeric', (1e-10, 1.), n_elements=self.xt.shape[1])]
        return hp

    def get_hyperparameter_values(self) -> list:
        hp_val = super(ArcDistance, self).get_hyperparameter_values()
        hp_val += [self.rho]
        return hp_val

    def set_hyperparameter_values(self, values: list):
        self.rho = values[-1]
        super(ArcDistance, self).set_hyperparameter_values(values[:-1])


@numba.jit(nopython=True, cache=True)
def _arc(u, v, u_is_active, v_is_active, is_cont_mask, is_discrete_mask, n_dis_values, theta, rho):
    # By default zeros
    d = np.zeros((len(u),))

    # If one variable is active and another is inactive, set to theta
    act_inact_mask = u_is_active != v_is_active
    d[act_inact_mask] = theta[act_inact_mask]

    # If both are active, determine distance
    active_mask = u_is_active & v_is_active

    active_cont_mask = is_cont_mask & active_mask
    act_cont_theta = theta[active_cont_mask]
    if len(act_cont_theta) > 0:
        act_cont_rho = rho[active_cont_mask]
        uv_diff = u[active_cont_mask]-v[active_cont_mask]
        d[active_cont_mask] = act_cont_theta*(2. - 2.*np.cos(np.pi*act_cont_rho*uv_diff))  # Zaefferer 2018 Eq. 1

    active_dis_mask = is_discrete_mask & active_mask
    act_dis_theta = theta[active_dis_mask]
    if len(act_dis_theta) > 0:
        act_dis_rho = rho[active_dis_mask]
        dis_is_diff = u[active_dis_mask] != v[active_dis_mask]
        n_dis_values = n_dis_values[active_dis_mask]
        d[active_dis_mask] = act_dis_theta*((act_dis_rho/(1+(n_dis_values-1)*(1-act_dis_rho)**2))*dis_is_diff)

    return d


@numba.jit(nopython=True, cache=True)
def _arc_dim_grad_rho(u, v, u_is_active, v_is_active, is_cont_mask, is_discrete_mask, n_dis_values, theta, rho):
    # By default zeros
    d = np.zeros((len(u),))

    # If both are active, determine distance
    active_mask = u_is_active & v_is_active

    active_cont_mask = is_cont_mask & active_mask
    act_cont_theta = theta[active_cont_mask]
    if len(act_cont_theta) > 0:
        act_cont_rho = rho[active_cont_mask]
        uv_diff = u[active_cont_mask]-v[active_cont_mask]
        d[active_cont_mask] = act_cont_theta * 2.*np.pi*uv_diff*act_cont_rho * np.sin(np.pi*act_cont_rho*uv_diff)

    active_dis_mask = is_discrete_mask & active_mask
    act_dis_theta = theta[active_dis_mask]
    if len(act_dis_theta) > 0:
        act_dis_rho = rho[active_dis_mask]
        dis_is_diff = u[active_dis_mask] != v[active_dis_mask]
        n_dis_values = n_dis_values[active_dis_mask]

        rho_den = 1+(n_dis_values-1)*(1-act_dis_rho)**2
        grad_term_1 = act_dis_rho/rho_den
        grad_term_2 = act_dis_rho * rho_den**-2 * ((n_dis_values-1)*(1-act_dis_rho)*-act_dis_rho)
        d[active_dis_mask] = act_dis_theta*dis_is_diff*(grad_term_1+grad_term_2)

    return d


class IndefiniteConditionalDistance(WeightedDistance):
    """
    Indefinite Conditional (Ico) distance, based on:
    Zaefferer 2018, "A First Analysis of Kernels for Kriging-Based Optimization in Hierarchical Search Spaces"

    Used kernel: exponential (Matern nu=.5)

    d(x, x') =
       0                   if del(x) = del(x') = False
       rho                 if del(x) != del(x')
       theta * d'(x, x')   if del(x) = del(x') = True
    """

    has_gradient = True

    def __init__(self, rho0=1., fix_rho=False, **kwargs):
        self.rho0 = rho0
        self.rho = None
        self.fix_rho0 = fix_rho
        self.fix_rho = fix_rho
        super(IndefiniteConditionalDistance, self).__init__(**kwargs)

    def _process_samples(self, x: np.ndarray, y: np.ndarray):
        self.rho = np.ones((self.xt.shape[1],))*self.rho0
        self.fix_rho = True if self.xt.shape[1] == 0 else self.fix_rho0

    def _call(self, u: np.ndarray, v: np.ndarray, u_is_active: np.ndarray, v_is_active: np.ndarray,
              eval_gradient=False) -> Union[float, Tuple[float, Sequence[float]]]:
        d = _ico(u, v, u_is_active, v_is_active, self.is_cont_mask, self.is_discrete_mask, self.theta, self.rho)
        if eval_gradient:
            grad = None
            if not self.fix_theta:
                grad = _ico_grad_theta(
                    u, v, u_is_active, v_is_active, self.is_cont_mask, self.is_discrete_mask, self.theta)
            if not self.fix_rho:
                grad_rho = _ico_grad_rho(u_is_active, v_is_active, self.rho)
                grad = np.concatenate([grad, grad_rho]) if grad is not None else grad_rho
            if grad is None:
                grad = np.zeros((0,))
            return d, grad
        return d

    def kernel(self, **kwargs):
        return CustomDistanceKernel(self, length_scale_bounds='fixed', **kwargs)

    def hyperparameters(self) -> Optional[List[Hyperparameter]]:
        hp = super(IndefiniteConditionalDistance, self).hyperparameters()
        if not self.fix_rho:
            hp += [Hyperparameter('rho', 'numeric', (1e-5, 1e5), n_elements=self.xt.shape[1])]
        return hp

    def get_hyperparameter_values(self) -> list:
        hp_val = super(IndefiniteConditionalDistance, self).get_hyperparameter_values()
        if not self.fix_rho:
            hp_val += [self.rho]
        return hp_val

    def set_hyperparameter_values(self, values: list):
        if self.fix_rho:
            return super(IndefiniteConditionalDistance, self).set_hyperparameter_values(values)

        self.rho = values[-1]
        super(IndefiniteConditionalDistance, self).set_hyperparameter_values(values[:-1])


@numba.jit(nopython=True, cache=True)
def _ico(u, v, u_is_active, v_is_active, is_cont_mask, is_discrete_mask, theta, rho):
    # By default zeros
    d = np.zeros((len(u),))

    # If one variable is active and another is inactive, set to rho
    act_inact_mask = u_is_active != v_is_active
    d[act_inact_mask] = rho[act_inact_mask]

    # If both are active, determine distance: square euclidean (continuous) or hamming (discrete)
    _ico_d(d, u, v, theta, u_is_active & v_is_active, is_cont_mask, is_discrete_mask)

    return np.sum(d)


@numba.jit(nopython=True, cache=True)
def _ico_d(d, u, v, theta, calc_mask, is_cont_mask, is_discrete_mask):
    cont_mask = is_cont_mask & calc_mask
    u_cont = u[cont_mask]
    if len(u_cont) > 0:
        d[cont_mask] = sqeuclidean(u_cont, v[cont_mask], w=theta[cont_mask])

    dis_mask = is_discrete_mask & calc_mask
    u_dis = u[dis_mask]
    if len(u_dis) > 0:
        d[dis_mask] = hamming(u_dis, v[dis_mask], w=theta[dis_mask])


@numba.jit(nopython=True, cache=True)
def _ico_grad_theta(u, v, u_is_active, v_is_active, is_cont_mask, is_discrete_mask, theta):
    # By default zeros
    d = np.zeros((len(u),))

    # If both are active, determine distance: square euclidean (continuous) or hamming (discrete)
    _ico_d_grad(d, u, v, theta, u_is_active & v_is_active, is_cont_mask, is_discrete_mask, theta)
    return d


@numba.jit(nopython=True, cache=True)
def _ico_d_grad(d, u, v, theta, calc_mask, is_cont_mask, is_discrete_mask, theta_grad):
    cont_mask = is_cont_mask & calc_mask
    u_cont = u[cont_mask]
    if len(u_cont) > 0:
        d[cont_mask] = sqeuclidean(u_cont, v[cont_mask], w=theta_grad[cont_mask])

    dis_mask = is_discrete_mask & calc_mask
    u_dis = u[dis_mask]
    if len(u_dis) > 0:
        d[dis_mask] = hamming_grad(u_dis, v[dis_mask], w=theta[dis_mask], w_grad=theta_grad[dis_mask])


@numba.jit(nopython=True, cache=True)
def _ico_grad_rho(u_is_active, v_is_active, rho):
    # By default zeros
    d = np.zeros((len(u_is_active),))

    # If one variable is active and another is inactive, set to rho
    act_inact_mask = u_is_active != v_is_active
    d[act_inact_mask] = rho[act_inact_mask]
    return d


@numba.jit(nopython=True, cache=True)
def sqeuclidean(u, v, w):  # Based on scipy.spatial.distance.sqeuclidean
    u_v = u-v
    return u_v*w*u_v


@numba.jit(nopython=True, cache=True)
def hamming(u, v, w):  # Based on scipy.spatial.distance.hamming
    u_ne_v = u != v
    w_ne = w[u_ne_v]

    d = np.zeros((u.size,))
    d[u_ne_v] = w_ne/np.sum(w)
    return d


class ImputationDistance(WeightedDistance):
    """
    Imputation (Imp) distance, based on:
    Zaefferer 2018, "A First Analysis of Kernels for Kriging-Based Optimization in Hierarchical Search Spaces"

    d(x, x') =
       0                    if del(x) = del(x') = False
       theta * d'(x', rho)  if del(x) = False != del(x')
       theta * d'(x, rho)   if del(x) = True != del(x')
       theta * d'(x, x')    if del(x) = del(x') = True
    """

    has_gradient = True

    def __init__(self, rho0=.5, fix_rho=False, **kwargs):
        self.rho0 = 10**rho0
        self.rho = None
        self.fix_rho0 = fix_rho
        self.fix_rho = fix_rho
        if 'theta_bounds' not in kwargs:
            kwargs['theta_bounds'] = (1e-2, 1e2)
        super(ImputationDistance, self).__init__(**kwargs)

        self.xl = None
        self.xu = None
        self.rho_l = None
        self.rho_u = None
        self.rho_x = None
        self.rho_x_grad = None
        self.n_cont = None

    def _process_samples(self, x: np.ndarray, y: np.ndarray):
        self.xl = xl = np.min(x, axis=0)
        self.xu = xu = np.max(x, axis=0)
        x_range = xu-xl
        self.rho_l = xl-2*x_range
        self.rho_u = xu+2*x_range
        self.n_cont = len(np.where(self.is_cont_mask)[0])

        self.rho = np.ones((self.n_cont,))*self.rho0
        self._set_rho_x()
        self.fix_rho = True if self.n_cont == 0 else self.fix_rho0

    def _set_rho_x(self):
        icm = self.is_cont_mask
        self.rho_x = rho_x = np.zeros((self.xt.shape[1],))-1
        rho_x[self.is_cont_mask] = np.log10(self.rho)*(self.rho_u[icm]-self.rho_l[icm])+self.rho_l[icm]

        self.rho_x_grad = rho_x_grad = np.zeros((self.xt.shape[1],))
        rho_x_grad[self.is_cont_mask] = (np.ones((len(self.rho),))/np.log(10.))*(self.rho_u[icm]-self.rho_l[icm])

    def _call(self, u: np.ndarray, v: np.ndarray, u_is_active: np.ndarray, v_is_active: np.ndarray,
              eval_gradient=False) -> Union[float, Tuple[float, Sequence[float]]]:
        d = _imp(u, v, u_is_active, v_is_active, self.is_cont_mask, self.is_discrete_mask, self.theta, self.rho_x)
        if eval_gradient:
            grad = None
            if not self.fix_theta:
                grad = _imp_grad_theta(u, v, u_is_active, v_is_active, self.is_cont_mask, self.is_discrete_mask,
                                       self.theta, self.rho_x)
            if not self.fix_rho:
                grad_rho = _imp_grad_rho(u, v, u_is_active, v_is_active, self.is_cont_mask, self.theta, self.rho_x,
                                         self.rho_x_grad)[self.is_cont_mask]
                grad = np.concatenate([grad, grad_rho]) if grad is not None else grad_rho
            if grad is None:
                grad = np.zeros((0,))
            return d, grad
        return d

    def kernel(self, **kwargs):
        return CustomDistanceKernel(self, length_scale_bounds='fixed', **kwargs)

    def hyperparameters(self) -> Optional[List[Hyperparameter]]:
        hp = super(ImputationDistance, self).hyperparameters()
        if not self.fix_rho:
            hp += [Hyperparameter('rho', 'numeric', (1e0, 1e1), n_elements=self.n_cont)]
        return hp

    def get_hyperparameter_values(self) -> list:
        hp_val = super(ImputationDistance, self).get_hyperparameter_values()
        if not self.fix_rho:
            hp_val += [self.rho]
        return hp_val

    def set_hyperparameter_values(self, values: list):
        if self.fix_rho:
            return super(ImputationDistance, self).set_hyperparameter_values(values)

        self.rho = values[-1]
        self._set_rho_x()
        super(ImputationDistance, self).set_hyperparameter_values(values[:-1])


@numba.jit(nopython=True, cache=True)
def _imp(u, v, u_is_active, v_is_active, is_cont_mask, is_discrete_mask, theta, rho_x):
    # By default zeros
    d = np.zeros((len(u),))

    # If one variable is active and another is inactive, calculate distance to rho
    _ico_d(d, u, rho_x, theta, u_is_active & ~v_is_active, is_cont_mask, is_discrete_mask)
    _ico_d(d, v, rho_x, theta, v_is_active & ~u_is_active, is_cont_mask, is_discrete_mask)

    # If both are active, determine distance: square euclidean (continuous) or hamming (discrete)
    _ico_d(d, u, v, theta, u_is_active & v_is_active, is_cont_mask, is_discrete_mask)

    return np.sum(d)


@numba.jit(nopython=True, cache=True)
def _imp_grad_theta(u, v, u_is_active, v_is_active, is_cont_mask, is_discrete_mask, theta, rho_x):
    # By default zeros
    d = np.zeros((len(u),))

    # If one variable is active and another is inactive, calculate distance to rho
    _ico_d_grad(d, u, rho_x, theta, u_is_active & ~v_is_active, is_cont_mask, is_discrete_mask, theta)
    _ico_d_grad(d, v, rho_x, theta, v_is_active & ~u_is_active, is_cont_mask, is_discrete_mask, theta)

    # If both are active, determine distance: square euclidean (continuous) or hamming (discrete)
    _ico_d_grad(d, u, v, theta, u_is_active & v_is_active, is_cont_mask, is_discrete_mask, theta)

    return d


@numba.jit(nopython=True, cache=True)
def _imp_grad_rho(u, v, u_is_active, v_is_active, is_cont_mask, theta, rho_x, rho_x_grad):
    # By default zeros
    d = np.zeros((len(u),))

    # If one variable is active and another is inactive, calculate distance to rho
    _imp_d_grad_v(d, u, rho_x, rho_x_grad, theta, u_is_active & ~v_is_active, is_cont_mask)
    _imp_d_grad_v(d, v, rho_x, rho_x_grad, theta, v_is_active & ~u_is_active, is_cont_mask)

    return d


@numba.jit(nopython=True, cache=True)
def _imp_d_grad_v(d, u, v, dv, theta, calc_mask, is_cont_mask):
    cont_mask = is_cont_mask & calc_mask
    u_cont = u[cont_mask]
    if len(u_cont) > 0:
        u_v = u_cont - v[cont_mask]
        dv_cont = dv[cont_mask]
        d[cont_mask] = -2*dv_cont*u_v*theta[cont_mask]


class WedgeDistance(WeightedDistance):
    """
    Wedge distance, based on:
    Horn 2019, "Surrogates for Hierarchical Search Spaces: The Wedge-Kernel and an Automated Analysis"
    """

    has_gradient = True

    def __init__(self, rho0=.5, fix_rho=False, **kwargs):
        self.rho0 = 10**rho0
        self.rho = None
        self.fix_rho0 = fix_rho
        self.fix_rho = fix_rho
        super(WedgeDistance, self).__init__(**kwargs)
        self.theta20 = self.theta0
        self.theta2 = None

        self._n_dis_values = None
        self.xl = None
        self.xu = None
        self.rho_x = None
        self.rho_x_grad = None

    def set_samples(self, *args, **kwargs):
        super(WedgeDistance, self).set_samples(*args, **kwargs)
        self._update_derived_values()

    def _process_samples(self, x: np.ndarray, y: np.ndarray):
        self.xl = np.min(x, axis=0)
        self.xu = np.max(x, axis=0)

        self.rho = np.ones((self.xt.shape[1],))*self.rho0
        self.fix_rho = True if self.xt.shape[1] == 0 else self.fix_rho0

        self.theta2 = np.ones((self.xt.shape[1],))*self.theta20

    def _update_derived_values(self):
        self.rho_x = np.log10(self.rho)*np.pi
        self.rho_x_grad = (np.ones((len(self.rho),))/np.log(10.))*np.pi

    def _call(self, u: np.ndarray, v: np.ndarray, u_is_active: np.ndarray, v_is_active: np.ndarray,
              eval_gradient=False) -> Union[float, Tuple[float, Sequence[float]]]:
        d = _wedge(u, v, u_is_active, v_is_active, self.xl, self.xu, self.is_cont_mask, self.is_discrete_mask,
                      self.theta, self.theta2, self.rho_x)
        if eval_gradient:
            grads = []
            if not self.fix_theta:
                grads += [
                    _wedge_grad_theta(u, v, u_is_active, v_is_active, self.xl, self.xu, self.is_cont_mask,
                                      self.is_discrete_mask, self.theta, self.theta2, self.rho_x),
                    _wedge_grad_theta2(u, v, u_is_active, v_is_active, self.xl, self.xu, self.is_cont_mask,
                                       self.is_discrete_mask, self.theta, self.theta2, self.rho_x),
                ]
            if not self.fix_rho:
                grads += [
                    _wedge_grad_rho(u, v, u_is_active, v_is_active, self.xl, self.xu, self.is_cont_mask,
                                    self.is_discrete_mask, self.theta, self.theta2, self.rho_x, self.rho_x_grad),
                ]
            grad = np.zeros((0,)) if len(grads) == 0 else np.concatenate(grads)
            return d, grad
        return d

    def kernel(self, **kwargs):
        return CustomDistanceKernel(self, length_scale_bounds='fixed', **kwargs)

    def hyperparameters(self) -> Optional[List[Hyperparameter]]:
        hp = super(WedgeDistance, self).hyperparameters()
        if not self.fix_theta:
            hp += [Hyperparameter('theta2', 'numeric', self.theta_bounds, n_elements=self.xt.shape[1]),]
        if not self.fix_rho:
            hp += [Hyperparameter('rho', 'numeric', (1e0, 1e1), n_elements=self.xt.shape[1])]
        return hp

    def get_hyperparameter_values(self) -> list:
        hp_val = super(WedgeDistance, self).get_hyperparameter_values()
        if not self.fix_theta:
            hp_val += [self.theta2]
        if not self.fix_rho:
            hp_val += [self.rho]
        return hp_val

    def set_hyperparameter_values(self, values: list):
        if not self.fix_rho:
            self.rho = values[-1]
            values = values[:-1]

        if not self.fix_theta:
            self.theta2 = values[-1]
            values = values[:-1]

        super(WedgeDistance, self).set_hyperparameter_values(values)
        self._update_derived_values()


@numba.jit(nopython=True, cache=True)
def _wedge(u, v, u_is_active, v_is_active, xl, xu, is_cont_mask, is_discrete_mask, theta, theta2, rho_x):
    # By default zeros
    d = np.zeros((len(u),))

    # If one variable is active and another is inactive, calculate triangular distance to rho
    _wedge_tri_d(d, u_is_active & ~v_is_active, u, xl, xu, theta, theta2, rho_x)
    _wedge_tri_d(d, v_is_active & ~u_is_active, v, xl, xu, theta, theta2, rho_x)

    # If both are active, determine normal wedge distance
    is_active_mask = u_is_active & v_is_active
    u_active = u[is_active_mask]
    if len(u_active) > 0:
        theta_act = theta**2 + theta2**2 - 2*theta*theta2*np.cos(rho_x)
        _ico_d(d, u, v, theta_act, is_active_mask, is_cont_mask, is_discrete_mask)

    return np.sum(d)


@numba.jit(nopython=True, cache=True)
def _wedge_grad_theta(u, v, u_is_active, v_is_active, xl, xu, is_cont_mask, is_discrete_mask, theta, theta2, rho_x):
    # By default zeros
    d = np.zeros((len(u),))

    # If one variable is active and another is inactive, calculate triangular distance to rho
    _wedge_tri_d_grad_theta(d, u_is_active & ~v_is_active, u, xl, xu, theta, theta2, rho_x)
    _wedge_tri_d_grad_theta(d, v_is_active & ~u_is_active, v, xl, xu, theta, theta2, rho_x)

    # If both are active, determine normal wedge distance
    is_active_mask = u_is_active & v_is_active
    u_active = u[is_active_mask]
    if len(u_active) > 0:
        theta_act = theta**2 + theta2**2 - 2*theta*theta2*np.cos(rho_x)
        theta_act_grad = 2*theta**2 - 2*theta*theta2*np.cos(rho_x)
        _ico_d_grad(d, u, v, theta_act, is_active_mask, is_cont_mask, is_discrete_mask, theta_act_grad)

    return d


@numba.jit(nopython=True, cache=True)
def _wedge_grad_theta2(u, v, u_is_active, v_is_active, xl, xu, is_cont_mask, is_discrete_mask, theta, theta2, rho_x):
    # By default zeros
    d = np.zeros((len(u),))

    # If one variable is active and another is inactive, calculate triangular distance to rho
    _wedge_tri_d_grad_theta2(d, u_is_active & ~v_is_active, u, xl, xu, theta, theta2, rho_x)
    _wedge_tri_d_grad_theta2(d, v_is_active & ~u_is_active, v, xl, xu, theta, theta2, rho_x)

    # If both are active, determine normal wedge distance
    is_active_mask = u_is_active & v_is_active
    u_active = u[is_active_mask]
    if len(u_active) > 0:
        theta_act = theta**2 + theta2**2 - 2*theta*theta2*np.cos(rho_x)
        theta_act_grad = 2*theta2**2 - 2*theta*theta2*np.cos(rho_x)
        _ico_d_grad(d, u, v, theta_act, is_active_mask, is_cont_mask, is_discrete_mask, theta_act_grad)

    return d


@numba.jit(nopython=True, cache=True)
def _wedge_grad_rho(u, v, u_is_active, v_is_active, xl, xu, is_cont_mask, is_discrete_mask, theta, theta2, rho_x,
                    rho_x_grad):
    # By default zeros
    d = np.zeros((len(u),))

    # If one variable is active and another is inactive, calculate triangular distance to rho
    _wedge_tri_d_grad_rho(d, u_is_active & ~v_is_active, u, xl, xu, theta, theta2, rho_x, rho_x_grad)
    _wedge_tri_d_grad_rho(d, v_is_active & ~u_is_active, v, xl, xu, theta, theta2, rho_x, rho_x_grad)

    # If both are active, determine normal wedge distance
    is_active_mask = u_is_active & v_is_active
    u_active = u[is_active_mask]
    if len(u_active) > 0:
        theta_act = theta**2 + theta2**2 - 2*theta*theta2*np.cos(rho_x)
        theta_act_grad = 2*theta*theta2*rho_x_grad*np.sin(rho_x)
        _ico_d_grad(d, u, v, theta_act, is_active_mask, is_cont_mask, is_discrete_mask, theta_act_grad)

    return d


@numba.jit(nopython=True, cache=True)
def _wedge_tri_d(d, calc_mask, u, xl, xu, theta, theta2, rho_x):
    u_calc = u[calc_mask]
    if len(u_calc) == 0:
        return

    theta_calc = theta[calc_mask]
    theta2_calc = theta2[calc_mask]
    rho_x_calc = rho_x[calc_mask]

    v = (u_calc-xl[calc_mask])/(xu[calc_mask]-xl[calc_mask])
    d[calc_mask] = (theta_calc + v*(theta2_calc*np.cos(rho_x_calc) - theta_calc))**2 + \
                   (v*theta2_calc*np.sin(rho_x_calc))**2


@numba.jit(nopython=True, cache=True)
def _wedge_tri_d_grad_theta(d, calc_mask, u, xl, xu, theta, theta2, rho_x):
    u_calc = u[calc_mask]
    if len(u_calc) == 0:
        return

    theta_calc = theta[calc_mask]
    theta2_calc = theta2[calc_mask]
    rho_x_calc = rho_x[calc_mask]

    v = (u_calc-xl[calc_mask])/(xu[calc_mask]-xl[calc_mask])
    d[calc_mask] = 2.*(theta_calc + v*(theta2_calc*np.cos(rho_x_calc) - theta_calc))*(theta_calc - v*theta_calc)


@numba.jit(nopython=True, cache=True)
def _wedge_tri_d_grad_theta2(d, calc_mask, u, xl, xu, theta, theta2, rho_x):
    u_calc = u[calc_mask]
    if len(u_calc) == 0:
        return

    theta_calc = theta[calc_mask]
    theta2_calc = theta2[calc_mask]
    rho_x_calc = rho_x[calc_mask]

    v = (u_calc-xl[calc_mask])/(xu[calc_mask]-xl[calc_mask])
    d[calc_mask] = 2.*(theta_calc + v*(theta2_calc*np.cos(rho_x_calc) - theta_calc)) *\
                   (v*theta2_calc*np.cos(rho_x_calc)) + \
                   2*(v*theta2_calc*np.sin(rho_x_calc))**2


@numba.jit(nopython=True, cache=True)
def _wedge_tri_d_grad_rho(d, calc_mask, u, xl, xu, theta, theta2, rho_x, rho_x_grad):
    u_calc = u[calc_mask]
    if len(u_calc) == 0:
        return

    theta_calc = theta[calc_mask]
    theta2_calc = theta2[calc_mask]
    rho_x_calc = rho_x[calc_mask]
    rho_x_grad_calc = rho_x_grad[calc_mask]

    v = (u_calc-xl[calc_mask])/(xu[calc_mask]-xl[calc_mask])
    d[calc_mask] = 2.*(theta_calc + v*(theta2_calc*np.cos(rho_x_calc) - theta_calc)) *\
                   (v*theta2_calc*rho_x_grad_calc*-np.sin(rho_x_calc)) + \
                   2.*(v*theta2_calc*np.sin(rho_x_calc))*(v*theta2_calc*rho_x_grad_calc*np.cos(rho_x_calc))


if __name__ == '__main__':
    from arch_opt_exp.surrogates.sklearn_models.gp import *

    from arch_opt_exp.problems.hierarchical import *
    # problem = ZaeffererHierarchicalProblem.from_mode(ZaeffererProblemMode.E_OPT_DIS_IMP_UNPR_BI)
    # problem = ZaeffererHierarchicalProblem()
    # problem = HierarchicalGoldsteinProblem()
    problem = HierarchicalRosenbrockProblem()
    # problem.impute = False

    # kernel = None
    # kernel = ArcDistance().kernel()
    kernel = IndefiniteConditionalDistance().kernel()
    # kernel = ImputationDistance().kernel()
    # kernel = WedgeDistance().kernel()

    sm = SKLearnGPSurrogateModel(kernel=kernel, alpha=1e-6, int_as_discrete=True)
    # sm = SKLearnGPSurrogateModel(alpha=1e-6)

    # from arch_opt_exp.algorithms.surrogate.surrogate_infill import SurrogateBasedInfill
    # SurrogateBasedInfill.plot_model_problem(sm, problem, n_pts=20)
    from arch_opt_exp.surrogates.validation import LOOCrossValidation
    # LOOCrossValidation.check_sample_sizes(sm, problem, show=True, print_progress=True)
    LOOCrossValidation.check_sample_sizes(sm, problem, show=False, print_progress=True, n_pts_test=[10])

    # import matplotlib.pyplot as plt
    # from arch_opt_exp.problems.hierarchical import *
    # from arch_opt_exp.surrogates.sklearn_models.gp import *
    # from arch_opt_exp.problems.discretization import MixedIntProblemHelper
    # from pymoo.model.initialization import Initialization
    # from pymoo.model.duplicate import DefaultDuplicateElimination
    # from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
    #
    # def plot_contour_small(sm_, prob_, sample_problem=False):
    #     is_int_mask = MixedIntProblemHelper.get_is_int_mask(prob_)
    #     is_cat_mask = MixedIntProblemHelper.get_is_cat_mask(prob_)
    #     repair = MixedIntProblemHelper.get_repair(prob_)
    #
    #     init_sampling = Initialization(LatinHypercubeSampling(), repair=repair,
    #                                    eliminate_duplicates=DefaultDuplicateElimination())
    #
    #     np.random.seed(1)
    #     xt = init_sampling.do(prob_, 20).get('X')
    #     yt = prob_.evaluate(xt)
    #     is_active, xt = MixedIntProblemHelper.is_active(prob_, xt)
    #
    #     x = np.linspace(0, 1, 100)
    #     xx1, xx2 = np.meshgrid(x, x)
    #     xx = np.column_stack([xx1.ravel(), xx2.ravel()])
    #     xx_is_active, xx = MixedIntProblemHelper.is_active(prob_, xx)
    #     if sample_problem:
    #         yy = prob_.evaluate(xx).reshape(xx1.shape)
    #     else:
    #         sm_.set_samples(xt, yt, is_int_mask=is_int_mask, is_cat_mask=is_cat_mask, is_active=is_active)
    #         sm_.train()
    #         yy = sm_.predict(xx, xx_is_active)[:, 0].reshape(xx1.shape)
    #
    #     plt.figure(figsize=(3, 3))
    #     c = plt.contourf(xx1, xx2, yy, 50, cmap='Blues_r')
    #     for edge in c.collections:
    #         edge.set_edgecolor('face')
    #     plt.contour(xx1, xx2, yy, 5, colors='k', alpha=.5)
    #     if not sample_problem:
    #         # plt.scatter(xt[:, 0], xt[:, 1], s=20, c='w', edgecolors='k')
    #         plt.scatter(xt[:, 0], xt[:, 1], s=30, marker='x', c='k')
    #     plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    #     plt.xlabel('$x_1$'), plt.ylabel('$y_1$')
    #
    # sms = [
    #     SKLearnGPSurrogateModel(alpha=1e-6),
    #     SKLearnGPSurrogateModel(alpha=1e-6),
    #     SKLearnGPSurrogateModel(kernel=WedgeDistance().kernel(), alpha=1e-6),
    # ]
    # problems = [
    #     ZaeffererHierarchicalProblem.from_mode(ZaeffererProblemMode.E_OPT_DIS_IMP_UNPR_BI),
    #     ZaeffererHierarchicalProblem.from_mode(ZaeffererProblemMode.E_OPT_DIS_IMP_UNPR_BI),
    #     ZaeffererHierarchicalProblem.from_mode(ZaeffererProblemMode.E_OPT_DIS_IMP_UNPR_BI),
    # ]
    # problems[0].impute = False
    # for i, sm in enumerate(sms):
    #     plot_contour_small(sm, problems[i])
    # plot_contour_small(sms[0], prob_=problems[0], sample_problem=True)
    # plt.show()

