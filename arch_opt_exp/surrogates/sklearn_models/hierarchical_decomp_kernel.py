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

import numpy as np
from typing import *
from arch_opt_exp.surrogates.sklearn_models.distance_base import *
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, clone, NormalizedKernelMixin, ConstantKernel

__all__ = ['SPWDecompositionKernel', 'DVWDecompositionKernel']


class SPWDecompositionKernel(Kernel, NormalizedKernelMixin, DiscreteHierarchicalKernelBase):
    """
    Sub-Problem-wise decomposition kernel, based on:
    Pelamatti 2020, "Bayesian Optimization of Variable-Size Design Space Problems", section 4.1

    The idea is different sub-problems are identified from the samples, based on the pattern of active variables. The
    samples are then separated by sub-problem and a separate mixed-integer kernel is trained for each sub-problem.

    The implementation discussed in the paper is based on the existence of so-called dimensional variables: only these
    variable determine which sub-problem is being evaluated. In our formulation, no such variable exist, because it is
    not know a-priori which variables determine the activity of other variables, only which variables are active for a
    given design vector.

    Additionally, no information is given in the paper as to what kernel is used to compute the cross-sub-problem
    covariance. Since here the decomposition is done sub-problem-wise, it is assumed that the covariance between
    sub-problems is constant, and therefore a constant kernel is used for this. Its hyperparameter can be tuned to find
    the correct covariance.
    """

    _add_shared_kernel = False

    def __init__(self, mixed_int_kernel: Kernel, _is_discrete_mask=None, _train_is_active=None,
                 _predict_is_active=None, _mi_kernel_mask=None, _cross_mi_kernel_mask=None, _shared_kernel_mask=None,
                 **mi_kernels):
        super(SPWDecompositionKernel, self).__init__()

        self._mi_kernel_base = mixed_int_kernel
        self._is_discrete_mask: Optional[IsDiscreteMask] = _is_discrete_mask
        self._is_cont_mask: Optional[IsDiscreteMask] = ~_is_discrete_mask if _is_discrete_mask is not None else None

        self._train_is_active = _train_is_active
        self._predict_is_active = _predict_is_active

        self._mi_kernel_mask = _mi_kernel_mask
        self._cross_mi_kernel_mask = _cross_mi_kernel_mask
        self._shared_kernel_mask = _shared_kernel_mask
        self._mi_kernels: List[Kernel] = [k for _, k in sorted(list(mi_kernels.items()), key=lambda k: int(k[0][1:]))]
        self._mi_kernels_n = [ker.n_dims for ker in self._mi_kernels]
        for key, ker in mi_kernels.items():
            setattr(self, key, ker)

    def get_params(self, deep=True):
        params = {
            'mixed_int_kernel': self._mi_kernel_base,
            '_is_discrete_mask': self._is_discrete_mask,
            '_train_is_active': self._train_is_active,
            '_predict_is_active': self._predict_is_active,
            '_mi_kernel_mask': self._mi_kernel_mask,
            '_cross_mi_kernel_mask': self._cross_mi_kernel_mask,
            '_shared_kernel_mask': self._shared_kernel_mask,
        }
        params.update({'k%d' % i: ker for i, ker in enumerate(self._mi_kernels)})
        if deep:
            for i, ker in enumerate(self._mi_kernels):
                params.update({'k%d__%s' % (i, key): val for key, val in ker.get_params().items()})

        return params

    @property
    def hyperparameters(self):
        r = []
        for i, ker in enumerate(self._mi_kernels):
            r += [Hyperparameter('k%d__%s' % (i, hp.name), hp.value_type, hp.bounds, hp.n_elements)
                  for hp in ker.hyperparameters]
        return r

    @property
    def theta(self):
        theta = []
        for ker in self._mi_kernels:
            theta += list(ker.theta)
        return np.array(theta)

    @theta.setter
    def theta(self, theta):
        remaining_theta = theta
        for i, ker in enumerate(self._mi_kernels):
            n = self._mi_kernels_n[i]
            ker.theta = remaining_theta[:n]
            remaining_theta = remaining_theta[n:]

    @property
    def bounds(self):
        bounds = []
        for ker in self._mi_kernels:
            ker_bounds = ker.bounds
            if ker_bounds.size > 0:
                bounds.append(ker_bounds)
        return np.vstack(bounds)

    def set_discrete_mask(self, is_discrete_mask: IsDiscreteMask):
        self._is_discrete_mask = MixedIntKernel.get_discrete_mask(is_discrete_mask)
        self._is_cont_mask = ~self._is_discrete_mask

        for i, (_, mask, _) in enumerate(self._mi_kernel_mask or []):
            ker = self._mi_kernels[i]
            if isinstance(ker, DiscreteHierarchicalKernelBase):
                ker.set_discrete_mask(is_discrete_mask[mask])

        for i_kernel, _, _, mask in (self._cross_mi_kernel_mask or []):
            ker = self._mi_kernels[i_kernel]
            if isinstance(ker, DiscreteHierarchicalKernelBase):
                ker.set_discrete_mask(is_discrete_mask[mask])

        if self._add_shared_kernel and self._shared_kernel_mask is not None:
            i_kernel, mask = self._shared_kernel_mask
            ker = self._mi_kernels[i_kernel]
            if isinstance(ker, DiscreteHierarchicalKernelBase):
                ker.set_discrete_mask(is_discrete_mask[mask])

    def set_samples(self, x, y, is_int_mask: IsDiscreteMask, is_cat_mask: IsDiscreteMask, is_active: np.ndarray = None):
        if is_active is None:
            is_active = np.ones(x.shape, dtype=bool)
        self._train_is_active = is_active
        if len(self._mi_kernels) == 0:
            self._create_mi_kernels(x, y, is_int_mask, is_cat_mask, is_active)

        for i, (is_active_mask, mask, _) in enumerate(self._mi_kernel_mask):
            ker = self._mi_kernels[i]
            i_samples = self._get_i_samples(is_active, is_active_mask)
            i_samples_2d = np.ix_(i_samples, mask)
            if isinstance(ker, DiscreteHierarchicalKernelBase):
                ker.set_discrete_mask(self._is_discrete_mask[mask])
                ker.set_samples(x[i_samples_2d], y[i_samples, :], is_int_mask[mask], is_cat_mask[mask],
                                is_active=is_active[i_samples_2d])

        for i_kernel, is_active_mask_i, is_active_mask_j, mask in (self._cross_mi_kernel_mask or []):
            ker = self._mi_kernels[i_kernel]
            i_samples = self._get_i_samples(is_active, is_active_mask_i, is_active_mask_j)
            i_samples_2d = np.ix_(i_samples, mask)
            if isinstance(ker, DiscreteHierarchicalKernelBase):
                ker.set_discrete_mask(self._is_discrete_mask[mask])
                ker.set_samples(x[i_samples_2d], y[i_samples, :], is_int_mask[mask], is_cat_mask[mask],
                                is_active=is_active[i_samples_2d])

        if self._add_shared_kernel and self._shared_kernel_mask is not None:
            i_kernel, mask = self._shared_kernel_mask
            ker = self._mi_kernels[i_kernel]
            if isinstance(ker, DiscreteHierarchicalKernelBase):
                ker.set_discrete_mask(self._is_discrete_mask[mask])
                ker.set_samples(x[:, mask], y, is_int_mask[mask], is_cat_mask[mask], is_active=is_active[:, mask])

    def predict_set_is_active(self, is_active: np.ndarray):
        self._predict_is_active = is_active
        for i, (is_active_mask, mask, _) in enumerate(self._mi_kernel_mask):
            ker = self._mi_kernels[i]
            i_samples = self._get_i_samples(is_active, is_active_mask)
            if isinstance(ker, DiscreteHierarchicalKernelBase):
                ker.predict_set_is_active(is_active[np.ix_(i_samples, mask)])

        for i_kernel, is_active_mask_i, is_active_mask_j, mask in (self._cross_mi_kernel_mask or []):
            ker = self._mi_kernels[i_kernel]
            i_samples = self._get_i_samples(is_active, is_active_mask_i, is_active_mask_j)
            if isinstance(ker, DiscreteHierarchicalKernelBase):
                ker.predict_set_is_active(is_active[np.ix_(i_samples, mask)])

        if self._add_shared_kernel and self._shared_kernel_mask is not None:
            i_kernel, mask = self._shared_kernel_mask
            ker = self._mi_kernels[i_kernel]
            if isinstance(ker, DiscreteHierarchicalKernelBase):
                ker.predict_set_is_active(is_active[:, mask])

    @staticmethod
    def _get_i_samples(is_active: np.ndarray, is_active_mask: np.ndarray, is_active_mask2: np.ndarray = None):
        i_samples = np.all(is_active == is_active_mask, axis=1)
        if is_active_mask2 is not None:
            i_samples |= np.all(is_active == is_active_mask2, axis=1)
        return i_samples

    def _create_mi_kernels(self, x, y, is_int_mask: IsDiscreteMask, is_cat_mask: IsDiscreteMask, is_active: np.ndarray):

        def _create_kernel(var_mask, active_mask=None, active_mask2=None):
            if active_mask is not None:
                i_samples = self._get_i_samples(is_active, active_mask, is_active_mask2=active_mask2)
            else:
                i_samples = np.ones((x.shape[0],), dtype=bool)
            i_samples_2d = np.ix_(i_samples, var_mask)

            # Before cloning, we need to set samples: some kernels need this to determine the amount of hyperparameters
            base_kernel = self._mi_kernel_base
            if isinstance(base_kernel, DiscreteHierarchicalKernelBase):
                base_kernel.set_discrete_mask(self._is_discrete_mask[var_mask])
                base_kernel.set_samples(x[i_samples_2d], y[i_samples, :], is_int_mask[var_mask], is_cat_mask[var_mask],
                                is_active=is_active[i_samples_2d])

            return clone(base_kernel)

        # Search all different sub-problems
        mi_kernel_x_map = {}
        self._mi_kernels = mi_kernels = []
        mi_kernel_act_map = []
        for i in range(is_active.shape[0]):
            is_active_i = tuple(is_active[i, :])  # is_active pattern of the sub-problem
            if is_active_i not in mi_kernel_x_map:
                # Track the different occurring design variable values
                values_tracker = [[x[i, j]] if is_active_i[j] else [] for j in range(x.shape[1])]

                mi_kernel_x_map[is_active_i] = values_tracker
                mi_kernel_act_map.append(is_active_i)

            else:
                for j in range(x.shape[1]):
                    if is_active_i[j]:
                        mi_kernel_x_map[is_active_i][j].append(x[i, j])

        # Determine design variables of each sub-problem: active variables with multiple values
        self._mi_kernel_mask = mi_kernel_mask = []
        for is_active_i in mi_kernel_act_map:
            values_tracker = mi_kernel_x_map[is_active_i]

            # Variables for which multiple values occur in the sample are considered part of the sub-problem
            is_x_sub_problem_map = np.zeros((len(values_tracker),), dtype=bool)
            for j in range(x.shape[1]):
                if len(np.unique(values_tracker[j])) > 1:
                    is_x_sub_problem_map[j] = True
            if np.all(~is_x_sub_problem_map):
                continue

            is_active_i = np.array(is_active_i)
            sub_problem_dimensional_vars_map = is_active_i & ~is_x_sub_problem_map
            mi_kernel_mask.append((is_active_i, is_x_sub_problem_map, sub_problem_dimensional_vars_map))
            mi_kernels.append(_create_kernel(is_x_sub_problem_map, is_active_i))

        # Construct kernels for covariances between sub-problems
        self._cross_mi_kernel_mask = cross_mi_kernel_mask = []
        for i, (is_active_i, sub_problem_mask_i, dim_vars_i) in enumerate(mi_kernel_mask):
            for j in range(i):
                is_active_j, sub_problem_mask_j, dim_vars_j = self._mi_kernel_mask[j]

                # Variables between sub-problems: exclusive-dimensional variables or shared variables
                cross_dim_vars = dim_vars_i | dim_vars_j | (sub_problem_mask_i & sub_problem_mask_j)

                i_kernel = len(mi_kernels)
                cross_mi_kernel_mask.append((i_kernel, is_active_i, is_active_j, cross_dim_vars))

                # Use constant covariance per sub-problem
                mi_kernels.append(ConstantKernel(constant_value=.5, constant_value_bounds=(1e-5, 1.)))

        # Create a kernel for relating shared variables
        if self._add_shared_kernel:
            is_shared = None
            for is_active_i, _, _ in mi_kernel_mask:
                if is_shared is None:
                    is_shared = np.copy(is_active_i)
                else:
                    is_shared &= np.copy(is_active_i)

            if is_shared is not None and np.any(is_shared):
                self._shared_kernel_mask = (len(mi_kernels), is_shared)
                mi_kernels.append(_create_kernel(is_shared))

        self._mi_kernels_n = [ker.n_dims for ker in self._mi_kernels]

    def is_stationary(self):
        return all([ker.is_stationary() for ker in self._mi_kernels])

    @property
    def requires_vector_input(self):
        return any([ker.requires_vector_input for ker in self._mi_kernels])

    def __call__(self, X, Y=None, eval_gradient=False):
        kernel_info = []
        for i, (is_active_mask, mask, _) in enumerate(self._mi_kernel_mask):  # Loop over sub-problems
            # Select samples and design variables corresponding to the sub-problem
            ker = self._mi_kernels[i]

            if Y is None:
                i_samples_x = self._get_i_samples(self._train_is_active, is_active_mask)
                i_samples_y = None
            else:
                i_samples_x = self._get_i_samples(self._predict_is_active, is_active_mask)
                i_samples_y = self._get_i_samples(self._train_is_active, is_active_mask)

            kernel_info.append((ker, i_samples_x, i_samples_y, mask, None))

        # Cross sub-problem kernels
        for i_kernel, is_active_i, is_active_j, mask in self._cross_mi_kernel_mask:
            ker = self._mi_kernels[i_kernel]

            if Y is None:
                # i_samples_x = self._get_i_samples(self._train_is_active, is_active_i, is_active_j)
                i_samples_x_shr_i = self._get_i_samples(self._train_is_active, is_active_i)
                i_samples_x_shr_j = self._get_i_samples(self._train_is_active, is_active_j)
                i_samples_x = i_samples_x_shr_i | i_samples_x_shr_j
                i_samples_y = i_samples_y_shr_i = i_samples_y_shr_j = None
            else:
                # i_samples_x = self._get_i_samples(self._predict_is_active, is_active_i, is_active_j)
                # i_samples_y = self._get_i_samples(self._train_is_active, is_active_i, is_active_j)
                i_samples_x_shr_i = self._get_i_samples(self._predict_is_active, is_active_i)
                i_samples_x_shr_j = self._get_i_samples(self._predict_is_active, is_active_j)
                i_samples_x = i_samples_x_shr_i | i_samples_x_shr_j
                i_samples_y_shr_i = self._get_i_samples(self._train_is_active, is_active_i)
                i_samples_y_shr_j = self._get_i_samples(self._train_is_active, is_active_j)
                i_samples_y = i_samples_y_shr_i | i_samples_y_shr_j

            samples_shared_masks = (i_samples_x_shr_i, i_samples_x_shr_j, i_samples_y_shr_i, i_samples_y_shr_j)
            kernel_info.append((ker, i_samples_x, i_samples_y, mask, samples_shared_masks))

        K = np.zeros((X.shape[0], X.shape[0] if Y is None else Y.shape[0]))
        K_gradients = []
        for ker, i_samples_x, i_samples_y, mask, sharing_masks in kernel_info:
            X_ker = X[np.ix_(i_samples_x, mask)]
            if X_ker.shape[0] == 0 or X_ker.shape[1] == 0:
                if eval_gradient:
                    K_gradients.append(int(ker.bounds.shape[0]))
                continue
            Y_ker = None
            if Y is not None:
                Y_ker = Y[np.ix_(i_samples_y, mask)]

            K_idx = np.ix_(i_samples_x, i_samples_x if Y is None else i_samples_y)

            # Evaluate the sub-problem kernel and integrate the results in the full-size kernel
            K_grad_all = None
            if eval_gradient:
                K_ker, K_grad = ker(X_ker, Y_ker, eval_gradient=eval_gradient)

                K_grad_all = np.zeros((X.shape[0], X.shape[0] if Y is None else Y.shape[0], K_grad.shape[2]))
                K_grad_all[K_idx] = K_grad
            else:
                K_ker = ker(X_ker, Y_ker)

            # Map away in-sub-problem covariances if we are dealing with a cross-sub-problem kernel
            K_ker_full = np.zeros((X.shape[0], X.shape[0] if Y is None else Y.shape[0]))
            K_ker_full[K_idx] = K_ker
            if sharing_masks is not None:
                i_samples_x_shr_i, i_samples_x_shr_j, i_samples_y_shr_i, i_samples_y_shr_j = sharing_masks
                K_idx_1 = np.ix_(i_samples_x_shr_i, i_samples_x_shr_i if Y is None else i_samples_y_shr_i)
                K_idx_2 = np.ix_(i_samples_x_shr_j, i_samples_x_shr_j if Y is None else i_samples_y_shr_j)

                K_ker_full[K_idx_1] = 0.
                K_ker_full[K_idx_2] = 0.
                if eval_gradient:
                    K_grad_all[K_idx_1] = 0.
                    K_grad_all[K_idx_2] = 0.

            if eval_gradient:
                K_gradients.append(K_grad_all)

            K += K_ker_full

        if eval_gradient:
            for i, K_grad in enumerate(list(K_gradients)):
                if isinstance(K_grad, int):
                    K_gradients[i] = np.zeros(K.shape+(K_grad,))

            K_gradients = np.dstack(K_gradients)

        if self._add_shared_kernel and self._shared_kernel_mask is not None:  # Evaluate the shared-variables kernel
            i_kernel, mask = self._shared_kernel_mask
            ker = self._mi_kernels[i_kernel]
            X_ker = X[:, mask]
            Y_ker = Y[:, mask] if Y is not None else None

            K_grad_shr = None
            if eval_gradient:
                K_shr, K_grad_shr = ker(X_ker, Y_ker, eval_gradient=eval_gradient)
            else:
                K_shr = ker(X_ker, Y_ker)

            # Integrate with other kernels
            if eval_gradient: # Cross product
                K_gradients = np.dstack([K_gradients*K_shr[:, :, None], K_grad_shr*K[:, :, None]])
            K *= K_shr

        return (K, K_gradients) if eval_gradient else K

    def diag(self, X):
        # This function is called with X being the prediction points (not training)
        # It is assumed that the kernel is normalized, meaning the diagonal of the kernel matrix is always one
        return np.ones((X.shape[0],))


class DVWDecompositionKernel(SPWDecompositionKernel):
    """
    Dimensional Variable-wise decomposition kernel, based on:
    Pelamatti 2020, "Bayesian Optimization of Variable-Size Design Space Problems", section 4.2

    The difference between the DVW and SPW kernels are that the former combines decomposed kernels for each different
    dimensional variable (instead of only per sub-problem), and adds a kernel relating shared variables to each other.
    In our notation, we don't know which variables are dimensional variables, so our only difference is that we add
    the shared variable kernel.
    """

    _add_shared_kernel = True


if __name__ == '__main__':
    from arch_opt_exp.surrogates.sklearn_models.gp import *
    from arch_opt_exp.surrogates.sklearn_models.mixed_int_dist import *
    from arch_opt_exp.surrogates.sklearn_models.hierarchical_dist import *

    from arch_opt_exp.problems.hierarchical import *
    # problem = ZaeffererHierarchicalProblem.from_mode(ZaeffererProblemMode.E_OPT_DIS_IMP_UNPR_BI)
    # problem = HierarchicalGoldsteinProblem()
    problem = HierarchicalRosenbrockProblem()
    # problem.impute = False

    # kernel = None
    # kernel = ArcDistance().kernel()
    # kernel = IndefiniteConditionalDistance().kernel()
    # kernel = ImputationDistance().kernel()
    # kernel = WedgeDistance().kernel()
    # kernel = SPWDecompositionKernel(MixedIntKernel.get_cont_kernel())
    kernel = SPWDecompositionKernel(CompoundSymmetryKernel().kernel())
    # kernel = DVWDecompositionKernel(MixedIntKernel.get_cont_kernel())
    # kernel = DVWDecompositionKernel(CompoundSymmetryKernel().kernel())

    sm = SKLearnGPSurrogateModel(kernel=kernel, alpha=1e-6, int_as_discrete=True)

    # from arch_opt_exp.algorithms.surrogate.surrogate_infill import SurrogateBasedInfill
    # SurrogateBasedInfill.plot_model_problem(sm, problem, n_pts=20)
    from arch_opt_exp.surrogates.validation import LOOCrossValidation
    # LOOCrossValidation.check_sample_sizes(sm, problem, show=True, print_progress=True)
    LOOCrossValidation.check_sample_sizes(sm, problem, show=False, print_progress=True, n_pts_test=[10])
