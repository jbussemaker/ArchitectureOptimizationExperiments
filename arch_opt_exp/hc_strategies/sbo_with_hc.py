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

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""
import numpy as np
from sb_arch_opt.problem import *
from sb_arch_opt.algo.arch_sbo.algo import *
from sb_arch_opt.algo.arch_sbo.infill import *
from sb_arch_opt.algo.arch_sbo.hc_strategy import *

__all__ = ['HiddenConstraintStrategy', 'HiddenConstraintsSBO', 'HCInfill']


class HiddenConstraintsSBO(SBOInfill):
    """SBO algorithm with hidden constraint strategy"""

    def plot_state(self, x_infill=None, save_path=None, plot_std=False, plot_g=False, plot_axes=None, show=True):
        import matplotlib.pyplot as plt
        from matplotlib.colors import CenteredNorm
        problem = self.problem
        total_pop = self.total_pop
        x_train = total_pop.get('X')
        if plot_g:
            is_failed_train = total_pop.get('CV')[:, 0] > 0.
        else:
            is_failed_train = ArchOptProblemBase.get_failed_points(total_pop)
        n_fail = np.sum(is_failed_train)

        x1, x2 = np.linspace(problem.xl[0], problem.xu[0], 100), np.linspace(problem.xl[1], problem.xu[1], 100)
        xx1, xx2 = np.meshgrid(x1, x2)
        x_eval = np.ones((len(xx1.ravel()), x_train.shape[1]))
        x_eval *= .5*(problem.xu-problem.xl)+problem.xl
        x_eval[:, 0] = xx1.ravel()
        x_eval[:, 1] = xx2.ravel()
        out_plot = problem.evaluate(x_eval, return_as_dictionary=True)
        is_active_eval = out_plot['is_active']
        if plot_g:
            is_failed_ref = np.max(out_plot['G'], axis=1) > 0.
        else:
            is_failed_ref = ArchOptProblemBase.get_failed_points(out_plot)
        pov_ref = (1-is_failed_ref.astype(float)).reshape(xx1.shape)

        figs = []

        def _plot_sfc(z, z_name, path_post, is_g=False, plot_pts=True, ax=None):
            fig = None
            if ax is None:
                fig = plt.figure()
                plt.title(f'SBO model for {problem.__class__.__name__}\n'
                          f'{len(total_pop)} points, {n_fail} failed ({100*n_fail/len(total_pop):.0f}%)')
            zz = z.reshape(xx1.shape)
            ax_ = ax or plt.gca()
            c = ax_.contourf(xx1, xx2, zz, 50, cmap='RdYlGn_r' if is_g else 'cividis',
                             norm=CenteredNorm() if is_g else None)
            if ax is None:
                plt.colorbar(c).set_label(z_name)
            if is_g and plot_pts:
                ax_.contour(xx1, xx2, zz, [0], linewidths=1, colors='k')
            if plot_pts:
                ax_.contour(xx1, xx2, pov_ref, [.5], linewidths=.5, colors='r')
                ax_.scatter(x_train[is_failed_train, 0], x_train[is_failed_train, 1], s=25, c='r', marker='x')
                ax_.scatter(x_train[~is_failed_train, 0], x_train[~is_failed_train, 1], s=25, color=(0, 1, 0), marker='x')
                if x_infill is not None:
                    ax_.scatter([x_infill[0]], [x_infill[1]], s=100, c='m', marker='P')
            if ax is None:
                plt.xlabel('$x_0$'), plt.ylabel('$x_1$')
                if save_path is not None:
                    plt.savefig(f'{save_path}_{path_post}.png')
                figs.append(fig)

        x_eval_norm = self.normalization.forward(x_eval)
        y_predicted = self.surrogate_model.predict_values(x_eval_norm)
        y_predicted_std = np.sqrt(self.surrogate_model.predict_variances(x_eval_norm))
        y_names = [f'f{i}' for i in range(problem.n_obj)]+[f'g{i}' for i in range(problem.n_ieq_constr)]
        for iy in range(y_predicted.shape[1]):
            for do_plot_std in [False, True]:
                if do_plot_std and not plot_std:
                    continue
                plot_ax = (plot_axes or {}).get(f'y{iy}') if not do_plot_std else None
                _plot_sfc((y_predicted_std if do_plot_std else y_predicted)[:, iy],
                          f'{y_names[iy]}{" std dev" if do_plot_std else ""}', f'y{iy}{"_std" if do_plot_std else ""}',
                          is_g=iy >= problem.n_obj, ax=plot_ax)

        infill = self.infill
        f_infill, g_infill = infill.evaluate(x_eval, is_active_eval)
        if self.hc_strategy.adds_infill_constraint():
            g_hc = self.hc_strategy.evaluate_infill_constraint(x_eval)
            g_infill = np.column_stack([g_infill, g_hc])
        for i in range(f_infill.shape[1]):
            plot_ax = (plot_axes or {}).get(f'f{i}')
            _plot_sfc(f_infill[:, i], f'Infill $f_{i}$', f'infill_f{i}', ax=plot_ax)
        for i in range(g_infill.shape[1]):
            plot_ax = (plot_axes or {}).get(f'g{i}')
            _plot_sfc(g_infill[:, i], f'Infill $g_{i}$', f'infill_g{i}', is_g=True, ax=plot_ax)

        if plot_axes is not None:
            if 'true_f' in plot_axes:
                f_true = out_plot['F'][:, 0].reshape(xx1.shape)
                _plot_sfc(f_true, 'True f', 'true_f', plot_pts=False, ax=plot_axes['true_f'])
            if 'true_failed' in plot_axes:
                failed_true = (is_failed_ref.astype(float)-.5).reshape(xx1.shape)
                _plot_sfc(failed_true, 'True Failed', 'true_failed', is_g=True, plot_pts=False, ax=plot_axes['true_failed'])

        if show:
            plt.show()
        if plot_axes is not None and len(figs) > 0:
            for fig in figs:
                plt.close(fig)
