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

This file contains various correction methods.
"""
import itertools
import numpy as np
from scipy.spatial import distance
from cached_property import cached_property
from sb_arch_opt.design_space import ArchDesignSpace, CorrectorUnavailableError
from sb_arch_opt.correction import CorrectorBase, EagerCorrectorBase, ClosestEagerCorrector
from typing import Callable, Optional, Tuple, Generator, List

__all__ = ['CorrectorBase', 'EagerCorrectorBase', 'AnyEagerCorrector', 'GreedyEagerCorrector',
           'ClosestEagerCorrector', 'IsCorrectFuncType', 'LazyCorrectorBase', 'FirstLazyCorrector', 'RandomLazyCorrector',
           'ClosestLazyCorrector', 'ProblemSpecificCorrector']


class ProblemSpecificCorrector(EagerCorrectorBase):

    def correct_x(self, x: np.ndarray, is_active: np.ndarray):
        self.design_space.use_auto_corrector = False
        raise CorrectorUnavailableError('Force problem-specific correction')

    def _get_corrected_x_idx(self, x: np.ndarray) -> np.ndarray:
        raise CorrectorUnavailableError('Force problem-specific correction')


class AnyEagerCorrector(EagerCorrectorBase):
    """
    Eager corrector that chooses a random valid design vector if random_if_multiple, otherwise selects the first.
    """

    def _get_corrected_x_idx(self, x: np.ndarray) -> np.ndarray:
        if self._random_if_multiple:
            x_valid, _ = self.x_valid_active
            return np.random.randint(0, x_valid.shape[0], x.shape[0])
        return np.zeros((x.shape[0],), dtype=int)


class GreedyEagerCorrector(EagerCorrectorBase):
    """
    Eager corrector that corrects design variables one-by-one, starting from the left.
    """

    def _get_corrected_x_idx(self, x: np.ndarray) -> np.ndarray:
        return np.array([self._get_corrected_x_idx_i(xi) for xi in x], dtype=int)

    def _get_corrected_x_idx_i(self, xi: np.ndarray) -> np.ndarray:
        x_valid, is_active_valid = self.x_valid_active

        matched_dv_idx = np.arange(x_valid.shape[0])
        x_valid_matched, is_active_valid_matched = x_valid, is_active_valid
        for i, is_discrete in enumerate(self.is_discrete_mask):
            # Ignore continuous vars
            if not is_discrete:
                continue

            # Match active valid x to value or inactive valid x
            is_active_valid_i = is_active_valid_matched[:, i]
            matched = (is_active_valid_i & (x_valid_matched[:, i] == xi[i])) | (~is_active_valid_i)

            # If there are no matches, match the closest value
            if not np.any(matched):
                x_val_dist = np.abs(x_valid_matched[:, i] - xi[i])
                matched = x_val_dist == np.min(x_val_dist)

            # Select vectors and check if there are any vectors left to choose from
            matched_dv_idx = matched_dv_idx[matched]
            x_valid_matched = x_valid_matched[matched, :]
            is_active_valid_matched = is_active_valid_matched[matched, :]

            # If there is only one matched vector left, there is no need to continue checking
            if len(matched_dv_idx) == 1:
                return matched_dv_idx[0]

        if self._random_if_multiple:
            return np.random.choice(matched_dv_idx)
        return matched_dv_idx[0]


IsCorrectFuncType = Callable[[np.ndarray], Optional[np.ndarray]]


class LazyCorrectorBase(CorrectorBase):
    """
    Corrector that does not have access to the list of all valid discrete design vectors.
    """
    n_try_max = 10000

    def __init__(self, design_space: ArchDesignSpace, is_correct_func: IsCorrectFuncType = None, correct_correct_x=None):
        self.is_correct_func = is_correct_func
        super().__init__(design_space, correct_correct_x=correct_correct_x)

    @cached_property
    def x_opts(self) -> List[List[float]]:
        ds = self._design_space
        x_opts = []
        for i, is_discrete in enumerate(ds.is_discrete_mask):
            if is_discrete:
                x_opts.append(list(range(int(ds.xl[i]), int(ds.xu[i]+1))))
            else:
                x_opts.append([ds.x_mid[i]])
        return x_opts

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        correct_correct_x = self.correct_correct_x
        for i, xi in enumerate(x):
            # Check if the vector is canonical: no need to correct if this is already the case
            is_active_i, is_canonical = self.is_canonical(xi)
            if is_active_i is not None:
                if not correct_correct_x or (correct_correct_x and is_canonical):
                    is_active[i, :] = is_active_i
                    continue

            # Correct the vector
            x[i, :], is_active[i, :] = self._correct_single_x(xi)

    def is_canonical(self, xi: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """
        Function that returns whether a given single design vector x (of length nx) is canonical (i.e. exactly matching
        an imputed design vector):
        - Returns the valid idx if the design vector is valid
        - Returns whether the design vector is also canonical
        """
        # Check whether the design vector is valid
        is_active_i = self.is_correct(xi)
        if is_active_i is None:
            return None, False

        # Check if inactive discrete variables are imputed
        is_canonical = self._is_canonical_inactive(xi, is_active_i)
        return is_active_i, is_canonical

    def is_correct(self, xi: np.ndarray) -> Optional[np.ndarray]:
        """
        Function that returns whether a given single design vector x (of length nx) is correct.
        If valid, returns the activeness vector, otherwise None.
        """
        if len(xi.shape) != 1:
            raise ValueError(f'Expecting vector of length nx, got {xi.shape}')

        is_active_i = self._is_correct(xi)
        if is_active_i is None:
            return

        if is_active_i.shape != xi.shape:
            raise ValueError(f'Expecting return vector of length {xi.shape[0]}, got {is_active_i.shape}')
        return is_active_i

    def _is_correct(self, xi: np.ndarray) -> Optional[np.ndarray]:
        """
        Function that returns whether a given single design vector x (of length nx) is correct.
        If valid, the function should return the activeness vector, otherwise None.
        """
        if self.is_correct_func is None:
            raise RuntimeError('Either provide is_correct_func or override _is_correct!')
        return self.is_correct_func(xi)

    def _correct_single_x(self, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct a single design vector and return the corrected vector and associated activeness information.
        Should generate possible design vectors and use is_valid to check whether generated vectors are valid and get
        activeness information.
        """
        correct_correct_x = self.correct_correct_x

        n_try = 0
        for xi_try in self._generate_x_try(xi):
            # If we originally want to correct also valid vectors, we are looking for a generated canonical vector
            if correct_correct_x:
                xi_active, is_canonical = self.is_canonical(xi_try)
                if not is_canonical:
                    xi_active = None
            else:
                xi_active = self.is_correct(xi_try)

            if xi_active is not None:
                return xi_try, xi_active

            n_try += 1
            if n_try >= self.n_try_max:
                break

        raise RuntimeError(f'No valid design vector found after trying {n_try} times!')

    def _generate_x_try(self, xi: np.ndarray) -> Generator[np.ndarray, None, None]:
        """
        Generate design vectors to try whether they are valid.
        """
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}(correct_correct_x={self.correct_correct_x})'


class FirstLazyCorrector(LazyCorrectorBase):
    """
    Returns the first valid design vector found.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._xi_first = None

    def _generate_x_try(self, xi: np.ndarray) -> Generator[np.ndarray, None, None]:
        # If we have already found the first valid design vector, return it
        if self._xi_first is not None:
            yield self._xi_first
            return

        # Start from 0 vector and generate
        for xi_try in itertools.product(*self.x_opts):
            yield np.array(xi_try)


class RandomLazyCorrector(LazyCorrectorBase):
    """
    Randomly generates design vectors and returns the first valid one.
    """

    def _generate_x_try(self, xi: np.ndarray) -> Generator[np.ndarray, None, None]:
        x_opts = self.x_opts
        while True:
            yield np.array([np.random.choice(opts) if len(opts) > 1 else opts[0] for opts in x_opts])


class ClosestLazyCorrector(LazyCorrectorBase):
    """
    Returns the closest design vector by applying deltas.
    """

    def __init__(self, *args, by_dist=False, euclidean=False, **kwargs):
        self.by_dist = by_dist
        self.euclidean = euclidean
        super().__init__(*args, **kwargs)

    def _generate_x_try(self, xi: np.ndarray) -> Generator[np.ndarray, None, None]:
        deltas = self._get_delta_xi(xi)

        # Sort itertools product vectors by distance
        if self.by_dist:
            try:  # Generate all delta vectors
                xi_deltas = np.array([list(xi_delta) for xi_delta in itertools.product(*deltas)])
                xi_deltas = xi_deltas[1:]  # Skip first because it represents no delta

                # Sort by distance
                metric = 'euclidean' if self.euclidean else 'cityblock'
                weights = np.linspace(1.1, 1, len(deltas))  # Prefer changes on the right side of the design vector
                delta_dist = distance.cdist(xi_deltas, np.zeros((1, len(deltas))), metric=metric, w=weights)
                xi_deltas = xi_deltas[np.argsort(delta_dist[:, 0]), :]

                # Yield delta vectors
                for xi_delta_try in xi_deltas:
                    yield xi+np.array(xi_delta_try)
                return

            except MemoryError:
                self.by_dist = False

        # Use normal itertools product if we do depth-first search
        first = True
        for xi_delta_try in itertools.product(*deltas):
            # Skip the first because it represents no delta
            if first:
                first = False
                continue

            yield xi+np.array(xi_delta_try)

    def _get_delta_xi(self, xi: np.ndarray) -> List[List[int]]:
        def _sort_by_dist(des_var_delta):
            i_sorted = np.argsort(np.abs(des_var_delta))
            return des_var_delta[i_sorted]

        is_discrete_mask = self.is_discrete_mask
        x_opts = self.x_opts
        return [_sort_by_dist(np.array(x_opts[i])-xii) if is_discrete_mask[i] else [0] for i, xii in enumerate(xi)]

    def __repr__(self):
        euc_str = f', euclidean={self.euclidean}' if self.by_dist else ''
        return f'{self.__class__.__name__}(correct_correct_x={self.correct_correct_x}, by_dist={self.by_dist}{euc_str})'
