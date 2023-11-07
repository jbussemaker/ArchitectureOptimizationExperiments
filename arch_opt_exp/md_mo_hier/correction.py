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
from sb_arch_opt.design_space import ArchDesignSpace
from typing import Callable, Optional, Tuple, Generator, List

__all__ = ['CorrectorBase', 'EagerCorrectorBase', 'AnyEagerCorrector', 'GreedyEagerCorrector',
           'ClosestEagerCorrector', 'IsValidFuncType', 'LazyCorrectorBase', 'FirstLazyCorrector', 'RandomLazyCorrector',
           'ClosestLazyCorrector']


class CorrectorBase:
    """
    Base class implementing some generic correction algorithm.
    Correction is the mechanism of taking any input design vector x and ensuring it is a valid design vector, that is:
    all (hierarchical) value constraints are satisfied.
    Imputation is the mechanism of turning a valid vector into a canonical vector, that is: an x where inactive
    variables are replaced by 0 (discrete) or mid-bounds (continuous).

    We assume that only discrete variables determine activeness and are subject to value constraints, so only
    discrete variables need to be corrected.

    From this, there are three "statuses" that design vectors can have:
    - Canonical: valid and inactive discrete variables are imputed
    - Valid: active discrete variables represent a valid combination (all value constraints are satisfied)
    - Invalid: one or more value constraints are violated (for discrete variables)

    Invalid design vectors always need to be corrected to a valid/canonical design vector.
    Valid design vectors may optionally be "corrected" to a canonical design vector too, which allows non-canonical
    design vectors to be modified.
    """

    default_correct_valid_x = True

    def __init__(self, design_space: ArchDesignSpace, correct_valid_x: bool = None):
        self._design_space = design_space
        self.correct_valid_x = self.default_correct_valid_x if correct_valid_x is None else correct_valid_x

    @property
    def design_space(self) -> ArchDesignSpace:
        """Mask specifying for each design variable whether it is a discrete variable or not."""
        return self._design_space

    @cached_property
    def is_discrete_mask(self) -> np.ndarray:
        return self._design_space.is_discrete_mask

    @cached_property
    def x_imp_discrete(self) -> np.ndarray:
        return self._design_space.xl[self._design_space.is_discrete_mask]

    def _is_canonical_inactive(self, xi: np.ndarray, is_active_i: np.ndarray) -> bool:
        # Check whether each discrete variable has its corresponding imputed value
        is_discrete_mask = self.is_discrete_mask
        is_x_imp_discrete = xi[is_discrete_mask] == self.x_imp_discrete

        # Check which discrete design variables are inactive
        is_inactive = ~is_active_i[is_discrete_mask]

        # Check if all inactive discrete design variables have their corresponding imputed values
        return np.all(is_x_imp_discrete[is_inactive])

    def correct_x(self, x: np.ndarray, is_active: np.ndarray):
        """
        Fill the activeness matrix (n x nx) and if needed correct design vectors (n x nx) that are partially inactive.
        No need to impute inactive design variables.
        """
        # Quit if there are no discrete design variables
        if not np.any(self.is_discrete_mask):
            return

        self._correct_x(x, is_active)

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        """
        Fill the activeness matrix (n x nx) and if needed correct design vectors (n x nx) that are partially inactive.
        No need to impute inactive design variables.
        """
        raise NotImplementedError

    def __str__(self):
        return repr(self)

    def __repr__(self):
        raise NotImplementedError


class EagerCorrectorBase(CorrectorBase):
    """
    Corrector that has access to the list of all valid discrete design vectors.
    """

    default_random_if_multiple = True

    def __init__(self, design_space: ArchDesignSpace, correct_valid_x: bool = None, random_if_multiple: bool = None):
        self._x_valid = None
        self._is_active_valid = None
        self._random_if_multiple = self.default_random_if_multiple if random_if_multiple is None else random_if_multiple
        super().__init__(design_space, correct_valid_x=correct_valid_x)

    @property
    def x_valid(self) -> np.ndarray:
        if self._x_valid is None:
            self._x_valid, self._is_active_valid = self._design_space.all_discrete_x
        return self._x_valid

    @property
    def is_active_valid(self) -> np.ndarray:
        if self._is_active_valid is None:
            self._x_valid, self._is_active_valid = self._design_space.all_discrete_x
        return self._is_active_valid

    @cached_property
    def _x_canonical_map(self) -> dict:
        x_canonical_map = {}
        is_discrete_mask = self.is_discrete_mask
        for i, xi in enumerate(self.x_valid):
            x_canonical_map[tuple(xi[is_discrete_mask])] = i
        return x_canonical_map

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        # Separate canonical design vectors
        correct_idx = self.get_canonical_idx(x) if self.correct_valid_x else self.get_valid_idx(x)
        is_correct = correct_idx != -1
        to_be_corrected = ~is_correct

        # Set activeness information of correct vectors
        is_active[is_correct, :] = self.is_active_valid[correct_idx[is_correct], :]

        # Get corrected design vector indices
        xi_corrected = self._get_corrected_x_idx(x[to_be_corrected, :])
        if len(xi_corrected.shape) != 1:
            raise ValueError(f'Expecting vector of length {x[to_be_corrected].shape[0]}, got {xi_corrected.shape}')

        # Correct design vectors and return activeness information
        x[to_be_corrected, :] = self.x_valid[xi_corrected, :]
        is_active[to_be_corrected, :] = self.is_active_valid[xi_corrected, :]

    def get_canonical_idx(self, x: np.ndarray) -> np.ndarray:
        """Returns a vector specifying for each vector the corresponding valid design vector if the vector is also
        canonical or -1 if not the case."""

        x_canonical_map = self._x_canonical_map
        is_discrete_mask = self.is_discrete_mask

        canonical_idx = -np.ones(x.shape[0], dtype=int)
        for i, xi in enumerate(x):
            ix_canonical = x_canonical_map.get(tuple(xi[is_discrete_mask]))
            if ix_canonical is not None:
                canonical_idx[i] = ix_canonical

        return canonical_idx

    def get_valid_idx(self, x: np.ndarray) -> np.ndarray:
        """Returns a vector specifying for each vector the corresponding valid design vector idx or -1 if not found."""
        valid_idx = -np.ones(x.shape[0], dtype=int)
        for i, xi in enumerate(x):
            ix_valid = self._get_valid_idx_single(xi)
            if ix_valid is not None:
                valid_idx[i] = ix_valid
        return valid_idx

    def _get_valid_idx_single(self, xi: np.ndarray) -> Optional[int]:
        """Returns a valid design vector index for a given design vector, or None if not found"""
        is_discrete_mask = self.is_discrete_mask

        # Check if vector is canonical
        x_canonical_map = self._x_canonical_map
        ix_canonical = x_canonical_map.get(tuple(xi[is_discrete_mask]))
        if ix_canonical is not None:
            return ix_canonical

        x_valid, is_active_valid = self.x_valid, self.is_active_valid
        matched_dv_idx = np.arange(x_valid.shape[0])
        x_valid_matched, is_active_valid_matched = x_valid, is_active_valid
        for i, is_discrete in enumerate(is_discrete_mask):
            # Ignore continuous vars
            if not is_discrete:
                continue

            # Match active valid x to value or inactive valid x
            is_active_valid_i = is_active_valid_matched[:, i]
            matched = (is_active_valid_i & (x_valid_matched[:, i] == xi[i])) | (~is_active_valid_i)

            # Select vectors and check if there are any vectors left to choose from
            matched_dv_idx = matched_dv_idx[matched]
            if len(matched_dv_idx) == 0:
                return
            x_valid_matched = x_valid_matched[matched, :]
            is_active_valid_matched = is_active_valid_matched[matched, :]

        return matched_dv_idx[0]

    def _get_corrected_x_idx(self, x: np.ndarray) -> np.ndarray:
        """
        Return for each vector in x (n x nx) the valid discrete vector index.
        Design vectors may be valid, however canonical vectors are never asked to be corrected.
        """
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}(correct_valid_x={self.correct_valid_x}, ' \
               f'random_if_multiple={self._random_if_multiple})'


class AnyEagerCorrector(EagerCorrectorBase):
    """
    Eager corrector that chooses a random valid design vector if random_if_multiple, otherwise selects the first.
    """

    def _get_corrected_x_idx(self, x: np.ndarray) -> np.ndarray:
        if self._random_if_multiple:
            return np.random.randint(0, self._x_valid.shape[0], x.shape[0])
        return np.zeros((x.shape[0],), dtype=int)


class GreedyEagerCorrector(EagerCorrectorBase):
    """
    Eager corrector that corrects design variables one-by-one, starting from the left.
    """

    def _get_corrected_x_idx(self, x: np.ndarray) -> np.ndarray:
        return np.array([self._get_corrected_x_idx_i(xi) for xi in x], dtype=int)

    def _get_corrected_x_idx_i(self, xi: np.ndarray) -> np.ndarray:
        x_valid, is_active_valid = self.x_valid, self.is_active_valid

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


class ClosestEagerCorrector(EagerCorrectorBase):
    """
    Eager corrector that corrects design vectors by matching them to the closest available canonical
    design vector, as measured by the Manhattan or Euclidean distance.
    Optionally distances are weighted to prefer changes on the right side of the design vectors.
    """

    def __init__(self, design_space: ArchDesignSpace, euclidean=False, correct_valid_x: bool = None,
                 random_if_multiple: bool = None):
        self.euclidean = euclidean
        super().__init__(design_space, correct_valid_x=correct_valid_x, random_if_multiple=random_if_multiple)

    def _get_corrected_x_idx(self, x: np.ndarray) -> np.ndarray:
        # Calculate distances from provided design vectors to canonical design vectors
        x_valid, is_active_valid = self.x_valid, self.is_active_valid
        is_discrete_mask = self.is_discrete_mask
        x_valid_discrete = x_valid[:, is_discrete_mask]

        metric = 'euclidean' if self.euclidean else 'cityblock'
        weights = np.linspace(1.1, 1, x_valid_discrete.shape[1])
        x_dist = distance.cdist(x[:, is_discrete_mask], x_valid_discrete, metric=metric, w=weights)

        xi_canonical = np.zeros((x.shape[0],), dtype=int)
        for i, xi in enumerate(x):
            # Select vector with minimum distance
            min_dist_idx, = np.where(x_dist[i, :] == np.min(x_dist[i, :]))

            if len(min_dist_idx) > 1 and self._random_if_multiple:
                xi_canonical[i] = np.random.choice(min_dist_idx)
            else:
                xi_canonical[i] = min_dist_idx[0]
        return xi_canonical

    def __repr__(self):
        return f'{self.__class__.__name__}(correct_valid_x={self.correct_valid_x}, ' \
               f'random_if_multiple={self._random_if_multiple}, euclidean={self.euclidean})'


IsValidFuncType = Callable[[np.ndarray], Optional[np.ndarray]]


class LazyCorrectorBase(CorrectorBase):
    """
    Corrector that does not have access to the list of all valid discrete design vectors.
    """

    def __init__(self, design_space: ArchDesignSpace, is_valid_func: IsValidFuncType = None, correct_valid_x=None):
        self.is_valid_func = is_valid_func
        super().__init__(design_space, correct_valid_x=correct_valid_x)

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
        correct_valid_x = self.correct_valid_x
        for i, xi in enumerate(x):
            # Check if the vector is canonical: no need to correct if this is already the case
            is_active_i, is_canonical = self.is_canonical(xi)
            if is_active_i is not None:
                if not correct_valid_x or (correct_valid_x and is_canonical):
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
        is_active_i = self.is_valid(xi)
        if is_active_i is None:
            return None, False

        # Check if inactive discrete variables are imputed
        is_canonical = self._is_canonical_inactive(xi, is_active_i)
        return is_active_i, is_canonical

    def is_valid(self, xi: np.ndarray) -> Optional[np.ndarray]:
        """
        Function that returns whether a given single design vector x (of length nx) is valid.
        If valid, returns the activeness vector, otherwise None.
        """
        if len(xi.shape) != 1:
            raise ValueError(f'Expecting vector of length nx, got {xi.shape}')

        is_active_i = self._is_valid(xi)
        if is_active_i is None:
            return

        if is_active_i.shape != xi.shape:
            raise ValueError(f'Expecting return vector of length {xi.shape[0]}, got {is_active_i.shape}')
        return is_active_i

    def _is_valid(self, xi: np.ndarray) -> Optional[np.ndarray]:
        """
        Function that returns whether a given single design vector x (of length nx) is valid.
        If valid, the function should return the activeness vector, otherwise None.
        """
        if self.is_valid_func is None:
            raise RuntimeError('Either provide is_valid_func or override _is_valid!')
        return self.is_valid_func(xi)

    def _correct_single_x(self, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct a single design vector and return the corrected vector and associated activeness information.
        Should generate possible design vectors and use is_valid to check whether generated vectors are valid and get
        activeness information.
        """
        correct_valid_x = self.correct_valid_x

        n_try = 0
        for xi_try in self._generate_x_try(xi):
            # If we originally want to correct also valid vectors, we are looking for a generated canonical vector
            if correct_valid_x:
                xi_active, is_canonical = self.is_canonical(xi_try)
                if not is_canonical:
                    xi_active = None
            else:
                xi_active = self.is_valid(xi_try)

            if xi_active is not None:
                return xi_try, xi_active
            n_try += 1

        raise RuntimeError(f'No valid design vector found after trying {n_try} times!')

    def _generate_x_try(self, xi: np.ndarray) -> Generator[np.ndarray, None, None]:
        """
        Generate design vectors to try whether they are valid.
        """
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}(correct_valid_x={self.correct_valid_x})'


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
    n_try_max = 10000

    def _generate_x_try(self, xi: np.ndarray) -> Generator[np.ndarray, None, None]:
        x_opts = self.x_opts
        for _ in range(self.n_try_max):
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
            # Generate all delta vectors
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
        return f'{self.__class__.__name__}(correct_valid_x={self.correct_valid_x}, by_dist={self.by_dist}{euc_str})'
