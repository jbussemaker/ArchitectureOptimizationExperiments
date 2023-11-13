import numpy as np
from typing import *
from pymoo.core.variable import Real, Choice
from sb_arch_opt.design_space import ArchDesignSpace
from arch_opt_exp.md_mo_hier.correction import *


class DummyArchDesignSpace(ArchDesignSpace):

    def __init__(self, x_all: np.ndarray, is_act_all: np.ndarray, is_discrete_mask: np.ndarray = None):
        self.corrector: Optional[CorrectorBase] = None
        self._x_all = x_all
        self._is_act_all = is_act_all

        if is_discrete_mask is None:
            is_discrete_mask = np.ones((x_all.shape[1],), dtype=bool)
        self._is_discrete_mask = is_discrete_mask
        super().__init__()

    def is_explicit(self) -> bool:
        return False

    def _get_variables(self):
        des_vars = []
        for i, is_discrete in enumerate(self._is_discrete_mask):
            if is_discrete:
                des_vars.append(Choice(options=list(sorted(np.unique(self._x_all[:, i])))))
            else:
                des_vars.append(Real(bounds=(0., 1.)))
        return des_vars

    def _is_conditionally_active(self) -> Optional[List[bool]]:
        pass  # Derived from is_active_all

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        self.corrector.correct_x(x, is_active)

    def _quick_sample_discrete_x(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        raise RuntimeError

    def _get_n_valid_discrete(self) -> Optional[int]:
        return self._x_all.shape[0]

    def _get_n_active_cont_mean(self) -> Optional[int]:
        pass

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._x_all, self._is_act_all

    def is_valid(self, xi: np.ndarray) -> Optional[np.ndarray]:
        eager_corr = EagerCorrectorBase(self)
        i_valid = eager_corr.get_valid_idx(np.array([xi]))[0]
        if i_valid == -1:
            return
        _, is_active_valid = eager_corr.x_valid_active
        return is_active_valid[i_valid, :]


def test_corrector():
    x_all = np.array([[0, 0],
                      [1, 0],
                      [0, 1],
                      [1, 1]])
    is_act_all = np.ones(x_all.shape, dtype=bool)
    ds = DummyArchDesignSpace(x_all=x_all, is_act_all=is_act_all)
    assert np.all(ds.is_discrete_mask)

    corr = CorrectorBase(ds)
    assert np.all(corr.is_discrete_mask)
    assert np.all(corr.x_imp_discrete == np.array([0, 0]))

    assert corr._is_canonical_inactive(np.array([1, 1]), np.array([True, True]))
    assert corr._is_canonical_inactive(np.array([1, 0]), np.array([True, False]))
    assert not corr._is_canonical_inactive(np.array([1, 1]), np.array([True, False]))


def test_eager_corrector():
    x_all = np.array([[0, 0],
                      [0, 1],
                      [0, 2],
                      [1, 0],
                      [1, 1],
                      [2, 0]])
    is_act_all = np.ones(x_all.shape, dtype=bool)
    is_act_all[-1, 1] = False
    ds = DummyArchDesignSpace(x_all=x_all, is_act_all=is_act_all)

    corr = EagerCorrectorBase(ds)
    x_try = np.array([[1, 0],  # Canonical, 3
                      [1, 2],  # Invalid
                      [2, 0],  # Canonical, 5
                      [2, 1]])  # Valid, non-canonical, 5

    assert np.all(corr.get_valid_idx(x_try) == np.array([3, -1, 5, 5]))
    assert np.all(corr.get_canonical_idx(x_try) == np.array([3, -1, 5, -1]))


def test_any_eager_corrector():
    x_all = np.array([[0, 0],
                      [0, 1],
                      [0, 2],
                      [1, 0],
                      [1, 1],
                      [2, 0]])
    is_act_all = np.ones(x_all.shape, dtype=bool)
    is_act_all[-1, 1] = False
    ds = DummyArchDesignSpace(x_all=x_all, is_act_all=is_act_all)

    for correct_valid_x in [False, True]:
        for random_if_multiple in [False, True]:
            ds.corrector = corr = \
                AnyEagerCorrector(ds, correct_valid_x=correct_valid_x, random_if_multiple=random_if_multiple)
            assert repr(corr)

            x_corr, is_act_corr = ds.correct_x(np.array([[1, 0],
                                                         [1, 2],
                                                         [2, 0],
                                                         [2, 1]]))

            x_corr_, is_act_corr_ = ds.correct_x(x_corr)
            assert np.all(x_corr == x_corr_)
            assert np.all(is_act_corr == is_act_corr_)

            assert np.all(x_corr[0, :] == np.array([1, 0]))
            if not random_if_multiple:
                assert np.all(x_corr[1, :] == np.array([0, 0]))
            assert np.all(x_corr[2, :] == np.array([2, 0]))
            if correct_valid_x:
                if not random_if_multiple:
                    assert np.all(x_corr[3, :] == np.array([0, 0]))
            else:
                assert np.all(x_corr[3, :] == np.array([2, 0]))


def test_greedy_eager_corrector():
    x_all = np.array([[0, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0],
                      [0, 1, 1],
                      [0, 1, 2],
                      [1, 0, 0],
                      [1, 0, 2],
                      [1, 1, 0],
                      [2, 0, 0]])
    is_act_all = np.ones(x_all.shape, dtype=bool)
    is_act_all[-2, 2] = False
    is_act_all[-1, 1:] = False
    ds = DummyArchDesignSpace(x_all=x_all, is_act_all=is_act_all)

    for correct_valid_x in [False, True]:
        for random_if_multiple in [False, True]:
            ds.corrector = corr = \
                GreedyEagerCorrector(ds, correct_valid_x=correct_valid_x, random_if_multiple=random_if_multiple)
            assert repr(corr)

            x_corr, is_act_corr = ds.correct_x(np.array([[0, 0, 0],
                                                         [0, 0, 2],
                                                         [0, 1, 2],
                                                         [1, 0, 0],
                                                         [1, 0, 1],
                                                         [2, 0, 0],
                                                         [2, 0, 1]]))

            x_corr_, is_act_corr_ = ds.correct_x(x_corr)
            assert np.all(x_corr == x_corr_)
            assert np.all(is_act_corr == is_act_corr_)

            corr_first = np.array([[0, 0, 0],
                                   [0, 0, 1],
                                   [0, 1, 2],
                                   [1, 0, 0],
                                   [1, 0, 0],
                                   [2, 0, 0],
                                   [2, 0, 0]])
            corr_second = corr_first.copy()
            corr_second[4, 2] = 2
            if random_if_multiple:
                assert np.all(x_corr == corr_first) or np.all(x_corr == corr_second)
            else:
                assert np.all(x_corr == corr_first)


def test_closest_eager_corrector():
    x_all = np.array([[0, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0],
                      [0, 1, 1],
                      [0, 1, 2],
                      [1, 0, 0],
                      [1, 0, 2],
                      [1, 1, 0],
                      [2, 0, 0],
                      [2, 1, 3]])
    is_act_all = np.ones(x_all.shape, dtype=bool)
    is_act_all[-3, 2] = False
    is_act_all[-2, 1:] = False
    ds = DummyArchDesignSpace(x_all=x_all, is_act_all=is_act_all)

    for correct_valid_x in [False, True]:
        for random_if_multiple in [False, True]:
            for _ in range(10 if random_if_multiple else 1):
                for euclidean in [False, True]:
                    ds.corrector = corr = ClosestEagerCorrector(
                        ds, euclidean=euclidean, correct_valid_x=correct_valid_x, random_if_multiple=random_if_multiple)
                    assert repr(corr)

                    x_corr, is_act_corr = ds.correct_x(np.array([[0, 0, 0],
                                                                 [0, 0, 3],
                                                                 [1, 0, 1],
                                                                 [1, 1, 1],
                                                                 [1, 1, 2],
                                                                 [2, 0, 2]]))

                    x_corr_, is_act_corr_ = ds.correct_x(x_corr)
                    assert np.all(x_corr == x_corr_)
                    assert np.all(is_act_corr == is_act_corr_)

                    corr_first = np.array([[0, 0, 0],
                                           [0, 0, 1],
                                           [1, 0, 0],
                                           [1, 1, 0],
                                           [1, 1, 0],
                                           [2, 0, 0]])
                    if euclidean:
                        corr_first[1, :] = [0, 1, 2]
                    if correct_valid_x:
                        corr_first[-2, :] = [1, 0, 2]
                        corr_first[-1, :] = [1, 0, 2]
                    corr_second = corr_first.copy()
                    corr_second[2, :] = [1, 0, 2]

                    if random_if_multiple:
                        assert np.all(x_corr == corr_first) or np.all(x_corr == corr_second)
                    else:
                        assert np.all(x_corr == corr_first)


def test_lazy_corrector():
    x_all = np.array([[0, 0],
                      [0, 1],
                      [0, 2],
                      [1, 0],
                      [1, 1],
                      [2, 0]])
    is_act_all = np.ones(x_all.shape, dtype=bool)
    is_act_all[-1, 1] = False
    ds = DummyArchDesignSpace(x_all=x_all, is_act_all=is_act_all)

    corr = LazyCorrectorBase(ds, ds.is_valid)
    assert corr.x_opts == [[0, 1, 2], [0, 1, 2]]

    x_try = np.array([[1, 0],  # Canonical, 3
                      [1, 2],  # Invalid
                      [2, 0],  # Canonical, 5
                      [2, 1]])  # Valid, non-canonical, 5

    is_valid_try = [np.array([True, True]),
                    None,
                    np.array([True, False]),
                    np.array([True, False])]
    is_canon_try = [True, False, True, False]
    for i, xi_try in enumerate(x_try):
        is_valid = corr.is_valid(xi_try)
        is_valid_, is_canon = corr.is_canonical(xi_try)
        if is_valid_try[i] is None:
            assert is_valid is None
            assert is_valid_ is None
            assert not is_canon
        else:
            assert np.all(is_valid == is_valid_try[i])
            assert np.all(is_valid_ == is_valid_try[i])
            assert is_canon == is_canon_try[i]


def test_first_lazy_corrector():
    x_all = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [0, 1, 1],
                      [0, 1, 2],
                      [1, 0, 0],
                      [1, 0, 2],
                      [1, 1, 0],
                      [2, 0, 0],
                      [2, 1, 3]])
    is_act_all = np.ones(x_all.shape, dtype=bool)
    is_act_all[-3, 2] = False
    is_act_all[-2, 1:] = False
    ds = DummyArchDesignSpace(x_all=x_all, is_act_all=is_act_all)

    for correct_valid_x in [False, True]:
        ds.corrector = corr = FirstLazyCorrector(ds, ds.is_valid, correct_valid_x=correct_valid_x)
        assert repr(corr)

        x_corr, is_act_corr = ds.correct_x(np.array([[0, 0, 1],
                                                     [0, 0, 3],
                                                     [1, 0, 1],
                                                     [2, 0, 2]]))

        x_corr_, is_act_corr_ = ds.correct_x(x_corr)
        assert np.all(x_corr == x_corr_)
        assert np.all(is_act_corr == is_act_corr_)

        x_corr_tgt = np.array([[0, 0, 1],
                               [0, 0, 1],
                               [0, 0, 1],
                               [2, 0, 0]])
        if correct_valid_x:
            x_corr_tgt[-1, :] = [0, 0, 1]
        assert np.all(x_corr == x_corr_tgt)


def test_random_lazy_corrector():
    x_all = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [0, 1, 1],
                      [0, 1, 2],
                      [1, 0, 0],
                      [1, 0, 2],
                      [1, 1, 0],
                      [2, 0, 0],
                      [2, 1, 3]])
    is_act_all = np.ones(x_all.shape, dtype=bool)
    is_act_all[-3, 2] = False
    is_act_all[-2, 1:] = False
    ds = DummyArchDesignSpace(x_all=x_all, is_act_all=is_act_all)

    for correct_valid_x in [False, True]:
        for _ in range(10):
            ds.corrector = corr = FirstLazyCorrector(ds, ds.is_valid, correct_valid_x=correct_valid_x)
            assert repr(corr)

            x_corr, is_act_corr = ds.correct_x(np.array([[0, 0, 1],
                                                         [0, 0, 3],
                                                         [1, 0, 1],
                                                         [2, 0, 2]]))

            x_corr_, is_act_corr_ = ds.correct_x(x_corr)
            assert np.all(x_corr == x_corr_)
            assert np.all(is_act_corr == is_act_corr_)

            x_corr_tgt = np.array([[0, 0, 1],
                                   [2, 0, 0]])
            if correct_valid_x:
                assert np.all(x_corr[[0], :] == x_corr_tgt[[0], :])
            else:
                assert np.all(x_corr[[0, 3], :] == x_corr_tgt)


def test_closest_lazy_corrector():
    x_all = np.array([[0, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0],
                      [0, 1, 1],
                      [0, 1, 2],
                      [1, 0, 0],
                      [1, 0, 2],
                      [1, 1, 0],
                      [2, 0, 0],
                      [2, 1, 3]])
    is_act_all = np.ones(x_all.shape, dtype=bool)
    is_act_all[-3, 2] = False
    is_act_all[-2, 1:] = False
    ds = DummyArchDesignSpace(x_all=x_all, is_act_all=is_act_all)

    for correct_valid_x in [False, True]:
        for by_dist in [False, True]:
            for euclidean in [False, True]:
                if not by_dist and euclidean:
                    continue
                ds.corrector = corr = ClosestLazyCorrector(
                    ds, ds.is_valid, by_dist=by_dist, euclidean=euclidean, correct_valid_x=correct_valid_x)
                assert repr(corr)

                x_corr, is_act_corr = ds.correct_x(np.array([[0, 0, 0],
                                                             [0, 0, 3],
                                                             [1, 0, 1],
                                                             [1, 1, 1],
                                                             [1, 1, 2],
                                                             [2, 0, 2]]))

                x_corr_, is_act_corr_ = ds.correct_x(x_corr)
                assert np.all(x_corr == x_corr_)
                assert np.all(is_act_corr == is_act_corr_)

                x_corr_tgt = np.array([[0, 0, 0],
                                       [0, 0, 1],
                                       [1, 0, 0],
                                       [1, 1, 0],
                                       [1, 1, 0],
                                       [2, 0, 0]])
                if euclidean:
                    x_corr_tgt[1, :] = [0, 1, 2]
                if correct_valid_x:
                    if by_dist:
                        x_corr_tgt[-2, :] = [1, 0, 2]
                        if euclidean and not correct_valid_x:
                            x_corr_tgt[-1, :] = [2, 1, 3]
                        else:
                            x_corr_tgt[-1, :] = [1, 0, 2]
                    else:
                        x_corr_tgt[-1, :] = [2, 0, 0]
                assert np.all(x_corr == x_corr_tgt)
