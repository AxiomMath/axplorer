from itertools import combinations
from src.envs.ramsey import RamseyDataPoint


def setup_function():
    RamseyDataPoint.R = 3
    RamseyDataPoint.S = 3


def test_init_random():
    dp = RamseyDataPoint(N=10, init=True)
    assert dp.data.shape == (10, 10)
    # symmetric
    assert (dp.data == dp.data.T).all()
    # diagonal is 0
    assert all(dp.data[i][i] == 0 for i in range(10))
    # score exists
    assert dp.score <= 0


def test_violations():
    dp = RamseyDataPoint(N=10, init=True)
    # independently count monochromatic triangles
    count = 0
    for clique in combinations(range(10), 3):
        edges = [dp.data[i][j] for i, j in combinations(clique, 2)]
        if len(set(edges)) == 1:
            count += 1
    assert len(dp.violations) == count


def test_known_valid_k4():
    """R(3,3)=6, so a valid 2-coloring of K4 exists"""
    dp = RamseyDataPoint(N=4, init=False)
    dp.data[0][1] = dp.data[1][0] = 0
    dp.data[1][2] = dp.data[2][1] = 0
    dp.data[2][3] = dp.data[3][2] = 0
    dp.data[3][0] = dp.data[0][3] = 0
    dp.data[0][2] = dp.data[2][0] = 1
    dp.data[1][3] = dp.data[3][1] = 1
    dp._compute_violations()
    dp.calc_score()
    assert dp.score == 0
    assert len(dp.violations) == 0


def test_known_invalid_k6():
    """R(3,3)=6, so every 2-coloring of K6 has a monochromatic K3"""
    dp = RamseyDataPoint(N=6, init=True)
    assert len(dp.violations) > 0
