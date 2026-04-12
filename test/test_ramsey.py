from itertools import combinations
from src.envs.ramsey import RamseyEnvironment, RamseyDataPoint


def setup_function():
    RamseyDataPoint.R = 3
    RamseyDataPoint.S = 3


def max_violations(N, r, s):
    return len(list(combinations(range(N), r))) + len(list(combinations(range(N), s)))


def test_maximum_score():
    N = 6
    r = 3
    s = 3

    print(
        f"the maximum score on (N = {N}, r = {r}, s = {s}) = {max_violations(N, r, s)}"
    )


def test_init_random():
    dp = RamseyDataPoint(N=10, init=True)
    assert dp.data.shape == (10, 10)
    # symmetric
    assert (dp.data == dp.data.T).all()
    # diagonal is 0
    assert all(dp.data[i][i] == 0 for i in range(10))
    # score exists
    assert dp.score >= 0


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
    assert dp.score == max_violations(4, 3, 3)
    assert len(dp.violations) == 0


def test_known_invalid_k6():
    """R(3,3)=6, so every 2-coloring of K6 has a monochromatic K3"""
    dp = RamseyDataPoint(N=6, init=True)
    assert len(dp.violations) > 0
    assert dp.score < max_violations(6, 3, 3)


def test_tokenizer():
    from src.envs.ramsey import RamseyEnvironment, RamseyDataPoint
    from src.envs.tokenizers import DenseTokenizer

    RamseyDataPoint.R = 3
    RamseyDataPoint.S = 3
    dp = RamseyDataPoint(N=6, init=True)

    SPECIAL_SYMBOLS = {"BOS": 0, "EOS": 1, "PAD": 2}
    tok = DenseTokenizer(RamseyDataPoint, 6, 2, True, SPECIAL_SYMBOLS, pow2base=1)

    encoded = tok.encode(dp)
    print(f"Encoded: {encoded}")
    print(f"Encoded length: {len(encoded)}")

    decoded = tok.decode(encoded)
    decoded.calc_features()
    decoded.calc_score()
    print(f"Decoded score: {decoded.score}")
    print(f"Round-trip match: {(decoded.data == dp.data).all()}")
