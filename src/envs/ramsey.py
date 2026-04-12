import numpy as np
from itertools import combinations
from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import DenseTokenizer
from src.utils import bool_flag

class RamseyDataPoint(DataPoint):
    """
    Represents a red-blue coloring of a graph of a complete graph of N.
    """
    R = 5
    S = 5

    def __init__(self, N, init=False):
        self.N = N
        self.r = self.__class__.R
        self.s = self.__class__.S
        self.data = np.zeros((N, N), dtype=np.uint8)
        self.violations = []
        if init:
            pass
            self._random_coloring()
            self.calc_features()
            self.calc_score()

    def _random_coloring(self):
        for i in range(self.N):
            for j in range(i + 1, self.N):
                c = np.random.randint(0, 2)
                self.data[i][j] = c
                self.data[j][i] = c

    def calc_score(self):
        """
        -x => x monochromatic cliques
        0 => ramsey-avoiding coloring
        """
        self._compute_violations()
        self.score = -len(self.violations)

    def _compute_violations(self):
        """
        TODO this can be optimized when r == s to only check that all colors in a clique are the same
        """
        self.violations = []
        # check for red (0) cliques of size r
        for clique in combinations(range(self.N), self.r):
            if all(self.data[i][j] == 0 for i, j in combinations(clique, 2)):
                self.violations.append(('red', clique))

        # check for blue (1) cliques of size s
        for clique in combinations(range(self.N), self.s):
            if all(self.data[i][j] == 1 for i, j in combinations(clique, 2)):
                self.violations.append(('blue', clique))

    def calc_features(self):
        """
        only encode upper triangle
        """
        w = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                w.append(self.data[i][j])
        self.features = ",".join(map(str, w))

    def local_search(self, improve_with_local_search=True, max_iter=1000):
        """
        rn not doing improvement, could do a greedy step like picking the edge to flip that appears in the most violations
        """
        self._compute_violations()

        for iters in range(max_iter):
            if not self.violations: break
            color, clique = self.violations[np.random.randint(len(self.violations))]
            edges = list(combinations(clique, 2))
            i, j = edges[np.random.randint(len(edges))]
            self.data[i][j] = 1 - self.data[i][j]
            self.data[j][i] = self.data[i][j]

            self._compute_violations()

        self.calc_features()
        self.calc_score()

    @classmethod
    def _update_class_params(cls, pars):
        pass

    @classmethod
    def _save_class_params(cls):
        pass

class RamseyEnvironment(BaseEnvironment):
    k = 2
    are_coordinates_symmetric = True
    data_class = RamseyDataPoint

    def __init__(self, params):
        super().__init__(params)
        self.data_class.R = params.r
        self.data_class.S = params.s
        self.tokenizer = DenseTokenizer(
            self.data_class, params.N, self.k, self.are_coordinates_symmetric,
            self.SPECIAL_SYMBOLS, pow2base=params.pow2base
        )

    @staticmethod
    def register_args(parser):
        parser.add_argument("--N", type=int, default=43, help="Number of vertices in complete graph")
        parser.add_argument("--r", type=int, default=5, help="Red clique size to avoid")
        parser.add_argument("--s", type=int, default=5, help="Blue clique size to avoid")
        parser.add_argument("--pow2base", type=int, default=1, help="Bits per token for adjacency encoding")
