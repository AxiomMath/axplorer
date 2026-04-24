import numpy as np

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import Tokenizer


class OrbitalDataPoint(DataPoint):
    p = 5
    _zeta = None

    def __init__(self, N, init=False):
        super().__init__()
        self.N = N
        self.sequence = np.empty(0, dtype=np.int8)
        if init:
            np.random.seed(None)
            self.sequence = np.random.randint(0, self.p, self.N).astype(np.int8)
            self.calc_features()
            self.calc_score()

    def calc_score(self):
        angles = 2j * np.pi * np.arange(self.p) / self.p
        roots = np.exp(angles).astype(np.complex64)
        path_sum = np.sum(roots[self.sequence])
        self.score = 1.0 / (np.abs(path_sum) + 1e-6)
        return self.score

    def calc_features(self):
        self.features = ",".join(map(str, self.sequence.tolist()))

    @classmethod
    def _update_class_params(cls, pars):
        cls.p = pars

    @classmethod
    def _save_class_params(cls):
        return cls.p


class OrbitalTokenizer(Tokenizer):
    def __init__(self, dataclass, N, p, extra_symbols):
        self.dataclass = dataclass
        self.N = N
        self.p = p
        self.extra_symbols = extra_symbols

        self.stoi, self.itos = {}, {}
        for idx in range(p):
            self.stoi[idx] = idx
            self.itos[idx] = idx
        len1 = len(self.stoi)
        for jdx, el in enumerate(extra_symbols):
            self.stoi[el] = len1 + jdx
            self.itos[len1 + jdx] = el

    def encode(self, datapoint_to_encode):
        w = np.empty(datapoint_to_encode.sequence.size + 2, dtype=np.int32)
        w[0] = self.stoi["BOS"]
        w[1:-1] = datapoint_to_encode.sequence
        w[-1] = self.stoi["EOS"]
        return w

    def decode(self, token_seq_to_decode):
        # remove the first token because it's always BOS
        token_seq_to_decode = token_seq_to_decode[1:]
        try:
            sequence = []
            for token in token_seq_to_decode:
                el = self.itos[int(token)]
                if el in self.extra_symbols:
                    break
                sequence.append(el)
            if len(sequence) != self.N:
                return None
            datapoint = self.dataclass(N=self.N)
            datapoint.sequence = np.array(sequence, dtype=np.int32)
            return datapoint
        except:
            return None


class OrbitalEnvironment(BaseEnvironment):
    data_class = OrbitalDataPoint

    def __init__(self, params):
        super().__init__(params)
        self.data_class.p = params.p
        self.tokenizer = OrbitalTokenizer(self.data_class, params.N, params.p, self.SPECIAL_SYMBOLS)

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument("--N", type=int, default=20, help="Sequence length")
        parser.add_argument("--p", type=int, default=5, help="Alphabet size (number of p-th roots of unity)")
