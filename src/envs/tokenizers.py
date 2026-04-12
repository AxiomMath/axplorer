import math
from abc import ABC, abstractmethod
from itertools import combinations, permutations, product

import numpy as np


def generate_index_tuples(N, k, are_coordinates_symmetric):
    if k == 1:
        yield from range(N)
    else:
        if are_coordinates_symmetric:
            yield from combinations(range(N), k)
        else:
            yield from product(range(N), repeat=k)


def count_index_tuples(N, k, are_coordinates_symmetric):
    if are_coordinates_symmetric:
        return math.comb(N, k)
    else:
        return N**k


class Tokenizer(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """

    def __init__(self):
        self.dataclass = None

    @abstractmethod
    def encode(self, val):
        pass

    @abstractmethod
    def decode(self, lst):
        pass

    def decode_batch(self, data, pars=None):
        """
        Worker function for detokenizing a batch of data
        """
        out = []
        if pars is not None:
            self.dataclass._update_class_params(pars)
        for _, lst in enumerate(data):
            l = self.decode(lst)
            if l is not None:
                out.append(l)
        return out


class SparseTokenizerSingleInteger(Tokenizer):
    def __init__(self, dataclass, N, k, are_coordinates_symmetric, extra_symbols, encoding_augmentation=None):
        self.dataclass = dataclass
        self.N = N
        self.k = k
        self.are_coordinates_symmetric = are_coordinates_symmetric
        self.extra_symbols = extra_symbols

        self.encoding_augmentation = encoding_augmentation
        self.stoi, self.itos = {}, {}

        for idx, el in enumerate(generate_index_tuples(N, k, are_coordinates_symmetric)):
            self.stoi[el] = idx
            self.itos[idx] = el

        len1 = len(self.stoi)
        for jdx, el in enumerate(extra_symbols):
            self.stoi[el] = len1 + jdx
            self.itos[len1 + jdx] = el

    def encode(self, datapoint_to_encode):
        if self.encoding_augmentation:
            data = self.encoding_augmentation(datapoint_to_encode.data)
        else:
            data = datapoint_to_encode.data

        coordinates = []
        for el in generate_index_tuples(self.N, self.k, self.are_coordinates_symmetric):
            if data[el] == 1:
                coordinates.append(el)

        w = []
        w.append(self.stoi["BOS"])
        for el in coordinates:
            w.append(self.stoi[el])
        w.append(self.stoi["EOS"])
        return np.array(w, dtype=np.int32)

    def decode(self, token_seq_to_decode):
        # remove the first token because it's always BOS
        token_seq_to_decode = token_seq_to_decode[1:]
        try:
            datapoint = self.dataclass(N=self.N)
            for token in token_seq_to_decode:
                el = self.itos[token]
                if el in self.extra_symbols:
                    break
                if self.are_coordinates_symmetric:
                    if len(set(el)) != len(el):
                        return None
                    for permutation in permutations(el):
                        datapoint.data[permutation] = 1
                else:
                    datapoint.data[el] = 1
            return datapoint
        except:
            return None


class SparseTokenizerSequenceKTokens(Tokenizer):
    def __init__(self, dataclass, N, k, are_coordinates_symmetric, extra_symbols, encoding_augmentation=None):
        self.dataclass = dataclass
        self.N = N
        self.k = k
        self.are_coordinates_symmetric = are_coordinates_symmetric

        self.encoding_augmentation = encoding_augmentation
        self.stoi, self.itos = {}, {}

        for idx in range(N):
            self.stoi[idx] = idx
            self.itos[idx] = idx

        len1 = len(self.stoi)
        for jdx, el in enumerate(extra_symbols):
            self.stoi[el] = len1 + jdx
            self.itos[len1 + jdx] = el

    def encode(self, datapoint_to_encode):
        if self.encoding_augmentation:
            data = self.encoding_augmentation(datapoint_to_encode.data)
        else:
            data = datapoint_to_encode.data

        coordinates = []
        for el in generate_index_tuples(self.N, self.k, self.are_coordinates_symmetric):
            if data[el] == 1:
                coordinates.append(el)

        w = []
        w.append(self.stoi["BOS"])
        for el in coordinates:
            for seq in el:
                w.append(self.stoi[seq])
        w.append(self.stoi["EOS"])
        return np.array(w, dtype=np.int32)

    def decode(self, token_seq_to_decode):
        # remove the first token because it's always BOS
        token_seq_to_decode = token_seq_to_decode[1:]
        try:
            datapoint = self.dataclass(N=self.N)
            for idx in range(0, len(token_seq_to_decode), self.k):
                el = tuple(self.itos[t] for t in token_seq_to_decode[idx : idx + self.k])
                if any(x in self.extra_symbols for x in el):
                    break
                if self.are_coordinates_symmetric:
                    if len(set(el)) != len(el):
                        return None
                    for permutation in permutations(el):
                        datapoint.data[permutation] = 1
                else:
                    datapoint.data[el] = 1
            return datapoint
        except:
            return None


class DenseTokenizer(Tokenizer):
    def __init__(self, dataclass, N, k, are_coordinates_symmetric, extra_symbols, pow2base, encoding_augmentation=None):
        self.dataclass = dataclass
        self.N = N
        self.k = k
        self.are_coordinates_symmetric = are_coordinates_symmetric
        self.pow2base = pow2base
        self.encoding_augmentation = encoding_augmentation
        self.stoi, self.itos = {}, {}
        self.extra_symbols = extra_symbols

        self.expected_elements_in_a_decoded_sequence = math.ceil(count_index_tuples(N, self.k, self.are_coordinates_symmetric) / self.pow2base)

        for idx, el in enumerate(range(2**pow2base)):
            self.stoi[el] = idx
            self.itos[idx] = el
        len1 = len(self.stoi)
        for jdx, el in enumerate(extra_symbols):
            self.stoi[el] = len1 + jdx
            self.itos[len1 + jdx] = el

    def _pack_bits(self, bits):
        tokens = []
        count = 1
        val = 0
        for bit in bits:
            val += count * bit
            count *= 2
            if count == 2**self.pow2base:
                tokens.append(self.stoi[val])
                count = 1
                val = 0
        if count > 1:
            tokens.append(self.stoi[val])
        return tokens

    def _unpack_bits(self, tokens):
        bits = []
        for token in tokens:
            val = token
            for _ in range(self.pow2base):
                bits.append(val % 2)
                val //= 2
        return bits

    def _row_indices(self, row, datapoint_to_encode_N):
        if self.are_coordinates_symmetric:
            return range(row + 1, datapoint_to_encode_N)
        else:
            return range(datapoint_to_encode_N)

    def encode(self, datapoint_to_encode):
        if self.encoding_augmentation:
            data = self.encoding_augmentation(datapoint_to_encode.data)
        else:
            data = datapoint_to_encode.data

        w = []
        bits = (data[el] for el in generate_index_tuples(datapoint_to_encode.N, self.k, self.are_coordinates_symmetric))
        w.append(self.stoi["BOS"])
        w.extend(self._pack_bits(bits))
        w.append(self.stoi["EOS"])
        return np.array(w, dtype=np.int32)

    def decode(self, token_seq_to_decode):
        token_seq_to_decode = token_seq_to_decode[1:]
        new_token_seq_to_decode = []
        for el in token_seq_to_decode:
            el = self.itos[el]
            if el in self.extra_symbols:
                break
            new_token_seq_to_decode.append(el)
        if len(new_token_seq_to_decode) != self.expected_elements_in_a_decoded_sequence:
            return None
        try:
            datapoint = self.dataclass(N=self.N)
            bits = self._unpack_bits(new_token_seq_to_decode)
            for bit, el in zip(bits, generate_index_tuples(self.N, self.k, self.are_coordinates_symmetric)):
                if bit == 1:
                    if self.are_coordinates_symmetric:
                        for permutation in permutations(el):
                            datapoint.data[permutation] = 1
                    else:
                        datapoint.data[el] = 1
            return datapoint
        except:
            return None
