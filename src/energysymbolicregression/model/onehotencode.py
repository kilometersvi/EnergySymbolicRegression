import numpy as np

class OneHotEncoder:
    def __init__(self, chars):
        self.chars = chars
        self.num_syms = len(chars)

    def encode(self, s):
        encoded = np.zeros((len(s), self.num_syms))
        for i, c in enumerate(s):
            encoded[i, self.chars.index(c)] = 1
        return encoded

    def decode(self, encoded):
        decoded = ""
        for row in encoded:
            idx = np.argmax(row)
            decoded += self.chars[idx]
        return decoded