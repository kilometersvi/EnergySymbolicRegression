import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

class QFactory:
    def __init__(self, max_str_len: int, chars: List[str], conf: Dict[str, float]=None, sets: Dict[str, List[str]]=None):
        self.max_str_len = max_str_len
        self.chars = chars
        self.conf = conf
        self.sets = sets

        self.num_syms = len(chars)
        self.nUnits = max_str_len * self.num_syms

        self.Q = np.zeros((self.nUnits, self.nUnits))

    def _get_idx(self, out_position: int, char: str):
        return self.num_syms * out_position + self.chars.index(char)

    def qget(self, out_position: int = None, char: str = None, out_position2: int = None, char2: str = None):
        if out_position is not None and char is not None:
            idx1 = self._get_idx(out_position, char)
            idx2 = idx1 if out_position2 is None or char2 is None else self._get_idx(out_position2, char2)

            return self.Q[idx1, idx2]
        elif char is None:
            return np.take[self.Q, [self._get_idx(out_position, c) for c in self.chars]]#self.Q[self.num_syms*out_position:(self.num_syms+1)*out_position,:]
        elif out_position is None:
            return np.take[self.Q, [self._get_idx(o, char) for o in range(self.max_str_len)]]
        else:
            return self.Q

    def qset(self, out_position: int, char: str, v: float, out_position2: int = None, char2: str = None, r: bool = True):
        idx1 = self._get_idx(out_position, char)

        # if out_position2 and char2 aren't specified, default to setting the self-connection
        idx2 = idx1 if out_position2 is None or char2 is None else self._get_idx(out_position2, char2)

        self.Q[idx1, idx2] = v

        if r:
           self.Q[idx2, idx1] = v

    def qadj(self, out_position: int, char: str, v: float, out_position2: int = None, char2: str = None, r: bool = True):
        idx1 = self._get_idx(out_position, char)
        idx2 = idx1 if out_position2 is None or char2 is None else self._get_idx(out_position2, char2)
        self.Q[idx1, idx2] += v

        if r and (out_position2 is not None) and (char2 is not None):
           self.Q[idx2, idx1] += v

    def averages(self, mean: str = None):
        avg_values = np.zeros((self.num_syms, self.max_str_len))

        for self_pos in range(self.max_str_len):
            for self_char in self.chars:
                self_char_idx = self.chars.index(self_char)

                #print(f'c: {self_pos},{self_char} ({self._get_idx(self_pos, self_char)})')

                sum_inhibition = 0
                for effector_pos in range(self.max_str_len):
                    for effector_char in self.chars:
                        sum_inhibition += self.qget(effector_pos, effector_char, self_pos, self_char)
                avg_values[self_char_idx, self_pos] = sum_inhibition/(self.max_str_len*self.num_syms)

        #average inhibition on characters, despite pos
        if mean=="char":
            return np.mean(avg_values,1)

        #average inhibition on pos, despite character
        elif mean=="pos":
            return np.mean(avg_values,0)

        return avg_values

    def plot(self):
        plt.imshow(self.Q)
        plt.show()

class IFactory:
    def __init__(self, max_str_len: int, chars: List[str], conf: Dict[str, float] = None, sets: Dict[str, List[str]] = None):
        self.max_str_len = max_str_len
        self.chars = chars
        self.conf = conf
        self.sets = sets

        self.num_syms = len(chars)
        self.I = np.zeros((self.max_str_len * self.num_syms, 1))#np.random.rand(self.max_str_len * self.num_syms, 1)

    def _get_idx(self, pos: int, char: str):
        return pos * self.num_syms + self.chars.index(char)

    def iadj(self, pos: int, char: str, value: float):
        self.I[self._get_idx(pos, char)] += value

    def iset(self, pos: int, char: str, value: float):
        self.I[self._get_idx(pos, char)] = value

    def iget(self, pos: int, char: str, value: float):
        return self.I[self._get_idx(pos, char)]

    def plot(self):
        plt.imshow(self.I.reshape((self.max_str_len, self.num_syms)))
        plt.show()