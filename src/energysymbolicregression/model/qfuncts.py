from factories import QFactory, IFactory
from typing import List

    ## char-position rules
def only_one_char_for_each_position(qf: 'QFactory', max_str_len: int, chars: List[str]) -> None:
    #inhibit results from within the same position
    for pos in range(max_str_len):
        for char in chars:
            for char2 in chars:
                if char != char2:
                    qf.qset(pos, char, qf.conf['c'], pos, char2)



## operators
def operator_adjs(qf: 'QFactory', max_str_len: int, chars: List[str]) -> None:
    # Additional rules for operators: penalize if an operator follows another operator (or empty char), penalize if it's the first or last position
    for pos in range(max_str_len):
        for char in qf.sets["operators"] + ['N']:
            if pos < max_str_len - 1:
                for next_char in qf.sets["operators"]:
                    qf.qset(pos, char, qf.conf["ofo"], pos + 1, next_char)
            if char != "N" and (pos == 0 or pos == max_str_len - 1):
                qf.qset(pos, char, qf.conf["ose"])



## ints
def consec_ints(qf: 'QFactory', max_str_len: int, chars: List[str]) -> None:
    # Add inhibitory connections to reduce consecutive integers
    for pos in range(max_str_len - 1):
        for char in qf.sets["numbers"]:
            for next_char in qf.sets["numbers"]:
                qf.qadj(pos, char, qf.conf["nfn"], pos + 1, next_char)



## empty char
def inhibit_empty_chars_following_char(qf: 'QFactory', max_str_len: int, chars: List[str]) -> None:
    # Inhibit empty chars when following non-empty char'
    for n_pos in range(max_str_len-1):
        for char in chars:
            if char != 'N':
                qf.qadj(n_pos, 'N', qf.conf["Nf!N"], n_pos+1, char)

#superceded by ifunc
def adjust_empty_chars_start_end(qf: 'QFactory', max_str_len: int, chars: List[str]) -> None:
    # Excite 1st pos and last pos to be N and not N from all chars
    for other_pos in range(1, max_str_len-1):
        for other_char in chars:
            qf.qadj(0, 'N', qf.conf["Ns"], other_pos, other_char)
            qf.qadj(max_str_len-1, 'N', qf.conf["Ne"], other_pos, other_char)

# superceded by ifunc
def excite_empty_chars_earlier(qf: 'QFactory', max_str_len: int, chars: List[str]) -> None:
    # Excite empty chars in earlier positions from previous empty chars
    qf.qadj(0, 'N', qf.conf["Ns"]*len(chars))
    for n_pos in range(max_str_len-1):
        vadj = qf.conf["NfNd"]*(max_str_len - n_pos)/max_str_len
        qf.qadj(n_pos, 'N', vadj, n_pos + 1, 'N')



## vars
def inhibit_numbers_after_vars(qf: 'QFactory', max_str_len: int, chars: List[str]) -> None:
    # Inhibit numbers following vars
    for pos in range(max_str_len - 1):
        for char in qf.sets["numbers"]:
            qf.qadj(pos, 'x', qf.conf["nfx"], pos + 1, char)

def inhibit_vars_after_vars(qf: 'QFactory', max_str_len: int, chars: List[str]) -> None:
    # Inhibit vars following vars
    for pos in range(max_str_len - 1):
        qf.qadj(pos, 'x', qf.conf["xfx"], pos + 1, 'x')

def excite_x_all_pos(qf: 'QFactory', max_str_len: int, chars: List[str]) -> None:
    # Excite the presence of 'x' in each position for all chars, while inhibiting x from other x's
    for pos_char in range(max_str_len):
        for char in chars:
            for pos_x in range(max_str_len):
                if char != 'x' or (char == 'x' and pos_char == pos_x):
                    qf.qadj(pos_char, char, qf.conf["x"], pos_x, 'x')
                elif pos_char != pos_x:
                    qf.qadj(pos_char, 'x', qf.conf["xm"], pos_x, 'x')



math_qfuncs = [
    only_one_char_for_each_position,
    operator_adjs,
    consec_ints,
    inhibit_empty_chars_following_char,
    adjust_empty_chars_start_end, #
    excite_empty_chars_earlier, #
    inhibit_numbers_after_vars,
    inhibit_vars_after_vars,
    excite_x_all_pos
]


def excite_early_empty_chars(if_: 'IFactory', max_str_len: int, chars: List[str]) -> None:
    for pos in range(max_str_len):
        adj = (max_str_len-pos)/max_str_len
        if_.iadj(pos, 'N', if_.conf['Nsd'] * adj)

def prevent_ending_empty_char(if_: 'IFactory', max_str_len: int, chars: List[str]) -> None:
    if_.iadj(max_str_len-1, 'N', if_.conf['Ne'])

math_ifuncs = [
    #excite_early_empty_chars,
    #prevent_ending_empty_char
]