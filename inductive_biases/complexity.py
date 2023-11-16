import numpy as np


def lempel_ziv_complexity(sequence):
    r"""Manual implementation of the Lempel-Ziv complexity.

    It is defined as the number of different substrings encountered as the stream is viewed from begining to the end.
    As an example:

    >>> s = '1001111011000010'
    >>> lempel_ziv_complexity(s)  # 1 / 0 / 01 / 11 / 10 / 110 / 00 / 010
    8

    Marking in the different substrings the sequence complexity :math:`\mathrm{Lempel-Ziv}(s) = 8`: :math:`s = 1 / 0 / 01 / 11 / 10 / 110 / 00 / 010`.

    - See the page https://en.wikipedia.org/wiki/Lempel-Ziv_complexity for more details.


    Other examples:

    >>> lempel_ziv_complexity('1010101010101010')  # 1, 0, 10, 101, 01, 010, 1010
    7
    >>> lempel_ziv_complexity('1001111011000010000010')  # 1, 0, 01, 11, 10, 110, 00, 010, 000
    9
    >>> lempel_ziv_complexity('100111101100001000001010')  # 1, 0, 01, 11, 10, 110, 00, 010, 000, 0101
    10

    - Note: it is faster to give the sequence as a string of characters, like `'10001001'`, instead of a list or a numpy array.
    - Note: see this notebook for more details, comparison, benchmarks and experiments: https://Nbviewer.Jupyter.org/github/Naereen/Lempel-Ziv_Complexity/Short_study_of_the_Lempel-Ziv_complexity.ipynb
    - Note: there is also a Cython-powered version, for speedup, see :download:`lempel_ziv_complexity_cython.pyx`.
    """
    sub_strings = set()
    n = len(sequence)

    ind = 0
    inc = 1
    while True:
        if ind + inc > len(sequence):
            break
        sub_str = tuple(sequence[ind : ind + inc])
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    return len(sub_strings)


def lzw_complexity(sequence: list):
    unique_values = list(set(sequence))

    # if len(unique_values) == 1:
    #     return 1  # * np.log2(len(sequence))

    dictionary = dict()
    for i, v in enumerate(unique_values):
        dictionary[(v,)] = i

    s = 0
    subsequence = (sequence[s],)
    for e in range(1, len(sequence)):
        print(s, e)
        subsequence += (sequence[e],)
        if subsequence not in dictionary:
            dictionary[subsequence] = len(dictionary)
            s = e
            subsequence = (sequence[s],)

    complexity = len(dictionary)
    return complexity


if __name__ == "__main__":
    print(lempel_ziv_complexity([1, 2, 3, 1, 2, 4, 5, 1, 2, 3, 4, 5]))
    print(lzw_complexity([1, 2, 3, 1, 2, 4, 5, 1, 2, 3, 4, 5]))
    print(lzw_complexity([5,] * 1024))
    print(lzw_complexity(list("abcabcasdabcd")))
