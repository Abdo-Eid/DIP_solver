from DIP.cv import LBP
from DIP.helper import display_matrices, make_random

matrix = [
    [ 13, 10, 10],
    [ 11, 11, 18],
    [ 19, 9, 12]
]

LBP_matrix = LBP(matrix)

display_matrices([LBP_matrix])