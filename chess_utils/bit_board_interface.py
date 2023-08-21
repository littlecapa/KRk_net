from chess_utils.bit_board import Bitboard
import logging
from chess_utils.chess_lib import turn_and_en_passant_square_bits_to_str, turn_and_en_passant_square_to_bits, substitute_piece

# Bitboard representation of the chess board
# currently not used!


class Bitboard_Interface:
    def __init__(self, board = None, fen = None):
        self.bb = Bitboard(board, fen)

    def setup_from_chess_board(self, board):
        self.bb.setup_from_chess_board(board)

    def setup_from_fen(self, fen):
        self.bb.setup_from_fen(fen)

    def get_13_64_bool_vector(self):
        return self.bb.get_13_64_bool_vector()