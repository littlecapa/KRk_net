import chess
import numpy as np
import logging
from chess_utils.chess_lib import turn_and_en_passant_square_bits_to_str, turn_and_en_passant_square_to_bits, substitute_piece

# Bitboard representation of the chess board
# currently not used!

class Bitboard:
    def __init__(self, board = None, fen = None):
        if board == None:
            if fen == None:
                self.setup_empty_bitboard()
            else:
                self.setup_from_fen(fen)
        else:
            self.setup_from_chess_board(board)
        
    def setup_empty_bitboard(self):
        self.bitboard = np.zeros((13, 8, 8), dtype=bool)

    def setup_from_chess_board(self, board):
        self.setup_empty_bitboard()
        # Iterate over the board and update the bitboard
        for rank in range(8):
            for file in range(8):
                #square = chess.square(file, 7 - rank)
                square = chess.square(file, rank)
                piece = board.piece_at(square)

                # piece_type: An enumeration representing the type of the chess piece (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, or KING).

                if piece is not None:
                    piece_type = piece.piece_type - 1
                    if piece.color == chess.BLACK:
                        piece_type += 6
                    self.bitboard[piece_type][rank][file] = True

        # Encode the castling rights
        if board.castling_rights & chess.BB_H1:
            self.bitboard[12][0][0] = True
        if board.castling_rights & chess.BB_A1:
            self.bitboard[12][0][1] = True
        if board.castling_rights & chess.BB_H8:
            self.bitboard[12][0][2] = True
        if board.castling_rights & chess.BB_A8:
            self.bitboard[12][0][3] = True

        # Encode turn and enp passant
        self.bitboard[12][1] = turn_and_en_passant_square_to_bits(board)

    def setup_from_fen(self, fen):
        # Set up the bitboard representation from a FEN string
        board = chess.Board(fen)
        self.setup_from_chess_board(board)
        
    def __repr__(self):
        blank_string = " " * 8
        board_string = [blank_string] * 8
        for rank in range(8):
            for file in range(8):
                for piece_type in range(12):
                    #
                    # For the Optics: Rank 0 will be displayed at Bottom (7-rank)
                    #
                    if self.bitboard[piece_type][7-rank][file]:
                        board_string[rank] = substitute_piece(board_string[7-rank], file, piece_type)
        extra_string = turn_and_en_passant_square_bits_to_str(self.bitboard[12][1][0:8]) + " "
        
        if self.bitboard[12][0][0]:
            extra_string += "K"
        if self.bitboard[12][0][1]:
            extra_string += "Q"
        if self.bitboard[12][0][2]:
            extra_string += "k"
        if self.bitboard[12][0][3]:
            extra_string += "q"
        
        return_string = ""
        for i in range(8):
            return_string += board_string[i] +"\n"
        return_string += extra_string
        return return_string
    
    def get_bitboard(self):
        return self.bitboard
    
    def set_bitboard(self, bitboard):
        self.bitboard = bitboard

    def get_13xint64_vector(self):
        pass

    def get_13_64_bool_vector(self):
        return self.bitboard.reshape(-1)
    
    def set_13_64_bool_vector(self, bit_vector):
        self.bitboard = bit_vector.reshape((13, 8, 8))


