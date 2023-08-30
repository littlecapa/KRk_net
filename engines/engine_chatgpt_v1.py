import chess

class ChessEngine_v1:

    def __init__(self, search_depth, initial_position):
        self.search_depth = search_depth
        self.initial_position = initial_position

    def evaluate_position(self, position):
        material_score = 0
        for piece in position.pieces:
            material_score += piece.piece_value()

        if position.is_checkmate():
            if position.turn:
                return -float('inf')
            else:
                return float('inf')

        if position.is_stalemate():
            return 0

        return material_score


    def get_best_move(self, position):
        best_move = None
        best_score = -float('inf')
        for move in position.legal_moves:
            new_position = position.copy()
            new_position.make_move(move)
            score = self.evaluate_position(new_position)
            if score > best_score:
                best_move = move
                best_score = score
        return best_move

