import copy

class OthelloLogic:
    SIZE = 8
    DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    @staticmethod
    def get_flips(board, player_id, x, y):
        """Retorna a lista de peças que seriam viradas."""
        if x < 0 or x >= 8 or y < 0 or y >= 8 or board[y][x] != 0:
            return []

        opponent = 3 - player_id
        flips = []

        for dx, dy in OthelloLogic.DIRECTIONS:
            nx, ny = x + dx, y + dy
            temp_flips = []

            while 0 <= nx < 8 and 0 <= ny < 8 and board[ny][nx] == opponent:
                temp_flips.append((nx, ny))
                nx += dx
                ny += dy

            if 0 <= nx < 8 and 0 <= ny < 8 and board[ny][nx] == player_id:
                flips.extend(temp_flips)
        return flips

    @staticmethod
    def simulate_move(board, player_id, x, y):
        """Retorna uma CÓPIA do tabuleiro após a jogada."""
        flips = OthelloLogic.get_flips(board, player_id, x, y)
        if not flips:
            return None # Jogada inválida
        
        # Criamos uma cópia profunda para não estragar o tabuleiro original
        new_board = copy.deepcopy(board)
        new_board[y][x] = player_id
        for fx, fy in flips:
            new_board[fy][fx] = player_id
            
        return new_board

    @staticmethod
    def get_valid_moves(board, player_id):
        """Retorna lista de [x, y] válidos."""
        actions = []
        for y in range(8):
            for x in range(8):
                if len(OthelloLogic.get_flips(board, player_id, x, y)) > 0:
                    actions.append([x, y])
        return actions