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

    def simulate_move(board, player_id, x, y):
        flips = OthelloLogic.get_flips(board, player_id, x, y)
        if not flips:
            return None

        new_board = [row[:] for row in board]

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

    @staticmethod
    def evaluate_board(board, player_id):
        opponent = 3 - player_id
        score = 0
        weights = [
            [100, -20, 10, 5, 5, 10, -20, 100],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [10, -2, 5, 1, 1, 5, -2, 10],
            [5, -2, 1, 0, 0, 1, -2, 5],
            [5, -2, 1, 0, 0, 1, -2, 5],
            [10, -2, 5, 1, 1, 5, -2, 10],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [100, -20, 10, 5, 5, 10, -20, 100],
        ]

        for y in range(8):
            for x in range(8):
                if board[y][x] == player_id:
                    score += weights[y][x]
                elif board[y][x] == opponent:
                    score -= weights[y][x]
        return score
