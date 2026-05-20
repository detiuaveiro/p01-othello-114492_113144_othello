import numpy as np
from numba import njit

# 1. Constantes 1D Otimizadas para o Numba
DIRECTIONS = np.array([
    [-1, -1], [-1, 0], [-1, 1], [0, -1], 
    [0, 1], [1, -1], [1, 0], [1, 1]
], dtype=np.int8)

WEIGHTS = np.array([
    100, -20, 10,  5,  5, 10, -20, 100,
    -20, -50, -2, -2, -2, -2, -50, -20,
     10,  -2,  5,  1,  1,  5,  -2,  10,
      5,  -2,  1,  0,  0,  1,  -2,   5,
      5,  -2,  1,  0,  0,  1,  -2,   5,
     10,  -2,  5,  1,  1,  5,  -2,  10,
    -20, -50, -2, -2, -2, -2, -50, -20,
    100, -20, 10,  5,  5, 10, -20, 100
], dtype=np.int32)

# 2. Funções Puras do Numba (Velocidade Máxima)
@njit
def get_flips(board_1d, player_id, x, y):
    """Calcula os flips operando num array 1D."""
    idx = y * 8 + x
    if x < 0 or x >= 8 or y < 0 or y >= 8 or board_1d[idx] != 0:
        return np.empty(0, dtype=np.int8)

    opponent = 3 - player_id
    flips = np.empty(64, dtype=np.int8)
    count = 0

    for d in range(8):
        dx = DIRECTIONS[d, 0]
        dy = DIRECTIONS[d, 1]
        nx, ny = x + dx, y + dy
        
        temp_flips = np.empty(8, dtype=np.int8)
        temp_count = 0

        while 0 <= nx < 8 and 0 <= ny < 8:
            n_idx = ny * 8 + nx
            if board_1d[n_idx] == opponent:
                temp_flips[temp_count] = n_idx
                temp_count += 1
                nx += dx
                ny += dy
            else:
                break

        if 0 <= nx < 8 and 0 <= ny < 8:
            n_idx = ny * 8 + nx
            if board_1d[n_idx] == player_id:
                for i in range(temp_count):
                    flips[count] = temp_flips[i]
                    count += 1

    return flips[:count]

@njit
def simulate_move(board_1d, player_id, x, y):
    """Aplica as mudanças e retorna cópia 1D."""
    flips = get_flips(board_1d, player_id, x, y)
    if len(flips) == 0:
        return np.empty(0, dtype=np.int8) 

    new_board = board_1d.copy()
    new_board[y * 8 + x] = player_id
    for i in range(len(flips)):
        new_board[flips[i]] = player_id

    return new_board

@njit
def get_valid_moves(board_1d, player_id):
    """Devolve um array 2D com [x, y] das jogadas válidas."""
    moves = np.empty((64, 2), dtype=np.int8)
    count = 0
    for y in range(8):
        for x in range(8):
            if len(get_flips(board_1d, player_id, x, y)) > 0:
                moves[count, 0] = x
                moves[count, 1] = y
                count += 1
    return moves[:count]

@njit
def evaluate_board(board_1d, player_id, use_mobility):
    """Itera sobre 64 posições de forma hiper-otimizada."""
    opponent = 3 - player_id
    score = 0

    for i in range(64):
        if board_1d[i] == player_id:
            score += WEIGHTS[i]
        elif board_1d[i] == opponent:
            score -= WEIGHTS[i]

    if use_mobility:
        my_moves = len(get_valid_moves(board_1d, player_id))
        opp_moves = len(get_valid_moves(board_1d, opponent))
        score += 15 * (my_moves - opp_moves)

    return score

# 3. Wrapper (O Tradutor Universal)
class OthelloLogic:
    @staticmethod
    def _to_1d(board_2d):
        """Converte a matriz 2D (lista de listas) num array 1D NumPy para o Numba."""
        return np.array(board_2d, dtype=np.int8).flatten()

    @staticmethod
    def get_flips(board, player_id, x, y):
        """Devolve em formato [x, y] para não partir o código externo."""
        b_1d = OthelloLogic._to_1d(board)
        flip_indices = get_flips(b_1d, player_id, x, y)
        return [[idx % 8, idx // 8] for idx in flip_indices]

    @staticmethod
    def simulate_move(board, player_id, x, y):
        b_1d = OthelloLogic._to_1d(board)
        new_b_1d = simulate_move(b_1d, player_id, x, y)
        if len(new_b_1d) == 0:
            return None
        # Converte de volta para 2D (Lista de listas)
        return new_b_1d.reshape((8, 8)).tolist()

    @staticmethod
    def get_valid_moves(board, player_id):
        b_1d = OthelloLogic._to_1d(board)
        # O Numba já devolve no formato Nx2, só precisamos de converter para lista Python
        return get_valid_moves(b_1d, player_id).tolist()

    @staticmethod
    def evaluate_board(board, player_id, use_mobility=False):
        b_1d = OthelloLogic._to_1d(board)
        return float(evaluate_board(b_1d, player_id, use_mobility))