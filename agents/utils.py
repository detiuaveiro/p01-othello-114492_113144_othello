import numpy as np
from numba import njit

# 1. Converter Constantes para Numpy Arrays (O Numba precisa disto)
DIRECTIONS = np.array([
    [-1, -1], [-1, 0], [-1, 1], [0, -1], 
    [0, 1], [1, -1], [1, 0], [1, 1]
], dtype=np.int8)

# Matriz de pesos achatada para 1D (64 posições)
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


import numpy as np

# Pesos Estratégicos Dinâmicos (não usar como pontuação estática)


# Colocamos as lógicas todas soltas fora de classe com @njit 
# porque o Numba trabalha melhor com funções puras do que métodos de classe.

@njit
def get_flips(board, player_id, x, y):
    """Calcula os flips operando num array 1D."""
    idx = y * 8 + x
    # Se a jogada é fora do tabuleiro ou a casa já está ocupada, devolve vazio
    if x < 0 or x >= 8 or y < 0 or y >= 8 or board[idx] != 0:
        return np.empty(0, dtype=np.int8)

    opponent = 3 - player_id
    flips = np.empty(64, dtype=np.int8) # Array temporário para guardar os índices
    count = 0

    for d in range(8):
        dx = DIRECTIONS[d, 0]
        dy = DIRECTIONS[d, 1]
        nx, ny = x + dx, y + dy
        
        temp_flips = np.empty(8, dtype=np.int8)
        temp_count = 0

        while 0 <= nx < 8 and 0 <= ny < 8:
            n_idx = ny * 8 + nx
            if board[n_idx] == opponent:
                temp_flips[temp_count] = n_idx
                temp_count += 1
                nx += dx
                ny += dy
            else:
                break

        if 0 <= nx < 8 and 0 <= ny < 8:
            n_idx = ny * 8 + nx
            if board[n_idx] == player_id:
                for i in range(temp_count):
                    flips[count] = temp_flips[i]
                    count += 1

    return flips[:count]

@njit
def simulate_move(board, player_id, x, y):
    """Cria uma cópia do array 1D e aplica as mudanças."""
    flips = get_flips(board, player_id, x, y)
    if len(flips) == 0:
        # Devolve um array vazio para simular o "None" antigo
        return np.empty(0, dtype=np.int8) 

    # Cópia super rápida no Numpy
    new_board = board.copy()
    new_board[y * 8 + x] = player_id
    for i in range(len(flips)):
        new_board[flips[i]] = player_id

    return new_board

@njit
def get_valid_moves(board, player_id):
    """Devolve um array 2D com [x, y] das jogadas válidas."""
    moves = np.empty((64, 2), dtype=np.int8)
    count = 0
    for y in range(8):
        for x in range(8):
            if len(get_flips(board, player_id, x, y)) > 0:
                moves[count, 0] = x
                moves[count, 1] = y
                count += 1
    return moves[:count]

@njit
def evaluate_board(board, player_id, use_mobility):
    """Itera sobre 64 posições em vez de criar nested loops."""
    opponent = 3 - player_id
    score = 0

    for i in range(64):
        if board[i] == player_id:
            score += WEIGHTS[i]
        elif board[i] == opponent:
            score -= WEIGHTS[i]

    if use_mobility:
        my_moves = len(get_valid_moves(board, player_id))
        opp_moves = len(get_valid_moves(board, opponent))
        score += 15 * (my_moves - opp_moves)

    return score

# Classe "wrapper" opcional para não partires o resto do teu código
class OthelloLogic:
    @staticmethod
    def get_flips(board, player_id, x, y):
        return get_flips(board, player_id, x, y)
        
    @staticmethod
    def simulate_move(board, player_id, x, y):
        return simulate_move(board, player_id, x, y)
        
    @staticmethod
    def get_valid_moves(board, player_id):
        return get_valid_moves(board, player_id)
        
    @staticmethod
    def evaluate_board(board, player_id, use_mobility=False):
        return evaluate_board(board, player_id, use_mobility)