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

@njit
def extract_features_njit(board_1d, player_id):
    """Gera as 132 features inteiramente em C/Numba, sem ir ao Python."""
    nnue_obs = np.zeros(132, dtype=np.float32)
    opponent_id = 3 - player_id
    
    # 1. Mapa de Peças (Loops simples no Numba são mais rápidos que máscaras do Numpy)
    for i in range(64):
        if board_1d[i] == player_id:
            nnue_obs[i] = 1.0
        elif board_1d[i] == opponent_id:
            nnue_obs[i+64] = 1.0
            
    # 2. Cantos
    corners = np.array([0, 7, 56, 63], dtype=np.int8)
    my_corners, opp_corners = 0, 0
    for c in corners:
        if board_1d[c] == player_id: my_corners += 1
        elif board_1d[c] == opponent_id: opp_corners += 1
    nnue_obs[128] = (my_corners - opp_corners) / 4.0
    
    # 3. X-Squares
    x_squares = np.array([9, 14, 49, 54], dtype=np.int8)
    my_x = 0
    for x in x_squares:
        if board_1d[x] == player_id: my_x += 1
    nnue_obs[129] = -my_x / 4.0
    
    # 4. Mobilidade (Como isto já está no Numba, chamar o get_valid_moves é instantâneo!)
    my_moves = len(get_valid_moves(board_1d, player_id))
    opp_moves = len(get_valid_moves(board_1d, opponent_id))
    total_moves = max(my_moves + opp_moves, 1)
    nnue_obs[130] = (my_moves - opp_moves) / total_moves
    
    # 5. Centro
    center = np.array([18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45], dtype=np.int8)
    my_center = 0
    for c in center:
        if board_1d[c] == player_id: my_center += 1
    nnue_obs[131] = my_center / 16.0
    
    return nnue_obs

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

    @staticmethod
    def extract_features(board_1d, player_id):
        return extract_features_njit(board_1d, player_id)