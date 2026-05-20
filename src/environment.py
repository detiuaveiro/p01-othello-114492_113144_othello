import numpy as np
import torch
from agents.utils import OthelloLogic

class OthelloEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reinicia o tabuleiro para o estado inicial usando um array 1D (64 posições)."""
        self.board = np.zeros(64, dtype=np.int8)
        self.board[3 * 8 + 3] = 2  # Branco
        self.board[4 * 8 + 4] = 2  # Branco
        self.board[3 * 8 + 4] = 1  # Preto
        self.board[4 * 8 + 3] = 1  # Preto
        return self.get_state()

    def get_state(self):
        """A rede neural (CNN) continua a precisar da matriz 2D (8x8)."""
        # Remoldamos apenas na hora de enviar para o PyTorch
        state_2d = self.board.reshape(8, 8)
        return torch.FloatTensor(state_2d).unsqueeze(0).unsqueeze(0)

    def step(self, action_idx, player_id):
        x, y = action_idx % 8, action_idx // 8
        valid_moves = OthelloLogic.get_valid_moves(self.board, player_id)

        # Verificar se a jogada [x, y] está na lista de válidas
        is_valid = False
        for i in range(len(valid_moves)):
            if valid_moves[i][0] == x and valid_moves[i][1] == y:
                is_valid = True
                break

        # 1. Punição por jogada inválida
        if not is_valid:
            return self.get_state(), -10, True 

        # 2. Executar a jogada
        self.board = OthelloLogic.simulate_move(self.board, player_id, x, y)

        # 3. Verificar se o jogo acabou
        p1_moves = len(OthelloLogic.get_valid_moves(self.board, 1))
        p2_moves = len(OthelloLogic.get_valid_moves(self.board, 2))
        done = (p1_moves == 0) and (p2_moves == 0)

        # 4. Calcular Recompensa
        reward = 0
        if done:
            # np.count_nonzero é instantâneo comparado com loops normais
            p1_count = np.count_nonzero(self.board == 1)
            p2_count = np.count_nonzero(self.board == 2)

            if player_id == 1:
                reward = 1 if p1_count > p2_count else -1
            else:
                reward = 1 if p2_count > p1_count else -1

            if p1_count == p2_count:
                reward = 0

        return self.get_state(), reward, done

    def get_valid_mask(self, player_id):
        """Retorna máscara de 64 posições."""
        mask = np.zeros(64, dtype=np.float32)
        valid_moves = OthelloLogic.get_valid_moves(self.board, player_id)
        for i in range(len(valid_moves)):
            x, y = valid_moves[i]
            mask[y * 8 + x] = 1
        return mask