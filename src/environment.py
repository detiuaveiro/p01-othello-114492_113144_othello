import numpy as np
import torch
from agents.utils import OthelloLogic

class OthelloEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reinicia o tabuleiro para o estado inicial usando listas 2D."""
        self.board = [[0] * 8 for _ in range(8)]
        self.board[3][3], self.board[4][4] = 2, 2  # Branco
        self.board[3][4], self.board[4][3] = 1, 1  # Preto
        return self.get_state_for_player(player_id=1)

    def get_state(self):
        # MANTIDO PARA COMPATIBILIDADE ANTIGA (Não usado no treino atual)
        state_2d = np.array(self.board)
        return torch.FloatTensor(state_2d).unsqueeze(0).unsqueeze(0)

    def get_state_for_player(self, player_id):
        """A rede neural (CNN) precisa da matriz 2D normalizada para 1 e -1."""
        state = np.array(self.board)
        
        # NORMALIZAÇÃO: O jogador atual vê as suas peças como 1 e as do outro como -1
        if player_id == 1:
            norm_state = np.where(state == 1, 1, np.where(state == 2, -1, 0))
        else:
            norm_state = np.where(state == 2, 1, np.where(state == 1, -1, 0))
            
        return torch.FloatTensor(norm_state).unsqueeze(0).unsqueeze(0)

    def step(self, action_idx, player_id):
        """Executa uma jogada e retorna: (novo_estado, recompensa, fim_de_jogo)"""
        x, y = action_idx % 8, action_idx // 8
        valid_moves = OthelloLogic.get_valid_moves(self.board, player_id)

        # Punição por jogada inválida
        if [x, y] not in valid_moves:
            return self.get_state_for_player(player_id), -10, True 

        # O OthelloLogic com Numba processa e devolve a matriz 2D
        self.board = OthelloLogic.simulate_move(self.board, player_id, x, y)

        p1_moves = len(OthelloLogic.get_valid_moves(self.board, 1))
        p2_moves = len(OthelloLogic.get_valid_moves(self.board, 2))
        done = (p1_moves == 0) and (p2_moves == 0)

        reward = 0
        if done:
            p1_count = sum(row.count(1) for row in self.board)
            p2_count = sum(row.count(2) for row in self.board)

            if player_id == 1:
                reward = 1 if p1_count > p2_count else -1
            else:
                reward = 1 if p2_count > p1_count else -1

            if p1_count == p2_count:
                reward = 0

        return self.get_state_for_player(player_id), reward, done

    def get_valid_mask(self, player_id):
        """Retorna máscara de 64 posições em float32 para tensores."""
        mask = np.zeros(64, dtype=np.float32)
        valid_moves = OthelloLogic.get_valid_moves(self.board, player_id)
        for x, y in valid_moves:
            mask[y * 8 + x] = 1
        return mask