import os
import torch
import random
import numpy as np
from src.network import OthelloNet
from src.environment import OthelloEnv
from agents.classical_agent import ClassicalAgent
from agents.utils import OthelloLogic
import time

def run_benchmark(model_path="models/othello_brain_final.pth", num_openings=5):
    print("="*60)
    print("INICIANDO BENCHMARK OTHELLO (TCEC Style)")
    print(f"Modelo: {model_path}")
    print(f"Número de Aberturas: {num_openings} (Joga como P1 e P2 em cada)")
    print(f"Total de jogos por nível: {num_openings * 2}")
    print("="*60)

    device = torch.device("cpu")
    model = OthelloNet().to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("[+] Modelo carregado com sucesso!\n")
    else:
        print(f"[!] Ficheiro não encontrado: {model_path}")
        return

    env = OthelloEnv()
    teacher = ClassicalAgent()
    
    levels = [
        ("Minimax Easy (D2)", "e", 2),
        ("Minimax Normal (D4)", "n", 4),
        ("Minimax Hard (D6 + Mob)", "n", 6)
    ]

    for level_name, mode, depth in levels:
        print(f"\n--- Teste vs {level_name} ---")
        teacher.set_difficulty(mode)
        
        wins = 0
        draws = 0
        total_margin = 0
        start_t = time.time()

        for opening_id in range(num_openings):
            for ai_role in [1, 2]:
                env.reset()
                done = False
                curr_p = 1
                
                # --- ABERTURA CAÓTICA SEGURA ---
                for _ in range(2):
                    mask = env.get_valid_mask(curr_p)
                    if any(mask):
                        idx = random.choice([i for i, m in enumerate(mask) if m == 1])
                        _, _, done = env.step(idx, curr_p)
                    curr_p = 3 - curr_p

                teacher.transposition_table = {}

                while not done:
                    mask = env.get_valid_mask(curr_p)
                    
                    if not any(mask):
                        curr_p = 3 - curr_p
                        if not any(env.get_valid_mask(curr_p)):
                            break
                        continue
                    
                    if curr_p == ai_role:
                        valid_moves = OthelloLogic.get_valid_moves(env.board, ai_role)
                        best_action = None
                        best_score = -float('inf')

                        with torch.no_grad():
                            for move in valid_moves:
                                next_board = OthelloLogic.simulate_move(env.board, ai_role, move[0], move[1])
                                opp_moves = OthelloLogic.get_valid_moves(next_board, 3 - ai_role)
                                
                                if len(opp_moves) == 0:
                                    obs_np = OthelloLogic.extract_features(next_board, ai_role)
                                    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                                    future_score = model(obs).item()
                                else:
                                    # Avaliação em lote da resposta do adversário
                                    batch_array = np.zeros((len(opp_moves), 132), dtype=np.float32)
                                    for i, opp_move in enumerate(opp_moves):
                                        after_opp = OthelloLogic.simulate_move(next_board, 3 - ai_role, opp_move[0], opp_move[1])
                                        batch_array[i] = OthelloLogic.extract_features(after_opp, ai_role)
                                    
                                    batch_obs = torch.as_tensor(batch_array, device=device)
                                    scores = model(batch_obs).squeeze(-1)
                                    future_score = torch.min(scores).item()

                                action_idx = move[1] * 8 + move[0]
                                immediate_reward = 0
                                if action_idx in [0, 7, 56, 63]: immediate_reward = 2.0
                                elif action_idx in [1, 8, 9, 6, 14, 15, 48, 49, 57, 62, 55, 54]: immediate_reward = -1.0
                                
                                total_score = future_score + immediate_reward
                                
                                if total_score > best_score:
                                    best_score = total_score
                                    best_action = move
                                    
                        action = best_action[1] * 8 + best_action[0]
                        _, _, done = env.step(action, ai_role)
                    
                    else:
                        opp_p = 3 - ai_role
                        _, move = teacher.minmax(
                            env.board, depth, -float('inf'), float('inf'), 
                            True, opp_p, teacher.use_mobility
                        )
                        if move is None:
                            idx = random.choice([i for i, m in enumerate(mask) if m == 1])
                        else:
                            idx = move[1] * 8 + move[0]
                        _, _, done = env.step(idx, opp_p)
                    
                    curr_p = 3 - curr_p

                # ----------------------------------------------------
                # ESTATÍSTICAS DO FIM DO JOGO
                # ----------------------------------------------------
                # env.board já é 1D no novo ambiente, não precisa flatten()
                p1_c = np.count_nonzero(env.board == 1)
                p2_c = np.count_nonzero(env.board == 2)
                
                my_s = p1_c if ai_role == 1 else p2_c
                opp_s = p2_c if ai_role == 1 else p1_c
                
                margin = my_s - opp_s
                total_margin += margin

                if my_s > opp_s:
                    wins += 1
                elif my_s == opp_s:
                    draws += 1
                
                cor = "Pretas(P1)" if ai_role == 1 else "Brancas(P2)"
                res = "Vitória" if my_s > opp_s else ("Empate" if my_s == opp_s else "Derrota")
                print(f" Abertura {opening_id+1} | {cor}: {res:7} | Score: {my_s:2} - {opp_s:2}")

        total_games = num_openings * 2
        win_rate = (wins / total_games) * 100
        avg_margin = total_margin / total_games
        elapsed = time.time() - start_t
        
        print(f"> Resultado Final vs {level_name}:")
        print(f"  WinRate: {win_rate:.1f}% ({wins}V - {draws}E - {total_games - wins - draws}D)")
        print(f"  Margem Média: {avg_margin:+.1f} peças")
        print(f"  Tempo Total: {elapsed:.1f}s\n")

if __name__ == "__main__":
    run_benchmark(num_openings=10)