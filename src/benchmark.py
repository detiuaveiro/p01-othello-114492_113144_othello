import os
import torch
import random
import numpy as np
from src.network import OthelloNet
from src.environment import OthelloEnv
from agents.classical_agent import ClassicalAgent
import time

def run_benchmark(model_path="models/evo_best.pth", num_openings=5):
    print("="*60)
    print("🚀 INICIANDO BENCHMARK OTHELLO (TCEC Style) 🚀")
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
        ("Minimax Hard (D6 + Mob)", "h", 6)
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
                        obs = env.get_state_for_player(ai_role)
                        with torch.no_grad():
                            q = model(obs)
                            q_vals = q.squeeze().numpy()
                            q_vals[mask == 0] = -1e8 
                            action = int(np.argmax(q_vals))
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

                flat_board = np.array(env.board).flatten()
                p1_c = np.count_nonzero(flat_board == 1)
                p2_c = np.count_nonzero(flat_board == 2)
                
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
    run_benchmark(num_openings=5)