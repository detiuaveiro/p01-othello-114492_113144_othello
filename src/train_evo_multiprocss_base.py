import torch
import random
import os
import time
import numpy as np
import gc
from concurrent.futures import ProcessPoolExecutor
from src.network import OthelloNet
from src.environment import OthelloEnv
from agents.classical_agent import ClassicalAgent

# --- FIX 1: Limitar threads do PyTorch para evitar bloqueio do CPU ---
torch.set_num_threads(1)

def evaluate_agent(model_state, generation):
    # Forçar 1 thread também dentro do processo worker
    torch.set_num_threads(1)
    try:
        env = OthelloEnv()
        teacher = ClassicalAgent()
        model = OthelloNet()
        model.load_state_dict(model_state)
        model.eval()

        total_fitness = 0.0
        
        # if generation < 15:
        #     mode, d, weight = "random", 1, 0.6
        # elif generation < 40:
        #     mode, d, weight = "e", 2, 0.8
        
        if generation < 50:
            mode, d, weight = "h", 4, 2.0
        else:
            mode, d, weight = "h", 6, 5.0

        teacher.set_difficulty(mode)
        
        match_configs = [1, 2, 1, 2] 

        for ai_p in match_configs:
            opp_p = 3 - ai_p
            env.reset()
            done = False
            curr_p = 1
            teacher.transposition_table = {} 

            moves_played = 0

            while not done:
                mask = env.get_valid_mask(curr_p)
                if not any(mask):
                    curr_p = 3 - curr_p
                    if not any(env.get_valid_mask(curr_p)):
                        break
                    continue
                
                if curr_p == ai_p:
                    obs = env.get_state_for_player(ai_p)
                    with torch.no_grad():
                        q = model(obs)
                        q_vals = q.squeeze().numpy()
                        q_vals[mask == 0] = -1e7
                        action = int(np.argmax(q_vals))
                    env.step(action, ai_p)
                else:
                    if moves_played < 2:
                        idx = random.choice([idx for idx, m in enumerate(mask) if m == 1])
                    else:
                        rand_d = random.choice([2, d])
                        rand_mob = random.choice([True, False])
                        
                        _, move = teacher.minmax(env.board, rand_d, -float('inf'), float('inf'), True, opp_p, rand_mob)
                        idx = move[1] * 8 + move[0]
                    
                    env.step(idx, opp_p)
                
                curr_p = 3 - curr_p
                moves_played += 1

            p1_c, p2_c = sum(row.count(1) for row in env.board), sum(row.count(2) for row in env.board)
            my_s, opp_s = (p1_c, p2_c) if ai_p == 1 else (p2_c, p1_c)
            
            win_bonus = 150 if ai_p == 2 else 100
            if my_s > opp_s:
                total_fitness += win_bonus + (my_s - opp_s)
            else:
                total_fitness += (my_s - opp_s)

            # Bónus de Canto
            corners = [(0,0), (0,7), (7,0), (7,7)]
            for cy, cx in corners:
                if env.board[cy][cx] == ai_p:
                    total_fitness += 50
            # bonus de 
            for y in range(7):
                for x in range(7):
                    if (env.board[y][x] == ai_p and 
                        env.board[y+1][x] == ai_p and 
                        env.board[y][x+1] == ai_p and 
                        env.board[y+1][x+1] == ai_p):
                        
                        # Se o quadrado estiver na borda, vale mais!
                        if x == 0 or x == 6 or y == 0 or y == 6:
                            total_fitness += 30 
                        else:
                            total_fitness += 10

        return (total_fitness * weight, total_fitness)
    except Exception:
        return (-1e9, -1e9)

class EvolutionTrainer:
    def __init__(self, pop_size=60, mutation_rate=0.15):
        self.pop_size = pop_size
        self.base_mutation_rate = mutation_rate
        self.current_mutation_rate = mutation_rate
        self.population = [OthelloNet() for _ in range(pop_size)]
        
        if os.path.exists("models/evo_best.pth"):
            try:
                self.population[0].load_state_dict(torch.load("models/evo_best.pth", map_location="cpu"))
                print("[*] Seed model loaded.")
                for i in range(1, self.pop_size):
                    self.population[i].load_state_dict(self.population[0].state_dict())
                    self.mutate_inplace(self.population[i], 1.5)
            except:
                print("[!] Incompatible seed found (likely old MLP). Starting fresh.")

    def mutate_inplace(self, model, intensity=1.0):
        with torch.no_grad():
            for param in model.parameters():
                mask = torch.rand(param.size()) < (self.current_mutation_rate * intensity)
                param.add_(mask * torch.randn(param.size()) * 0.05 * intensity)

    def train(self, generations=1000):
        best_overall_weighted = -float('inf')
        stagnation_counter = 0
        num_workers = 3 
        
        print(f"[*] CNN Evolution | Pop: {self.pop_size} | Workers: {num_workers}")

        try:
            for gen in range(generations):
                t0 = time.time()
                pop_states = [p.state_dict() for p in self.population]
                
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(evaluate_agent, state, gen) for state in pop_states]
                    results = [f.result() for f in futures]

                weighted_scores = [r[0] for r in results]
                raw_scores = [r[1] for r in results]
                indices = np.argsort(weighted_scores)[::-1]
                
                curr_best_w = weighted_scores[indices[0]]
                elites = [self.population[i] for i in indices[:self.pop_size//4]]
                
                elapsed = time.time() - t0

                if curr_best_w > best_overall_weighted:
                    best_overall_weighted = curr_best_w
                    stagnation_counter = 0
                    torch.save(elites[0].state_dict(), "models/evo_best.pth")
                    print(f"Gen {gen:3} | RECORD! Raw: {raw_scores[indices[0]]:.1f} | T: {elapsed:.1f}s")
                else:
                    stagnation_counter += 1
                    print(f"Gen {gen:3} | Best Raw: {raw_scores[indices[0]]:.1f} | Stag: {stagnation_counter} | T: {elapsed:.1f}s")

                if stagnation_counter >= 15:
                    self.current_mutation_rate = min(0.5, self.current_mutation_rate + 0.05)
                    stagnation_counter = 0

                new_pop = []
                for e in elites:
                    new_pop.append(e)
                    for _ in range(3):
                        child = OthelloNet()
                        child.load_state_dict(e.state_dict())
                        self.mutate_inplace(child)
                        new_pop.append(child)
                self.population = new_pop
                gc.collect()

        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            torch.save(self.population[0].state_dict(), "models/evo_last_checkpoint.pth")

if __name__ == "__main__":
    EvolutionTrainer().train()