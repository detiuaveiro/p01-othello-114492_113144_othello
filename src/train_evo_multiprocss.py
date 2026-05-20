import torch
import random
import os
import time
import numpy as np
import gc
from concurrent.futures import ProcessPoolExecutor # NOVO: Para usar todos os cores
from src.network import OthelloNet
from src.environment import OthelloEnv
from agents.classical_agent import ClassicalAgent

def evaluate_agent(model_state, generation):
    """Worker function to evaluate a single agent in a separate process."""
    # Cada processo cria o seu próprio ambiente e professor (isolamento total)
    env = OthelloEnv()
    teacher = ClassicalAgent()
    model = OthelloNet()
    model.load_state_dict(model_state)
    model.eval()

    total_fitness = 0.0
    
    if generation < 15:
        mode, depth, weight = "random", 1, 0.6
    elif generation < 40:
        mode, depth, weight = "e", 2, 0.8
    else:
        mode, depth, weight = "h", 4, 1.4

    teacher.set_difficulty(mode)
    match_configs = [1, 2, 1, 2]

    for ai_p in match_configs:
        opp_p = 3 - ai_p
        env.reset()
        done = False
        curr_p = 1
        teacher.transposition_table = {} 
        
        while not done:
            mask = env.get_valid_mask(curr_p)
            if not any(mask):
                curr_p = 3 - curr_p
                if not any(env.get_valid_mask(curr_p)): break
                continue
            
            if curr_p == ai_p:
                obs = env.get_state(ai_p)
                with torch.no_grad():
                    q = model(obs)
                    q_vals = q.squeeze().numpy()
                    q_vals[mask == 0] = -1e7
                    action = int(np.argmax(q_vals))
                env.step(action, ai_p)
            else:
                _, move = teacher.minmax(env.board, depth, -float('inf'), float('inf'), True, opp_p, teacher.use_mobility)
                env.step(move[1] * 8 + move[0], opp_p)
            curr_p = 3 - curr_p

        p1_c, p2_c = sum(row.count(1) for row in env.board), sum(row.count(2) for row in env.board)
        my_s, opp_s = (p1_c, p2_c) if ai_p == 1 else (p2_c, p1_c)
        win_bonus = 150 if ai_p == 2 else 100
        if my_s > opp_s: total_fitness += win_bonus + (my_s - opp_s)
        else: total_fitness += (my_s - opp_s)

    return (total_fitness * weight, total_fitness)

class EvolutionTrainer:
    def __init__(self, pop_size=50, mutation_rate=0.15):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.population = [OthelloNet() for _ in range(pop_size)]
        
        # Carregar semente se existir
        if os.path.exists("models/evo_best.pth"):
            weights = torch.load("models/evo_best.pth", map_location="cpu")
            for i, p in enumerate(self.population):
                p.load_state_dict(weights)
                if i > 0: self.mutate_inplace(p, 1.5)

    def mutate_inplace(self, model, intensity=1.0):
        with torch.no_grad():
            for param in model.parameters():
                mask = torch.rand(param.size()) < (self.mutation_rate * intensity)
                param.add_(mask * torch.randn(param.size()) * 0.05 * intensity)

    def train(self, generations=1000):
        best_weighted = -float('inf')
        # Determinar número de cores (usa todos menos 2 para o sistema respirar)
        num_workers = max(1, os.cpu_count() - 2)
        print(f"[*] Training with {num_workers} parallel processes")

        try:
            for gen in range(generations):
                t0 = time.time()
                
                # PARALELIZAÇÃO ACONTECE AQUI
                # Passamos o state_dict porque o objeto completo não viaja bem entre processos
                pop_states = [p.state_dict() for p in self.population]
                
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Lançar todos os agentes para os cores do CPU
                    futures = [executor.submit(evaluate_agent, state, gen, i) for i, state in enumerate(pop_states)]
                    results = [f.result() for f in futures]

                weighted_scores = [r[0] for r in results]
                raw_scores = [r[1] for r in results]
                indices = np.argsort(weighted_scores)[::-1]
                
                # ... (resto da lógica de Elitismo e Gravação igual) ...
                elites = [self.population[i] for i in indices[:self.pop_size//4]]
                curr_best_w = weighted_scores[indices[0]]
                
                if curr_best_w > best_weighted:
                    best_weighted = curr_best_w
                    torch.save(elites[0].state_dict(), "models/evo_best.pth")
                    print(f"Gen {gen} | NEW RECORD! Raw: {raw_scores[indices[0]]} | T: {time.time()-t0:.1f}s")
                else:
                    print(f"Gen {gen} | Best Raw: {raw_scores[indices[0]]} | Avg: {np.mean(raw_scores):.1f} | T: {time.time()-t0:.1f}s")

                # Reprodução (igual)
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

if __name__ == "__main__":
    trainer = EvolutionTrainer(pop_size=100, mutation_rate=0.15)
    trainer.train(generations=1000)