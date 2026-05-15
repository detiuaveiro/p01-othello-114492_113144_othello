import torch
import random
import os
import time
import numpy as np
import gc
from src.network import OthelloNet
from src.environment import OthelloEnv
from agents.classical_agent import ClassicalAgent

class EvolutionTrainer:
    """
    Trainer for Othello agents using Neuroevolution.
    Includes Difficulty Weighting, Adaptive Mutation, and Seed Loading.
    """
    def __init__(self, pop_size: int = 20, mutation_rate: float = 0.15):
        self.pop_size = pop_size
        self.base_mutation_rate = mutation_rate
        self.current_mutation_rate = mutation_rate
        self.device = torch.device("cpu")
        self.env = OthelloEnv()
        self.teacher = ClassicalAgent()
        
        # Initialize population
        self.population = [OthelloNet().to(self.device) for _ in range(pop_size)]
        os.makedirs("models/evo", exist_ok=True)

        # SEED LOADING: Continue from the best model found so far
        model_path = "models/evo_best.pth"
        if os.path.exists(model_path):
            print(f"[*] Loading seed weights from {model_path}...")
            best_weights = torch.load(model_path, map_location=self.device)
            for i, agent in enumerate(self.population):
                agent.load_state_dict(best_weights)
                # Leave the first one as an exact clone, mutate the others to create diversity
                if i > 0: 
                    self.mutate_inplace(agent, intensity=1.5)
        else:
            print("[!] No seed model found. Starting with random weights.")

    @torch.no_grad()
    def get_fitness(self, model: torch.nn.Module, generation: int):
        """Evaluates an agent and returns (weighted_fitness, raw_fitness)."""
        model.eval()
        raw_fitness = 0.0
        
        # Difficulty Curriculum and Weights
        if generation < 15:
            mode, depth, weight = "random", 1, 0.6
        elif generation < 40:
            mode, depth, weight = "e", 2, 0.8
        else:
            mode, depth, weight = "h", 4, 1.4

        self.teacher.set_difficulty(mode)

        # Evaluate as both Player 1 and Player 2
        for ai_p in [1, 2]:
            opp_p = 3 - ai_p
            self.env.reset()
            done = False
            curr_p = 1
            self.teacher.transposition_table = {} # Aggressive RAM management
            
            while not done:
                mask = self.env.get_valid_mask(curr_p)
                if not any(mask):
                    curr_p = 3 - curr_p
                    if not any(self.env.get_valid_mask(curr_p)):
                        break
                    continue
                
                if curr_p == ai_p:
                    obs = self.env.get_state(ai_p)
                    q = model(obs)
                    q_vals = q.squeeze().numpy()
                    q_vals[mask == 0] = -1e8
                    action = int(np.argmax(q_vals))
                    self.env.step(action, ai_p)
                else:
                    _, move = self.teacher.minmax(
                        self.env.board, depth, -float('inf'), float('inf'), 
                        True, opp_p, self.teacher.use_mobility
                    )
                    self.env.step(move[1] * 8 + move[0], opp_p)
                curr_p = 3 - curr_p

            p1_c, p2_c = sum(row.count(1) for row in self.env.board), sum(row.count(2) for row in self.env.board)
            my_s, opp_s = (p1_c, p2_c) if ai_p == 1 else (p2_c, p1_c)
            raw_fitness += (100 + (my_s - opp_s)) if my_s > opp_s else (my_s - opp_s)

        return (raw_fitness * weight), raw_fitness

    def mutate_inplace(self, model: torch.nn.Module, intensity: float = 1.0):
        """Applies Gaussian mutation to the network weights."""
        with torch.no_grad():
            for param in model.parameters():
                mutation_mask = torch.rand(param.size()) < (self.current_mutation_rate * intensity)
                noise = torch.randn(param.size()) * 0.05 * intensity
                param.add_(mutation_mask * noise)

    def train(self, generations: int = 1000):
        best_overall_weighted = -float('inf')
        stagnation_counter = 0

        print(f"[*] Evolution started | Pop: {self.pop_size} | Rate: {self.base_mutation_rate}")
        
        try:
            for gen in range(generations):
                start_t = time.time()
                
                # 1. Evaluation
                results = [self.get_fitness(p, gen) for p in self.population]
                weighted_scores = [r[0] for r in results]
                raw_scores = [r[1] for r in results]
                
                # 2. Selection (Top 25%)
                indices = np.argsort(weighted_scores)[::-1]
                elites = [self.population[i] for i in indices[:self.pop_size // 4]]
                
                curr_best_weighted = weighted_scores[indices[0]]
                curr_best_raw = raw_scores[indices[0]]
                avg_raw = np.mean(raw_scores)
                elapsed = time.time() - start_t

                # 3. Record Keeping
                if curr_best_weighted > best_overall_weighted:
                    best_overall_weighted = curr_best_weighted
                    stagnation_counter = 0
                    self.current_mutation_rate = self.base_mutation_rate # Reset mutation
                    torch.save(elites[0].state_dict(), "models/evo_best.pth")
                    print(f"Gen {gen:3} | NEW RECORD! Raw: {curr_best_raw:5.1f} | Weighted: {curr_best_weighted:6.1f} | T: {elapsed:5.1f}s")
                else:
                    stagnation_counter += 1
                    print(f"Gen {gen:3} | Best Raw: {curr_best_raw:5.1f} | Avg: {avg_raw:5.1f} | Stag: {stagnation_counter:2} | T: {elapsed:5.1f}s")

                # 4. Adaptive Mutation
                if stagnation_counter >= 15:
                    self.current_mutation_rate = min(0.5, self.current_mutation_rate + 0.05)
                    print(f" [!] Stagnation detected. Boosting mutation rate to {self.current_mutation_rate:.2f}")
                    stagnation_counter = 0

                # 5. Reproduction
                new_pop = []
                for elite in elites:
                    new_pop.append(elite)
                    for _ in range(3):
                        child = OthelloNet()
                        child.load_state_dict(elite.state_dict())
                        self.mutate_inplace(child)
                        new_pop.append(child)
                
                self.population = new_pop
                if gen % 10 == 0:
                    gc.collect()

        except KeyboardInterrupt:
            print("\n[!] Training interrupted by user.")
        finally:
            torch.save(self.population[0].state_dict(), "models/evo_last_checkpoint.pth")
            print("[+] Best weights remain in models/evo_best.pth. Final state saved.")

if __name__ == "__main__":
    trainer = EvolutionTrainer(pop_size=20, mutation_rate=0.15)
    trainer.train(generations=1000)