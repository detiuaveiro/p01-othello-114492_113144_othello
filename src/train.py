import torch
from torch import nn
import torch.optim as optim
import random
import os
import time
from collections import deque
from src.network import OthelloNet
from src.environment import OthelloEnv
from agents.utils import OthelloLogic
from agents.classical_agent import ClassicalAgent
import torch.nn.functional as F
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=30000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, m_next, d):
        self.buffer.append((s, a, r, s_next, m_next, d))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


def evaluate_vs_minimax(
    policy_net: nn.Module,
    env: OthelloEnv,
    classical_agent: ClassicalAgent,
    device: torch.device,
    last_winrate: float,
    num_games: int = 10,
) -> float:
    policy_net.eval()
    wins = 0
    classical_agent.set_difficulty("n") if last_winrate < 0.6 else classical_agent.set_difficulty("h")

    for _ in range(num_games):
        state = env.reset().to(device)
        done = False
        while not done:
            valid_mask = env.get_valid_mask(player_id=1)
            if not any(valid_mask):
                break

            with torch.no_grad():
                best_action = None
                best_score = -float('inf')
                valid_moves = OthelloLogic.get_valid_moves(env.board, 1)
                
                for move in valid_moves:
                    next_board = OthelloLogic.simulate_move(env.board, 1, move[0], move[1])
                    
                    opp_moves = OthelloLogic.get_valid_moves(next_board, 2)
                    
                    if len(opp_moves) == 0:
                        obs_np = OthelloLogic.extract_features(next_board, 1)
                        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                        future_score = policy_net(obs).item()
                    else:
                        batch_array = np.zeros((len(opp_moves), 132), dtype=np.float32)
                        
                        for i, opp_move in enumerate(opp_moves):
                            after_opp = OthelloLogic.simulate_move(next_board, 2, opp_move[0], opp_move[1])
                            batch_array[i] = OthelloLogic.extract_features(after_opp, 1)
                        
                        batch_obs = torch.as_tensor(batch_array, device=device)
                        scores = policy_net(batch_obs).squeeze(-1)
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

            state, _, done = env.step(action, player_id=1)
            state = state.to(device)
            if done:
                break

            opp_mask = env.get_valid_mask(player_id=2)
            if any(opp_mask):
                classical_agent.transposition_table = {}
                _, move = classical_agent.minmax(
                    env.board,
                    classical_agent.depth,
                    float("-inf"),
                    float("inf"),
                    True,
                    2,
                    classical_agent.use_mobility,
                )
                state, _, done = env.step(move[1] * 8 + move[0], player_id=2)
                state = state.to(device)

        p1 = np.count_nonzero(env.board == 1)
        p2 = np.count_nonzero(env.board == 2)
        if p1 > p2:
            wins += 1

    policy_net.train()
    return wins / num_games

def board_to_nnue_format(board_1d, player_id):
    """NOVO FORMATO: 132 posições (Peças + 4 Features Estratégicas Avançadas)."""
    nnue_obs = np.zeros(132, dtype=np.float32)
    opponent_id = 3 - player_id
    
    nnue_obs[:64][board_1d == player_id] = 1.0
    nnue_obs[64:128][board_1d == opponent_id] = 1.0
    
    corners = np.array([0, 7, 56, 63])
    my_corners = np.count_nonzero(board_1d[corners] == player_id)
    opp_corners = np.count_nonzero(board_1d[corners] == opponent_id)
    nnue_obs[128] = (my_corners - opp_corners) / 4.0
    
    x_squares = np.array([9, 14, 49, 54])
    my_x = np.count_nonzero(board_1d[x_squares] == player_id)
    
    my_moves = len(OthelloLogic.get_valid_moves(board_1d, player_id))
    opp_moves = len(OthelloLogic.get_valid_moves(board_1d, opponent_id))
    total_moves = max(my_moves + opp_moves, 1)
    nnue_obs[130] = (my_moves - opp_moves) / total_moves
    
    center = np.array([18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45])
    my_center = np.count_nonzero(board_1d[center] == player_id)
    nnue_obs[131] = my_center / 16.0
            
    return torch.FloatTensor(nnue_obs)


def train(episodes: int = 2000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = OthelloNet().to(device)
    target_net = OthelloNet().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    memory = ReplayBuffer(30000)
    env = OthelloEnv()
    classical_agent = ClassicalAgent()
    win_history = deque(maxlen=100)
    best_test_winrate = -1.0
    last_test_winrate = 0
    os.makedirs("models", exist_ok=True)
    epsilon = 0.9
    eps_min = 0.05
    eps_decay = (epsilon - eps_min) / (episodes * 0.7)
    batch_size = 64
    gamma = 0.99
    step_count = 0
    FASE=1

    print(f"Treino iniciado no {device}...")
    start_time = time.time()
    start_time_program = time.time()
    print(time.strftime("%H:%M:%S"))

    for ep in range(episodes):
        ai_player = 1 if ep % 2 == 0 else 2
        opp_player = 3 - ai_player
        
        env.reset()
        done = False
        current_player = 1
        is_hard_mode = ep % 5 == 0
        is_dummy = random.random() < 0.5
        done = False
        match FASE:
            case 1:
                opponent_type = "dummy"
            case 2:
                opponent_type = "mixed"
            case 3:
                opponent_type = "minimax"
            
        while not done:
            valid_mask = env.get_valid_mask(current_player)
            
            if not any(valid_mask):
                current_player = 3 - current_player
                if not any(env.get_valid_mask(current_player)):
                    break
                continue

            if current_player == ai_player:
                state_nnue = board_to_nnue_format(env.board, ai_player)
                valid_moves = OthelloLogic.get_valid_moves(env.board, ai_player)
                
                if random.random() < epsilon:
                    move = random.choice(valid_moves)
                    action = move[1] * 8 + move[0]
                else:
                    with torch.no_grad():
                        best_action = None
                        best_score = -float('inf')
                        
                        for move in valid_moves:
                            next_board = OthelloLogic.simulate_move(env.board, ai_player, move[0], move[1])
                            
                            opp_moves = OthelloLogic.get_valid_moves(next_board, opp_player)
                            if len(opp_moves) == 0:
                                obs = board_to_nnue_format(next_board, ai_player).unsqueeze(0).to(device)
                                future_score = policy_net(obs).item()
                            else:
                                obs_list = []
                                for opp_move in opp_moves:
                                    after_opp = OthelloLogic.simulate_move(next_board, opp_player, opp_move[0], opp_move[1])
                                    obs_list.append(board_to_nnue_format(after_opp, ai_player))
                                
                                batch_obs = torch.stack(obs_list).to(device)
                                scores = policy_net(batch_obs).squeeze(-1)
                                
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
                        
                _, reward, done = env.step(action, ai_player)
                
                if action in [0, 7, 56, 63]: reward += 2.0
                if action in [1, 8, 9, 6, 14, 15, 48, 49, 57, 62, 55, 54]: reward -= 1.0

                next_state_nnue = board_to_nnue_format(env.board, ai_player)
                memory.push(state_nnue, None, reward, next_state_nnue, None, done)

                step_count += 1
                if len(memory.buffer) > batch_size and step_count % 4 == 0:
                    transitions = memory.sample(batch_size)
                    
                    b_state = torch.stack([t[0] for t in transitions]).to(device)
                    b_reward = torch.tensor([t[2] for t in transitions], device=device, dtype=torch.float32)
                    b_next_state = torch.stack([t[3] for t in transitions]).to(device)
                    b_done = torch.tensor([t[5] for t in transitions], device=device, dtype=torch.float32)

                    current_v = policy_net(b_state).squeeze()

                    with torch.no_grad():
                        next_v = target_net(b_next_state).squeeze()
                        target_v = b_reward + (gamma * next_v * (1 - b_done))

                    loss = F.smooth_l1_loss(current_v, target_v)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                    optimizer.step()
                
                current_player = opp_player

            else:
                if opponent_type == "dummy":
                    opp_idx = random.choice([i for i, m in enumerate(valid_mask) if m == 1])
                elif opponent_type == "mixed":
                    if is_dummy:
                        opp_idx = random.choice([i for i, m in enumerate(valid_mask) if m == 1])
                    else:  
                        classical_agent.set_difficulty("n")
                        classical_agent.transposition_table = {}
                        _, move = classical_agent.minmax(env.board, 2, float('-inf'), float('inf'), True, opp_player, classical_agent.use_mobility)
                        opp_idx = move[1] * 8 + move[0]
                        
                elif opponent_type == "minimax":
                    if is_hard_mode:
                        classical_agent.set_difficulty("h")
                        classical_agent.transposition_table = {}
                        _, move = classical_agent.minmax(env.board, 4, float('-inf'), float('inf'), True, opp_player, classical_agent.use_mobility)
                        opp_idx = move[1] * 8 + move[0]
                    else:
                        classical_agent.set_difficulty("n")
                        classical_agent.transposition_table = {}
                        _, move = classical_agent.minmax(env.board, 2, float('-inf'), float('inf'), True, opp_player, classical_agent.use_mobility)
                        opp_idx = move[1] * 8 + move[0]
                    
                _, _, done = env.step(opp_idx, opp_player)
                
                if done:
                    p1 = np.count_nonzero(env.board == 1)
                    p2 = np.count_nonzero(env.board == 2)
                    final_reward = 1 if p1 > p2 else -1 if p1 < p2 else 0
                    if ai_player == 2: final_reward = -final_reward
                    
                    next_state_nnue = board_to_nnue_format(env.board, ai_player)
                    memory.push(state_nnue, None, final_reward, next_state_nnue, None, done)
                
                current_player = ai_player

        if (ep + 1) % 1000 == 0:
            test_winrate = evaluate_vs_minimax(policy_net, env, classical_agent, device, last_test_winrate)
            last_test_winrate = test_winrate
            if test_winrate >= best_test_winrate:
                best_test_winrate = test_winrate
                torch.save(policy_net.state_dict(), "models/othello_best_strategic.pth")
                print(f"!!! NOVO RECORDE ESTRATÉGICO SALVO ({test_winrate*100}%) !!!\n")

        p1_c = np.count_nonzero(env.board == 1)
        p2_c = np.count_nonzero(env.board == 2)
        
        if ai_player == 1:
            win_history.append(1 if p1_c > p2_c else 0)
        else:
            win_history.append(1 if p2_c > p1_c else 0)

        epsilon = max(eps_min, epsilon - eps_decay)

        if (ep + 1) % 500 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            win_rate = sum(win_history) / 100
            if win_rate < 0.35:
                epsilon = min(0.5, epsilon + 0.15)
                print(f"-- RECOVERY: WinRate {win_rate:.2f} baixa, Eps subiu para {epsilon:.2f} --")

        if (ep + 1) % 100 == 0:
            curr_time = time.time()
            win_rate = sum(win_history) / len(win_history)
            if (win_rate>0.63 or (ep + 1) % 1800) and FASE<3:
                nomes_fases = {
                    1: "DUMMY (Aleatório)", 
                    2: "MIXED (Misto)", 
                    3: "MINIMAX (Árvore de Decisão)"
                }
                
                fase_antiga = nomes_fases[FASE]
                FASE += 1
                fase_nova = nomes_fases[FASE]
                
                print(f"fase_antiga: {fase_antiga}")
                print(f"fase_nova: {fase_nova}")
            duration_time=curr_time - start_time
            left_time=(duration_time / 100*(episodes-(ep+1)))
            hours = int(left_time // 3600)
            minutes = int((left_time % 3600) // 60)
            seconds = int(left_time % 60)
            duration_time=curr_time - start_time_program
            duration_hours = int(duration_time // 3600)
            duration_minutes = int((duration_time % 3600) // 60)
            duration_seconds = int(duration_time % 60)

            print(
                f"Ep {ep + 1}/{episodes} | Loss: {loss.item():.4f} | WinRate: {win_rate:.2f} | Eps: {epsilon:.2f} | Time: {curr_time - start_time:.1f}s | Est: {hours:02d}h {minutes:02d}m {seconds:02d}s | Dur: {duration_hours:02d}h {duration_minutes:02d}m {duration_seconds:02d}s"
            )
            start_time = curr_time

    torch.save(policy_net.state_dict(), "models/othello_brain_final.pth")

if __name__ == "__main__":
    train(episodes=10000)