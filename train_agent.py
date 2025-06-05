import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from chess_env import ChessEnv
import chess
import pygame
import os
import matplotlib.pyplot as plt

# === DEVICE 설정 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === ACTION SPACE ===
ALL_UCI_MOVES = []
board = chess.Board()
for from_sq in chess.SQUARES:
    for to_sq in chess.SQUARES:
        move = chess.Move(from_sq, to_sq)
        ALL_UCI_MOVES.append(move.uci())
        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            promo_move = chess.Move(from_sq, to_sq, promotion=promo)
            if promo_move.uci() != move.uci():
                ALL_UCI_MOVES.append(promo_move.uci())
ALL_UCI_MOVES = list(set(ALL_UCI_MOVES))
uci_to_index = {uci: idx for idx, uci in enumerate(ALL_UCI_MOVES)}
index_to_uci = {idx: uci for uci, idx in uci_to_index.items()}

# === DQN AGENT ===
class DQNAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, len(ALL_UCI_MOVES))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# === REPLAY BUFFER ===
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)

# === SELECT ACTION ===
def select_action(state, model, board):
    if random.random() < epsilon:
        return random.choice(list(board.legal_moves)).uci()

    state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)[0].cpu().numpy()

    legal_moves = list(board.legal_moves)
    legal_indices = [uci_to_index[m.uci()] for m in legal_moves if m.uci() in uci_to_index]

    if not legal_indices:
        return random.choice(list(board.legal_moves)).uci()

    masked_q = np.full_like(q_values, -np.inf)
    masked_q[legal_indices] = q_values[legal_indices]

    return index_to_uci[np.argmax(masked_q)]

# === RULE-BASED OPPONENT ===
def rule_based_move(board):
    for move in board.legal_moves:
        board.push(move)
        if not board.is_check():
            board.pop()
            return move.uci()
        board.pop()

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }
    for move in board.legal_moves:
        if board.is_capture(move):
            attacker = board.piece_at(move.from_square)
            victim = board.piece_at(move.to_square)
            if attacker and victim:
                if piece_values.get(attacker.piece_type, 0) <= piece_values.get(victim.piece_type, 0):
                    return move.uci()

    return random.choice(list(board.legal_moves)).uci()

# === TRAIN ===
def train(render=False, start_episode=1, model_path=None):
    env = ChessEnv(render=False)
    model = DQNAgent().to(device)
    target_model = DQNAgent().to(device)

    buffer = ReplayBuffer(10000)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    gamma = 0.99
    batch_size = 64
    global epsilon
    epsilon_min = 0.1
    epsilon_decay = 0.9997
    update_target_every = 100

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        target_model.load_state_dict(model.state_dict())
        print(f"[INFO] 모델 {model_path} 로드 완료")
        raw_epsilon = epsilon_decay ** (start_episode - 1)
        epsilon = max(raw_epsilon, epsilon_min)
    else:
        epsilon = 1.0

    reward_history = []
    result_history = []

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }

    for episode in range(start_episode, 100001):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        was_check_last_turn = False

        while not done:
            board = env.board
            before_pieces = board.piece_map().copy()

            action = select_action(state, model, board)
            move = chess.Move.from_uci(action)
            if (move.promotion is None and board.piece_at(move.from_square) is not None
                and board.piece_at(move.from_square).piece_type == chess.PAWN
                and chess.square_rank(move.to_square) in [0, 7]):
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
                action = move.uci()

            next_state, reward, done, info = env.step(action)
            after_pieces = board.piece_map()
            reward -= 0.005

            captured = set(before_pieces.keys()) - set(after_pieces.keys())
            for sq in captured:
                piece = before_pieces[sq]
                if piece.color != board.turn:
                    attacker = before_pieces.get(move.from_square)
                    attacker_value = piece_values.get(attacker.piece_type, 0) if attacker else 0
                    captured_value = piece_values.get(piece.piece_type, 0)
                    reward += captured_value - attacker_value
                else:
                    reward -= piece_values.get(piece.piece_type, 0)

            if board.is_check():
                reward += 1
                if was_check_last_turn:
                    reward += 0.7
                was_check_last_turn = True
            else:
                was_check_last_turn = False

            if move.promotion is not None:
                reward += 10

            buffer.push(state, uci_to_index[action], reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1

            if render and step % 20 == 0:
                env.render()
                pygame.event.pump()

            if step > 75:
                print(f"[경고] {episode} 에피소드가 75수 넘음. 강제 종료합니다.")
                reward -= 200
                break

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states_tensor = torch.FloatTensor(states).permute(0, 3, 1, 2).to(device)
                next_states_tensor = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(device)
                actions_tensor = torch.LongTensor(actions).to(device)
                rewards_tensor = torch.FloatTensor(rewards).to(device)
                dones_tensor = torch.BoolTensor(dones).to(device)

                q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_model(next_states_tensor).max(1)[0]
                    targets = rewards_tensor + gamma * next_q_values * (~dones_tensor)

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epsilon > epsilon_min:
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if episode % update_target_every == 0:
            target_model.load_state_dict(model.state_dict())

        if env.board.is_checkmate():
            result = -1 if env.board.turn == chess.WHITE else 1
        else:
            result = 0

        print(f"현재 ε: {epsilon:.4f}")
        res_str = "승리" if result == 1 else ("패배" if result == -1 else "무승부 또는 강제 종료")
        print(f"{episode} : {res_str} (총 수 {step}회, 총 보상 {total_reward:.2f})\n")

        reward_history.append(total_reward)
        result_history.append(result)

        if episode % 1000 == 0:
            torch.save(model.state_dict(), os.path.join("train_data", f"dqn_chess_ep{episode}.pt"))
            if len(reward_history) >= 100:
                moving_avg = np.convolve(reward_history, np.ones(100)/100, mode='valid')
                
                # 일반 그래프
                plt.figure(figsize=(10, 5))
                plt.plot(moving_avg, label='Avg Reward')
                plt.xlabel('Episode')
                plt.ylabel('100-Episode Moving Avg Reward')
                plt.title(f'Reward Trend up to Episode {episode}')
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join("train_data", f"reward_plot_ep{episode}.png"))
                plt.close()

                # final 형태도 함께 저장
                plt.figure(figsize=(10, 5))
                plt.plot(moving_avg, label='Avg Reward')
                plt.xlabel('Episode')
                plt.ylabel('100-Episode Moving Avg Reward')
                plt.title(f'Final Reward Trend Snapshot at {episode}')
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join("train_data", f"reward_plot_ep{episode}_final.png"))
                plt.close()


if __name__ == '__main__':
    resume_model = "train_data/dqn_chess_ep3000.pt"
    train(render=False, start_episode=1, model_path=resume_model)
