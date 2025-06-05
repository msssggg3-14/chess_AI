import pygame
import time
import numpy as np
import chess.engine
from chess_env import ChessEnv
from train_agent import DQNAgent, uci_to_index, index_to_uci
import torch

MODE = "human_vs_ai"
STOCKFISH_PATH = r"stockfish\stockfish-windows-x86-64-avx2.exe"
STOCKFISH_RATING = 400  # chess.com 기반 스톡피쉬 난이도 설정
AI_THINK_TIME = 0.1
MODEL_PATH = "train_data/dqn_chess_ep77000.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rating_to_skill_level(rating):
    return min(20, max(0, (rating - 400) // 80))

def load_model(path):
    model = DQNAgent().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

model = load_model(MODEL_PATH)

def my_ai_move(board, state):
    state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)[0].cpu().numpy()

    legal_moves = list(board.legal_moves)
    legal_indices = [uci_to_index[m.uci()] for m in legal_moves if m.uci() in uci_to_index]
    masked_q = np.full_like(q_values, -np.inf)
    masked_q[legal_indices] = q_values[legal_indices]
    return index_to_uci[np.argmax(masked_q)]

def get_mouse_move(board, env):
    stack = []
    while True:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                col = x // env.SQUARE_SIZE
                row = y // env.SQUARE_SIZE
                square = chess.square(col, 7 - row)
                stack.append(square)

                if len(stack) == 1:
                    env.selected_square = square

                elif len(stack) == 2:
                    move = chess.Move(stack[0], stack[1])
                    if (
                        board.piece_at(stack[0]) is not None and
                        board.piece_at(stack[0]).piece_type == chess.PAWN and
                        chess.square_rank(stack[1]) in [0, 7]
                    ):
                        move = chess.Move(stack[0], stack[1], promotion=chess.QUEEN)

                    if move in board.legal_moves:
                        env.selected_square = None
                        return move.uci()
                    else:
                        print("Illegal move")
                        if board.piece_at(stack[1]):
                            stack = [stack[1]]
                            env.selected_square = stack[0]
                        else:
                            stack = []
                            env.selected_square = None

def main():
    env = ChessEnv()
    env.reset()
    done = False
    env.render()
    time.sleep(1)

    while not done:
        turn = env.board.turn

        if MODE == "human_vs_ai":
            if turn == chess.WHITE:
                action = get_mouse_move(env.board, env)
            else:
                action = my_ai_move(env.board, env.get_state())
                print(f"AI plays: {action}")
        else:
            print("Unknown mode.")
            break

        _, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.3)

    print("게임 종료")

if __name__ == "__main__":
    main()
