import pygame
import time
import numpy as np
import chess.engine
from chess_env import ChessEnv

# ====================
# 설정
# ====================
MODE = "human_vs_stockfish"  # "human_vs_ai", "ai_vs_stockfish", "human_vs_stockfish"
STOCKFISH_PATH = r"stockfish\stockfish-windows-x86-64-avx2.exe"
AI_THINK_TIME = 0.1

# ====================
# 에이전트 정의 (랜덤)
# ====================
def my_ai_move(board):
    legal_moves = list(board.legal_moves)
    move = np.random.choice(legal_moves)
    print(f"My AI plays: {move}")
    return move.uci()

def stockfish_move(board, engine):
    result = engine.play(board, chess.engine.Limit(time=AI_THINK_TIME))
    print(f"Stockfish plays: {result.move}")
    return result.move.uci()

def get_mouse_move(board, env):
    stack = []
    while True:
        env.render()  # 계속 그려줘야 하이라이트가 보임
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
                    if move in board.legal_moves:
                        env.selected_square = None
                        return move.uci()
                    else:
                        print("Illegal move")

                        # 목적지 칸에 기물이 있으면 → 새 선택으로 간주
                        if board.piece_at(stack[1]):
                            stack = [stack[1]]
                            env.selected_square = stack[0]
                        else:
                            stack = []
                            env.selected_square = None




# ====================
# 메인 게임 루프
# ====================
def main():
    env = ChessEnv()
    env.reset()
    engine = None
    if "stockfish" in MODE:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    done = False
    env.render()
    time.sleep(1)

    while not done:
        turn = env.board.turn  # True = white

        if MODE == "human_vs_ai":
            if turn == chess.WHITE:
                action = get_mouse_move(env.board, env)
            else:
                action = my_ai_move(env.board)

        elif MODE == "ai_vs_stockfish":
            if turn == chess.WHITE:
                action = my_ai_move(env.board)
            else:
                action = stockfish_move(env.board, engine)

        elif MODE == "human_vs_stockfish":
            if turn == chess.WHITE:
                action = get_mouse_move(env.board, env)
            else:
                action = stockfish_move(env.board, engine)

        else:
            print("Unknown mode.")
            break

        _, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.3)

    print("게임 종료")
    if engine:
        engine.quit()

if __name__ == "__main__":
    main()
