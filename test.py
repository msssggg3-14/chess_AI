import chess.engine
import time
from chess_env import ChessEnv
import numpy as np
import pygame

env = ChessEnv()
env.reset()

# Stockfish 엔진 로드
engine = chess.engine.SimpleEngine.popen_uci("stockfish\stockfish-windows-x86-64-avx2.exe")

done = False
env.render()
time.sleep(1)

while not done:
    if env.board.turn == chess.WHITE:
        # 랜덤 에이전트 (또는 너가 직접 두어도 됨)
        # legal_moves = list(env.board.legal_moves)
        # action = str(np.random.choice(legal_moves))
        # print(f"Random agent plays: {action}")
        running = True
        stack = []
        while running:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    x, y = x // 80, y // 80
                    print(x, y)
                    print(env.board.piece_at(8*y + x))




                


    else:
        # Stockfish가 수 계산
        result = engine.play(env.board, chess.engine.Limit(time=0.1))
        action = result.move.uci()
        print(f"Stockfish plays: {action}")
    
    _, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.5)

print("게임 종료")
engine.quit()
