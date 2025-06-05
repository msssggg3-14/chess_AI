import matplotlib.pyplot as plt

# 네가 만든 평가 함수 임포트 필요
from evaluate_vs_stockfish import evaluate_vs_stockfish

model_path = "train_data/dqn_chess_ep10000.pt"
stockfish_path = "stockfish/stockfish-windows-x86-64-avx2.exe"
ratings = list(range(400, 1800, 200))  # [400, 600, ..., 1600]

win_rates = []
avg_moves = []

for rating in ratings:
    print(f"\n[INFO] Stockfish Rating {rating} 평가 중...")
    win_rate, avg_move = evaluate_vs_stockfish(
        model_path=model_path,
        stockfish_path=stockfish_path,
        rating=rating,
        num_games=10
    )
    win_rates.append(win_rate)
    avg_moves.append(avg_move)

# 그래프 1: 승률
plt.figure(figsize=(8, 5))
plt.plot(ratings, win_rates, marker='o')
plt.title("DQN Agent vs Stockfish - Winning_Rate")
plt.xlabel("Stockfish 체스닷컴 Rating")
plt.ylabel("승률 (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.tight_layout()
plt.show()

# 그래프 2: 평균 수
plt.figure(figsize=(8, 5))
plt.plot(ratings, avg_moves, marker='o', color='orange')
plt.title("DQN Agent vs Stockfish - Avarage")
plt.xlabel("Stockfish Rating")
plt.ylabel("평균 수")
plt.grid(True)
plt.tight_layout()
plt.show()
