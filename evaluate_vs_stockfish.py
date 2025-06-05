import torch
import chess.engine
from chess_env import ChessEnv
from train_agent import DQNAgent, uci_to_index, index_to_uci
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):
    model = DQNAgent().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def select_action(model, state, board):
    state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)[0].cpu().numpy()

    legal_moves = list(board.legal_moves)
    legal_indices = [uci_to_index[m.uci()] for m in legal_moves if m.uci() in uci_to_index]
    masked_q = np.full_like(q_values, -np.inf)
    masked_q[legal_indices] = q_values[legal_indices]

    return index_to_uci[np.argmax(masked_q)]

def rating_to_skill_level(rating):
    return min(20, max(0, (rating - 400) // 80))

def evaluate_vs_stockfish(model_path, stockfish_path, rating, num_games=10):
    model = load_model(model_path)
    skill_level = rating_to_skill_level(rating)

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": skill_level})

    results = {"win": 0, "draw": 0, "loss": 0, "moves": []}

    for _ in range(num_games):
        env = ChessEnv(render=False)
        env.reset()
        done = False
        move_count = 0

        while not done:
            if env.board.turn == chess.WHITE:
                action = select_action(model, env.get_state(), env.board)
            else:
                result = engine.play(env.board, chess.engine.Limit(time=0.1))
                action = result.move.uci()

            # 프로모션 자동 처리
            move = chess.Move.from_uci(action)
            if (move.promotion is None and env.board.piece_at(move.from_square) is not None
                and env.board.piece_at(move.from_square).piece_type == chess.PAWN
                and chess.square_rank(move.to_square) in [0, 7]):
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
                action = move.uci()

            _, reward, done, info = env.step(action)
            move_count += 1

            if move_count > 300:
                break

        # 게임 종료 후 결과 기록
        if env.board.is_checkmate():
            if env.board.turn == chess.BLACK:  # 백이 승
                results["win"] += 1
            else:
                results["loss"] += 1
        elif env.board.is_stalemate() or env.board.is_insufficient_material():
            results["draw"] += 1
        else:
            results["draw"] += 1  # timeout or 300 move rule

        results["moves"].append(move_count)

    engine.quit()
    win_rate = results["win"] / num_games * 100
    avg_moves = np.mean(results["moves"])
    print(f"Rating {rating}: {results['win']}W {results['draw']}D {results['loss']}L | Win rate: {win_rate:.1f}%, Avg moves: {avg_moves:.1f}")
    return win_rate, avg_moves

evaluate_vs_stockfish(
    model_path="train_data/dqn_chess_ep3000.pt",
    stockfish_path="stockfish/stockfish-windows-x86-64-avx2.exe",
    rating=1000,
    num_games=10
)
