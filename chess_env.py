import chess
import numpy as np
import pygame

class ChessEnv:
    def __init__(self, agent_color=chess.WHITE, render=True):
        self.board = chess.Board()
        self.agent_color = agent_color
        self.render_enabled = render
        self.selected_square = None
        if render:
            self._init_render()

    def _init_render(self):
        self.WIDTH, self.HEIGHT = 640, 640
        self.SQUARE_SIZE = self.WIDTH // 8

        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("C-H-E-E-S")
        self.clock = pygame.time.Clock()

        self.board_background = pygame.image.load("chess_img/board.png")
        self.board_background = pygame.transform.scale(self.board_background, (self.WIDTH, self.HEIGHT))

        self.piece_images = {}
        pieces = ['p', 'r', 'n', 'b', 'q', 'k']
        for piece in pieces:
            self.piece_images['w' + piece] = pygame.image.load(f'chess_img/w{piece}.png')
            self.piece_images['b' + piece] = pygame.image.load(f'chess_img/b{piece}.png')

    def reset(self, agent_color=chess.WHITE):
        if agent_color is not None:
            self.agent_color = agent_color
        self.board.reset()
        return self.get_state()

    def step(self, action):
        reward = 0
        done = False
        # promotion 체크 전에 move 먼저 만들고
        move = chess.Move.from_uci(action)

        # 자동 프로모션 로직 수행
        if (
            move.promotion is None and
            self.board.piece_at(move.from_square) is not None and
            self.board.piece_at(move.from_square).piece_type == chess.PAWN and
            chess.square_rank(move.to_square) in [0, 7]
        ):
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            action = move.uci()



        if move in self.board.legal_moves:
            self.board.push(move)
        else:
            reward = -100
            return self.get_state(), reward, False, {"illegal": True}

        if self.board.is_checkmate():
            if self.board.turn == self.agent_color:
                reward -= 200
            else:
                reward += 300
            done = True

        elif self.board.is_stalemate():
            print("스테일메이트입니다.")
            done = True

        elif self.board.is_check():
            print("현재 체크 상태입니다.")

        info = {
            "turn": self.board.turn,
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate()
        }

        return self.get_state(), reward, done, info

    def get_state(self):
        state = np.zeros((8, 8, 12), dtype=np.uint8)
        pieces = 'prnbqkPRNBQK'
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece:
                row = 7 - (sq // 8)
                col = sq % 8
                channel = pieces.find(str(piece))
                state[row][col][channel] = 1
        return state

    def legal_actions(self):
        return list(self.board.legal_moves)

    def render(self):
        if not self.render_enabled:
            return

        self.screen.blit(self.board_background, (0, 0))

        if self.selected_square is not None:
            row = 7 - (self.selected_square // 8)
            col = self.selected_square % 8
            highlight = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
            highlight.set_alpha(100)
            highlight.fill((50, 50, 50))
            self.screen.blit(highlight, (col * self.SQUARE_SIZE, row * self.SQUARE_SIZE))

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8
                key = ('w' if piece.color else 'b') + piece.symbol().lower()
                image = self.piece_images[key]
                self.screen.blit(image, (col * self.SQUARE_SIZE, row * self.SQUARE_SIZE))

        pygame.display.flip()
        self.clock.tick(60)
