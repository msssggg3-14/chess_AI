import chess
import numpy as np
import pygame


class ChessEnv:
    def __init__(self, agent_color = chess.WHITE):
           self.board = chess.Board()
           self.agent_color = agent_color
           self._init_render()


    def _init_render(self):
        # 사이즈
        self.WIDTH, self.HEIGHT = 640, 640
        self.SQUARE_SIZE = self.WIDTH // 8

        # pygame 초기화
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("C-H-E-E-S")
        self.clock = pygame.time.Clock()

        # 보드 이미지 불러오기
        self.board_background = pygame.image.load("chess_img/board.png")
        self.board_background = pygame.transform.scale(self.board_background, (self.WIDTH, self.HEIGHT))

        # 말 이미지 불러오기
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
        move = chess.Move.from_uci(action)
        # 움직임 구현현
        if move in self.board.legal_moves:
            self.board.push(move)
        else : 
            reward = -1
            done = True
        
        # 체스판 상태 확인 밑 점수 부여 
        if self.board.is_checkmate():
            if self.board.turn == self.agent_color:
                reward -= 1; done = True
            else:
                reward += 1; done = True

        elif self.board.is_stalemate():
            print("스테일메이트입니다.")
            #reward += 0
            done = True

        elif self.board.is_check():
            print("현재 체크 상태입니다.")
        
        info = {
            "turn": self.board.turn,
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate()
               }

        return self.get_state(), reward, done, info


    def get_state(self): # 보드를 읽어서 8,8,12 넘파이 배열로 반환환
        state = np.zeros((8,8,12), dtype=np.uint8)
        pieces = 'prnbqkPRNBQK'
        for sqares in chess.SQUARES:
            piece = self.board.piece_at(sqares)
            if piece :
                row = 7 - (sqares // 8)
                col = sqares % 8
                channel = pieces.find(str(piece))
                state[row][col][channel] = 1
        return state 


    def legal_actions(self):
         return list(self.board.legal_moves)
    

    def render(self):
        self.screen.blit(self.board_background, (0, 0))  # 배경 보드 그리기
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