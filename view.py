import pygame
import chess

    # 사이즈
WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8

# pygame 초기화
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("C-H-E-E-S")
clock = pygame.time.Clock()

# 보드 
board = chess.Board()

# 보드 이미지 불러오기
board_background = pygame.image.load("chess_img/board.png")
board_background = pygame.transform.scale(board_background, (WIDTH, HEIGHT))

# 말 이미지 불러오기
piece_images = {}
pieces = ['p', 'r', 'n', 'b', 'q', 'k']
for piece in pieces:
    piece_images['w' + piece] = pygame.image.load(f'chess_img/w{piece}.png')
    piece_images['b' + piece] = pygame.image.load(f'chess_img/b{piece}.png')

# 보드 그리기
def draw_board():
    screen.blit(board_background, (0, 0))  # 배경 보드 그리기

# 말 그리기
def draw_pieces():
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            key = ('w' if piece.color else 'b') + piece.symbol().lower()
            image = piece_images[key]
            screen.blit(image, (col * SQUARE_SIZE, row * SQUARE_SIZE))

# 메인 루프
prev = False
running = True
while running:
    draw_board()
    draw_pieces()
    pygame.display.flip()
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            print('MOUSEBUTTONDOWN')


pygame.quit()