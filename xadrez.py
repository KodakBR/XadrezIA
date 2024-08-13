import pygame
import chess
import os
import random
import numpy as np
import tensorflow as tf
from collections import deque
import logging

# Inicializar logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Iniciando o treinamento")

# Inicializar pygame
pygame.init()

# Dimensões do monitor (definidas pelo usuário)
monitor_width = 1280
monitor_height = 720

# Definir as cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)
GRAY = (169, 169, 169)
HIGHLIGHT = (255, 0, 0)  # Cor para destacar as divisões

# Diretório das imagens das peças
img_dir = "D:/projeto/xadrezIA/xadrez ico"

# Carregar as imagens das peças
PIECE_IMAGES = {
    'r': pygame.image.load(os.path.join(img_dir, 'black_rook.png')),
    'n': pygame.image.load(os.path.join(img_dir, 'black_knight.png')),
    'b': pygame.image.load(os.path.join(img_dir, 'black_bishop.png')),
    'q': pygame.image.load(os.path.join(img_dir, 'black_queen.png')),
    'k': pygame.image.load(os.path.join(img_dir, 'black_king.png')),
    'p': pygame.image.load(os.path.join(img_dir, 'black_pawn.png')),
    'R': pygame.image.load(os.path.join(img_dir, 'white_rook.png')),
    'N': pygame.image.load(os.path.join(img_dir, 'white_knight.png')),
    'B': pygame.image.load(os.path.join(img_dir, 'white_bishop.png')),
    'Q': pygame.image.load(os.path.join(img_dir, 'white_queen.png')),
    'K': pygame.image.load(os.path.join(img_dir, 'white_king.png')),
    'P': pygame.image.load(os.path.join(img_dir, 'white_pawn.png'))
}

# Função para desenhar o tabuleiro
def draw_board(screen, offset_x, offset_y, square_size):
    for row in range(8):
        for col in range(8):
            color = LIGHT_BROWN if (row + col) % 2 == 0 else DARK_BROWN
            pygame.draw.rect(screen, color, pygame.Rect(offset_x + col * square_size, offset_y + row * square_size, square_size, square_size))
            pygame.draw.rect(screen, HIGHLIGHT, pygame.Rect(offset_x + col * square_size, offset_y + row * square_size, square_size, square_size), 2)

# Função para desenhar as peças
def draw_pieces(screen, board, offset_x, offset_y, square_size):
    for row in range(8):
        for col in range(8):
            piece = board.piece_at(chess.square(col, 7 - row))
            if piece:
                piece_image = PIECE_IMAGES[piece.symbol()]
                piece_image = pygame.transform.scale(piece_image, (square_size, square_size))
                screen.blit(piece_image, (offset_x + col * square_size, offset_y + row * square_size))

# Função para converter o tabuleiro em uma representação numérica
def board_to_input(board):
    piece_map = board.piece_map()
    input_array = np.zeros((64, 12), dtype=np.int8)
    for square, piece in piece_map.items():
        piece_type = piece.piece_type - 1
        color = 0 if piece.color == chess.WHITE else 6
        input_array[square][piece_type + color] = 1
    return input_array.flatten()

# Função de perda registrada
@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

# Classe do Agente DQN
class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target_model()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.train_start = 1000  # Começar a treinar após acumular 1000 experiências

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(768,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(loss=custom_mse, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, legal_moves):
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_moves)
        act_values = []
        for move in legal_moves:
            next_state = chess.Board(state.fen())  # Criar um novo tabuleiro a partir do estado atual
            next_state.push(move)
            next_state_input = board_to_input(next_state)
            act_values.append(self.model.predict(next_state_input.reshape(1, -1)))
        return legal_moves[np.argmax(act_values)]

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state.reshape(1, -1))[0]))
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][0] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = tf.keras.models.load_model(filename, custom_objects={'custom_mse': custom_mse})
        self.target_model = tf.keras.models.load_model(filename, custom_objects={'custom_mse': custom_mse})
        # Recompile the model after loading
        self.model.compile(loss=custom_mse, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        self.target_model.compile(loss=custom_mse, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Inicializar os agentes
agent_white = DQNAgent()
agent_black = DQNAgent()

# Carregar modelos se existirem
if os.path.exists('ana_model.h5'):
    agent_white.load('ana_model.h5')

if os.path.exists('pedro_model.h5'):
    agent_black.load('pedro_model.h5')

# Variáveis do jogo
class Game:
    def __init__(self):
        self.board = chess.Board()
        self.move_count = {'white': 0, 'black': 0}
        self.repetitive_moves = {'white': 0, 'black': 0}
        self.winner = None
        self.winner_points = 0
        self.points = {'white': 0, 'black': 0}

    def count_pieces(self):
        piece_counts = {'white': 0, 'black': 0}
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                if piece.color == chess.WHITE:
                    piece_counts['white'] += 1
                else:
                    piece_counts['black'] += 1
        return piece_counts

# Implementação do MCTS
class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.visits = 0
        self.wins = 0
        self.children = []

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def best_child(self, exploration_weight=1.0):
        choices_weights = [
            (child.wins / child.visits) + exploration_weight * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

def mcts(board, num_simulations):
    root = MCTSNode(board)
    for _ in range(num_simulations):
        node = root
        # Selecionar
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # Expandir
        if not node.is_fully_expanded():
            legal_moves = list(node.board.legal_moves)
            for move in legal_moves:
                if move not in [child.move for child in node.children]:
                    new_board = node.board.copy()
                    new_board.push(move)
                    node.children.append(MCTSNode(new_board, node, move))
                    node = node.children[-1]
                    break

        # Simular
        current_board = node.board.copy()
        while not current_board.is_game_over():
            legal_moves = list(current_board.legal_moves)
            move = random.choice(legal_moves)
            current_board.push(move)

        # Retropropagar
        reward = 1 if current_board.result() == '1-0' else -1 if current_board.result() == '0-1' else 0
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent
    return root.best_child(exploration_weight=0).move

# Função para resetar jogos terminados
def reset_finished_games(games):
    for game in games:
        if game.board.is_game_over():
            logging.info(f"Jogo terminado. Vencedor: {game.winner} com {game.winner_points} pontos")
            game.board.reset()
            game.move_count = {'white': 0, 'black': 0}
            game.repetitive_moves = {'white': 0, 'black': 0}
            game.winner = None
            game.winner_points = 0
            game.points = {'white': 0, 'black': 0}

# Função para atualizar o jogo
def update_game(game, agent_white, agent_black, num_simulations):
    if game.board.is_game_over():
        return

    state = board_to_input(game.board)
    legal_moves = list(game.board.legal_moves)

    if game.board.turn == chess.WHITE:
        move = mcts(game.board, num_simulations)
        reward = calculate_reward(game, move, chess.WHITE)
        game.board.push(move)
        next_state = board_to_input(game.board)
        done = game.board.is_game_over()
        agent_white.remember(state, move, reward, next_state, done)
        game.move_count['white'] += 1
    else:
        move = mcts(game.board, num_simulations)
        reward = calculate_reward(game, move, chess.BLACK)
        game.board.push(move)
        next_state = board_to_input(game.board)
        done = game.board.is_game_over()
        agent_black.remember(state, move, reward, next_state, done)
        game.move_count['black'] += 1

    # Limitar movimentos repetitivos do rei
    if game.board.is_repetition():
        if game.board.turn == chess.WHITE:
            game.repetitive_moves['white'] += 1
        else:
            game.repetitive_moves['black'] += 1

    if game.repetitive_moves['white'] > 10 or game.repetitive_moves['black'] > 10:
        game.board.reset()
        game.move_count = {'white': 0, 'black': 0}
        game.repetitive_moves = {'white': 0, 'black': 0}

    # Verificar se o jogo terminou e calcular a pontuação
    if game.board.is_game_over():
        result = game.board.result()
        piece_counts = game.count_pieces()
        game.points['white'] += piece_counts['white']
        game.points['black'] += piece_counts['black']

        if result == '1-0':
            game.winner = 'Ana (Branco)'
            game.winner_points = game.points['white'] + max(1, 100 - game.move_count['white'])
        elif result == '0-1':
            game.winner = 'Pedro (Preto)'
            game.winner_points = game.points['black'] + max(1, 100 - game.move_count['black'])
        else:
            game.winner = 'Empate'
            game.winner_points = -50  # Penalidade para empate
            game.points['white'] -= 25
            game.points['black'] -= 25

# Função para calcular a recompensa
def calculate_reward(game, move, color):
    reward = 0
    board = game.board
    board.push(move)
    if board.is_checkmate():
        reward = 100  # Grande recompensa para vitória
    elif board.is_check():
        reward = 10  # Recompensa menor para cheque
    elif board.is_capture(move):
        reward = 5  # Recompensa para captura
    else:
        reward = 1  # Recompensa mínima para qualquer movimento válido

    # Penalidade para repetição de movimentos
    if game.board.is_repetition():
        reward -= 5

    # Incentivar controle do centro do tabuleiro
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    if move.to_square in center_squares:
        reward += 2

    # Penalidade para empates
    if board.is_game_over() and board.result() == '1/2-1/2':
        reward -= 50

    board.pop()
    return reward

# Função para exibir o vencedor
def display_winner(screen, game, offset_x, offset_y, square_size):
    if game.winner:
        font = pygame.font.Font(None, 36)
        text = f'Vencedor: {game.winner} com {game.winner_points} pontos'
        text_surface = font.render(text, True, BLACK)
        screen.blit(text_surface, (offset_x + 10, offset_y + 8 * square_size + 10))

# Perguntar o número de simulações
num_simulations = int(input("Quantas simulações você quer iniciar? "))

# Calcular o tamanho do tabuleiro com base na resolução do monitor
square_size = min(monitor_width // 8, monitor_height // 8)
screen_width = 8 * square_size
screen_height = 8 * square_size

# Criar a tela
screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
pygame.display.set_caption('Jogo de Xadrez')

# Inicializar o jogo principal
main_game = Game()

# Loop principal do jogo
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            screen_width = event.w
            screen_height = event.h
            square_size = min(screen_width // 8, screen_height // 8)

    screen.fill(WHITE)  # Limpar a tela para evitar sobreposição de texto

    update_game(main_game, agent_white, agent_black, num_simulations)
    draw_board(screen, 0, 0, square_size)
    draw_pieces(screen, main_game.board, 0, 0, square_size)
    display_winner(screen, main_game, 0, 0, square_size)

    # Atualizar a tela
    pygame.display.flip()

    # Treinar os agentes
    agent_white.replay()
    agent_black.replay()

    # Salvar os modelos
    agent_white.save('ana_model.h5')
    agent_black.save('pedro_model.h5')

pygame.quit()

# Salvar modelos ao sair
agent_white.save('ana_model.h5')
agent_black.save('pedro_model.h5')
