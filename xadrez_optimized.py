import pygame
import chess
import os
import random
import numpy as np
import tensorflow as tf
from collections import deque
import logging
import time
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle

# Configuração otimizada de logging
logging.basicConfig(
    filename='training.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    buffering=8192  # Buffer para reduzir I/O
)

# Configurações de performance
pygame.init()
pygame.mixer.quit()  # Desabilitar áudio para melhor performance

# Dimensões do monitor
monitor_width = 1280
monitor_height = 720

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)
GRAY = (169, 169, 169)
HIGHLIGHT = (255, 0, 0)

# Cache para imagens das peças
class ImageCache:
    def __init__(self):
        self.cache = {}
        self.square_size = None
        self.img_dir = "xadrez ico"
        self._load_original_images()
    
    def _load_original_images(self):
        """Carrega as imagens originais uma única vez"""
        self.original_images = {}
        piece_files = {
            'r': 'black_rook.png', 'n': 'black_knight.png', 'b': 'black_bishop.png',
            'q': 'black_queen.png', 'k': 'black_king.png', 'p': 'black_pawn.png',
            'R': 'white_rook.png', 'N': 'white_knight.png', 'B': 'white_bishop.png',
            'Q': 'white_queen.png', 'K': 'white_king.png', 'P': 'white_pawn.png'
        }
        
        for piece, filename in piece_files.items():
            try:
                path = os.path.join(self.img_dir, filename)
                self.original_images[piece] = pygame.image.load(path)
            except pygame.error:
                # Fallback: criar uma superfície colorida se a imagem não carregar
                surf = pygame.Surface((64, 64))
                surf.fill((100, 100, 100) if piece.islower() else (200, 200, 200))
                self.original_images[piece] = surf
    
    def get_scaled_image(self, piece_symbol, square_size):
        """Retorna imagem redimensionada com cache"""
        cache_key = f"{piece_symbol}_{square_size}"
        
        if cache_key not in self.cache:
            if square_size != self.square_size:
                # Limpar cache se o tamanho mudou
                self.cache.clear()
                self.square_size = square_size
            
            original = self.original_images.get(piece_symbol)
            if original:
                scaled = pygame.transform.scale(original, (square_size, square_size))
                self.cache[cache_key] = scaled
            else:
                # Fallback
                scaled = pygame.Surface((square_size, square_size))
                scaled.fill((100, 100, 100) if piece_symbol.islower() else (200, 200, 200))
                self.cache[cache_key] = scaled
        
        return self.cache[cache_key]

# Instância global do cache de imagens
image_cache = ImageCache()

# Cache para conversão de tabuleiro
@lru_cache(maxsize=1024)
def board_to_input_cached(fen_string):
    """Versão em cache da conversão de tabuleiro"""
    board = chess.Board(fen_string)
    piece_map = board.piece_map()
    input_array = np.zeros((64, 12), dtype=np.int8)
    for square, piece in piece_map.items():
        piece_type = piece.piece_type - 1
        color = 0 if piece.color == chess.WHITE else 6
        input_array[square][piece_type + color] = 1
    return input_array.flatten()

def board_to_input(board):
    """Wrapper para usar o cache"""
    return board_to_input_cached(board.fen())

# Função de perda otimizada
@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

# Classe DQN otimizada
class OptimizedDQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target_model()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.train_start = 1000
        self.last_save_time = time.time()
        self.save_interval = 60  # Salvar a cada 60 segundos
        
        # Configurações de TensorFlow para melhor performance
        tf.config.optimizer.set_jit(True)
        tf.config.optimizer.set_experimental_options({
            "layout_optimizer": True,
            "constant_folding": True,
            "shape_optimization": True,
            "remapping": True,
            "arithmetic_optimization": True,
            "dependency_optimization": True,
            "loop_optimization": True,
            "function_optimization": True,
            "debug_stripper": True,
        })

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(768,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),  # Adicionar dropout para regularização
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(
            loss=custom_mse, 
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['mae']
        )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, legal_moves):
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_moves)
        
        # Avaliação em lote para melhor performance
        next_states = []
        for move in legal_moves:
            next_board = chess.Board(state.fen())
            next_board.push(move)
            next_states.append(board_to_input(next_board))
        
        if next_states:
            next_states_batch = np.array(next_states)
            predictions = self.model.predict(next_states_batch, verbose=0)
            best_idx = np.argmax(predictions.flatten())
            return legal_moves[best_idx]
        
        return random.choice(legal_moves)

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        
        # Treinamento em lote
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([exp[0] for exp in minibatch])
        actions = [exp[1] for exp in minibatch]
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        
        targets = self.model.predict(states, verbose=0)
        next_targets = self.target_model.predict(next_states, verbose=0)
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][0] = rewards[i]
            else:
                targets[i][0] = rewards[i] + self.gamma * np.amax(next_targets[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            try:
                self.model.save(filename)
                self.last_save_time = current_time
            except Exception as e:
                logging.warning(f"Erro ao salvar modelo: {e}")

    def load(self, filename):
        try:
            self.model = tf.keras.models.load_model(filename, custom_objects={'custom_mse': custom_mse})
            self.target_model = tf.keras.models.load_model(filename, custom_objects={'custom_mse': custom_mse})
            self.model.compile(loss=custom_mse, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
            self.target_model.compile(loss=custom_mse, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        except Exception as e:
            logging.warning(f"Erro ao carregar modelo: {e}")

# MCTS otimizado
class OptimizedMCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.visits = 0
        self.wins = 0
        self.children = []
        self.untried_moves = list(board.legal_moves) if not board.is_game_over() else []

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.414):
        if not self.children:
            return None
        
        choices_weights = [
            (child.wins / child.visits) + 
            exploration_weight * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        if not self.untried_moves:
            return None
        
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        new_board.push(move)
        child = OptimizedMCTSNode(new_board, self, move)
        self.children.append(child)
        return child

def optimized_mcts(board, num_simulations, max_depth=50):
    root = OptimizedMCTSNode(board)
    
    for _ in range(num_simulations):
        node = root
        
        # Seleção
        while node.is_fully_expanded() and node.children and not node.board.is_game_over():
            node = node.best_child()
        
        # Expansão
        if not node.is_fully_expanded() and not node.board.is_game_over():
            node = node.expand()
        
        # Simulação
        current_board = node.board.copy()
        depth = 0
        while not current_board.is_game_over() and depth < max_depth:
            legal_moves = list(current_board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            current_board.push(move)
            depth += 1
        
        # Retropropagação
        result = current_board.result()
        reward = 1 if result == '1-0' else -1 if result == '0-1' else 0
        
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent
    
    if root.children:
        return root.best_child(exploration_weight=0).move
    else:
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None

# Classe de jogo otimizada
class OptimizedGame:
    def __init__(self):
        self.board = chess.Board()
        self.move_count = {'white': 0, 'black': 0}
        self.repetitive_moves = {'white': 0, 'black': 0}
        self.winner = None
        self.winner_points = 0
        self.points = {'white': 0, 'black': 0}
        self.last_update_time = time.time()
        self.update_interval = 0.1  # Atualizar a cada 100ms

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

    def reset(self):
        self.board.reset()
        self.move_count = {'white': 0, 'black': 0}
        self.repetitive_moves = {'white': 0, 'black': 0}
        self.winner = None
        self.winner_points = 0
        self.points = {'white': 0, 'black': 0}

# Funções otimizadas
def calculate_reward_optimized(game, move, color):
    reward = 0
    board = game.board
    
    # Verificar se é captura
    if board.is_capture(move):
        reward += 5
    
    # Verificar se é cheque
    board.push(move)
    if board.is_checkmate():
        reward += 100
    elif board.is_check():
        reward += 10
    
    # Incentivar controle do centro
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    if move.to_square in center_squares:
        reward += 2
    
    # Penalidade para empates
    if board.is_game_over() and board.result() == '1/2-1/2':
        reward -= 50
    
    board.pop()
    return reward

def update_game_optimized(game, agent_white, agent_black, num_simulations):
    current_time = time.time()
    if current_time - game.last_update_time < game.update_interval:
        return
    
    if game.board.is_game_over():
        return
    
    game.last_update_time = current_time
    
    state = board_to_input(game.board)
    legal_moves = list(game.board.legal_moves)
    
    if not legal_moves:
        return
    
    if game.board.turn == chess.WHITE:
        move = optimized_mcts(game.board, num_simulations)
        if move:
            reward = calculate_reward_optimized(game, move, chess.WHITE)
            game.board.push(move)
            next_state = board_to_input(game.board)
            done = game.board.is_game_over()
            agent_white.remember(state, move, reward, next_state, done)
            game.move_count['white'] += 1
    else:
        move = optimized_mcts(game.board, num_simulations)
        if move:
            reward = calculate_reward_optimized(game, move, chess.BLACK)
            game.board.push(move)
            next_state = board_to_input(game.board)
            done = game.board.is_game_over()
            agent_black.remember(state, move, reward, next_state, done)
            game.move_count['black'] += 1
    
    # Verificar repetições
    if game.board.is_repetition():
        if game.board.turn == chess.WHITE:
            game.repetitive_moves['white'] += 1
        else:
            game.repetitive_moves['black'] += 1
    
    if game.repetitive_moves['white'] > 10 or game.repetitive_moves['black'] > 10:
        game.reset()
    
    # Verificar fim do jogo
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
            game.winner_points = -50
            game.points['white'] -= 25
            game.points['black'] -= 25

# Funções de renderização otimizadas
def draw_board_optimized(screen, offset_x, offset_y, square_size):
    # Pré-calcular cores
    colors = [LIGHT_BROWN, DARK_BROWN]
    
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            rect = pygame.Rect(offset_x + col * square_size, offset_y + row * square_size, square_size, square_size)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, HIGHLIGHT, rect, 2)

def draw_pieces_optimized(screen, board, offset_x, offset_y, square_size):
    for row in range(8):
        for col in range(8):
            piece = board.piece_at(chess.square(col, 7 - row))
            if piece:
                piece_image = image_cache.get_scaled_image(piece.symbol(), square_size)
                screen.blit(piece_image, (offset_x + col * square_size, offset_y + row * square_size))

def display_winner_optimized(screen, game, offset_x, offset_y, square_size):
    if game.winner:
        font = pygame.font.Font(None, 36)
        text = f'Vencedor: {game.winner} com {game.winner_points} pontos'
        text_surface = font.render(text, True, BLACK)
        screen.blit(text_surface, (offset_x + 10, offset_y + 8 * square_size + 10))

# Função principal otimizada
def main():
    # Configurações iniciais
    num_simulations = int(input("Quantas simulações você quer iniciar? "))
    
    # Calcular tamanho do tabuleiro
    square_size = min(monitor_width // 8, monitor_height // 8)
    screen_width = 8 * square_size
    screen_height = 8 * square_size
    
    # Criar tela
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
    pygame.display.set_caption('Jogo de Xadrez - Versão Otimizada')
    
    # Inicializar agentes
    agent_white = OptimizedDQNAgent()
    agent_black = OptimizedDQNAgent()
    
    # Carregar modelos existentes
    if os.path.exists('ana_model.h5'):
        agent_white.load('ana_model.h5')
    
    if os.path.exists('pedro_model.h5'):
        agent_black.load('pedro_model.h5')
    
    # Inicializar jogo
    main_game = OptimizedGame()
    
    # Loop principal otimizado
    running = True
    clock = pygame.time.Clock()
    frame_count = 0
    
    while running:
        frame_count += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                screen_width = event.w
                screen_height = event.h
                square_size = min(screen_width // 8, screen_height // 8)
        
        # Atualizar jogo (com throttling)
        update_game_optimized(main_game, agent_white, agent_black, num_simulations)
        
        # Renderizar (apenas a cada 2 frames para melhor performance)
        if frame_count % 2 == 0:
            screen.fill(WHITE)
            draw_board_optimized(screen, 0, 0, square_size)
            draw_pieces_optimized(screen, main_game.board, 0, 0, square_size)
            display_winner_optimized(screen, main_game, 0, 0, square_size)
            pygame.display.flip()
        
        # Treinar agentes (com throttling)
        if frame_count % 10 == 0:
            agent_white.replay()
            agent_black.replay()
        
        # Salvar modelos (com throttling)
        if frame_count % 300 == 0:  # A cada ~5 segundos a 60 FPS
            agent_white.save('ana_model.h5')
            agent_black.save('pedro_model.h5')
        
        # Controlar FPS
        clock.tick(60)
    
    # Salvar modelos ao sair
    agent_white.save('ana_model.h5')
    agent_black.save('pedro_model.h5')
    pygame.quit()

if __name__ == "__main__":
    main()