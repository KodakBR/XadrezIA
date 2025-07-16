import pygame
import chess
import os
import random
import numpy as np
import tensorflow as tf
from collections import deque
import logging
import threading
import time
import psutil
from functools import lru_cache
import gc

# Configuração otimizada de logging com rotação
from logging.handlers import RotatingFileHandler

# Configurar logging otimizado
logger = logging.getLogger('chess_ai')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('training.log', maxBytes=10*1024*1024, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("Iniciando o treinamento otimizado")

# Configuração do TensorFlow para otimização
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# Inicializar pygame
pygame.init()

# Dimensões do monitor (adaptável)
monitor_width = 1280
monitor_height = 720

# Definir as cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)
GRAY = (169, 169, 169)
HIGHLIGHT = (255, 0, 0)

# Cache para imagens redimensionadas
class ImageCache:
    def __init__(self):
        self.cache = {}
        self.max_cache_size = 100
        
    def get_scaled_image(self, piece_symbol, size):
        cache_key = (piece_symbol, size)
        if cache_key not in self.cache:
            if len(self.cache) >= self.max_cache_size:
                # Limpar cache mais antigo
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            original_image = PIECE_IMAGES[piece_symbol]
            scaled_image = pygame.transform.scale(original_image, (size, size))
            self.cache[cache_key] = scaled_image
            
        return self.cache[cache_key]
    
    def clear_cache(self):
        self.cache.clear()
        gc.collect()

# Carregamento lazy de imagens com verificação
def load_piece_images():
    img_dir = "xadrez ico"
    if not os.path.exists(img_dir):
        # Fallback para diretório absoluto se relativo não existir
        img_dir = "D:/projeto/xadrezIA/xadrez ico"
    
    piece_files = {
        'r': 'black_rook.png', 'n': 'black_knight.png', 'b': 'black_bishop.png',
        'q': 'black_queen.png', 'k': 'black_king.png', 'p': 'black_pawn.png',
        'R': 'white_rook.png', 'N': 'white_knight.png', 'B': 'white_bishop.png',
        'Q': 'white_queen.png', 'K': 'white_king.png', 'P': 'white_pawn.png'
    }
    
    images = {}
    for piece, filename in piece_files.items():
        filepath = os.path.join(img_dir, filename)
        if os.path.exists(filepath):
            try:
                images[piece] = pygame.image.load(filepath).convert_alpha()
            except pygame.error as e:
                logger.error(f"Erro ao carregar {filepath}: {e}")
                # Criar uma imagem de fallback simples
                images[piece] = pygame.Surface((64, 64))
                images[piece].fill(GRAY)
        else:
            logger.warning(f"Arquivo não encontrado: {filepath}")
            images[piece] = pygame.Surface((64, 64))
            images[piece].fill(GRAY)
    
    return images

# Carregar imagens uma única vez
PIECE_IMAGES = load_piece_images()
image_cache = ImageCache()

# Função otimizada para desenhar o tabuleiro (usando surface para cache)
def create_board_surface(square_size):
    surface = pygame.Surface((8 * square_size, 8 * square_size))
    for row in range(8):
        for col in range(8):
            color = LIGHT_BROWN if (row + col) % 2 == 0 else DARK_BROWN
            pygame.draw.rect(surface, color, 
                           pygame.Rect(col * square_size, row * square_size, square_size, square_size))
            pygame.draw.rect(surface, HIGHLIGHT, 
                           pygame.Rect(col * square_size, row * square_size, square_size, square_size), 2)
    return surface

# Cache do tabuleiro
board_surface_cache = {}

def draw_board(screen, offset_x, offset_y, square_size):
    if square_size not in board_surface_cache:
        board_surface_cache[square_size] = create_board_surface(square_size)
    
    screen.blit(board_surface_cache[square_size], (offset_x, offset_y))

# Função otimizada para desenhar as peças
def draw_pieces(screen, board, offset_x, offset_y, square_size):
    for row in range(8):
        for col in range(8):
            piece = board.piece_at(chess.square(col, 7 - row))
            if piece:
                piece_image = image_cache.get_scaled_image(piece.symbol(), square_size)
                screen.blit(piece_image, (offset_x + col * square_size, offset_y + row * square_size))

# Função otimizada para conversão do tabuleiro
@lru_cache(maxsize=1000)
def board_to_input_cached(fen):
    board = chess.Board(fen)
    piece_map = board.piece_map()
    input_array = np.zeros((64, 12), dtype=np.int8)
    for square, piece in piece_map.items():
        piece_type = piece.piece_type - 1
        color = 0 if piece.color == chess.WHITE else 6
        input_array[square][piece_type + color] = 1
    return input_array.flatten()

def board_to_input(board):
    return board_to_input_cached(board.fen())

# Função de perda otimizada
@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

# Classe otimizada do Agente DQN
class OptimizedDQNAgent:
    def __init__(self):
        self.model = self.create_optimized_model()
        self.target_model = self.create_optimized_model()
        self.update_target_model()
        self.memory = deque(maxlen=1000)  # Reduzido para economia de memória
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 16  # Reduzido para melhor performance
        self.train_start = 500
        self.last_save_time = time.time()
        self.save_interval = 300  # Salvar a cada 5 minutos

    def create_optimized_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(768,)),
            tf.keras.layers.Dense(256, activation='relu'),  # Reduzido de 512
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),  # Camada adicional menor
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        # Otimizador com learning rate adaptativo
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss=custom_mse, optimizer=optimizer)
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act_batch_optimized(self, state, legal_moves):
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_moves)
        
        # Batch prediction para todos os movimentos legais
        states_batch = []
        for move in legal_moves:
            next_state = chess.Board(state.fen())
            next_state.push(move)
            next_state_input = board_to_input(next_state)
            states_batch.append(next_state_input)
        
        if states_batch:
            # Predição em lote (muito mais eficiente)
            predictions = self.model.predict(np.array(states_batch), verbose=0)
            best_move_idx = np.argmax(predictions)
            return legal_moves[best_move_idx]
        
        return random.choice(legal_moves)

    def replay_optimized(self):
        if len(self.memory) < self.train_start:
            return
        
        minibatch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        
        # Preparar dados em lote
        states = np.array([transition[0] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        
        # Predições em lote
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(next_q_values[i])
            current_q_values[i][0] = target
        
        # Treinamento em lote
        self.model.fit(states, current_q_values, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def should_save(self):
        return time.time() - self.last_save_time > self.save_interval

    def save_optimized(self, filename):
        if self.should_save():
            try:
                # Salvar com compressão
                self.model.save(filename, save_format='h5', include_optimizer=False)
                self.last_save_time = time.time()
                logger.info(f"Modelo salvo: {filename}")
            except Exception as e:
                logger.error(f"Erro ao salvar modelo {filename}: {e}")

    def load_optimized(self, filename):
        if os.path.exists(filename):
            try:
                self.model = tf.keras.models.load_model(filename, custom_objects={'custom_mse': custom_mse})
                self.target_model = tf.keras.models.load_model(filename, custom_objects={'custom_mse': custom_mse})
                logger.info(f"Modelo carregado: {filename}")
            except Exception as e:
                logger.error(f"Erro ao carregar modelo {filename}: {e}")

# Classe para MCTS otimizado com threading
class OptimizedMCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.visits = 0
        self.wins = 0
        self.children = []
        self._is_expanded = False

    def is_fully_expanded(self):
        if not self._is_expanded:
            self._is_expanded = len(self.children) == len(list(self.board.legal_moves))
        return self._is_expanded

    def best_child(self, exploration_weight=1.0):
        if not self.children:
            return None
        
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                choices_weights.append(float('inf'))
            else:
                exploitation = child.wins / child.visits
                exploration = exploration_weight * np.sqrt(2 * np.log(self.visits) / child.visits)
                choices_weights.append(exploitation + exploration)
        
        return self.children[np.argmax(choices_weights)]

# MCTS otimizado com limite de tempo
def mcts_optimized(board, max_simulations=1000, time_limit=1.0):
    root = OptimizedMCTSNode(board)
    start_time = time.time()
    simulations_done = 0
    
    while simulations_done < max_simulations and (time.time() - start_time) < time_limit:
        node = root
        
        # Fase de seleção otimizada
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            if node is None:
                break

        # Fase de expansão otimizada
        if not node.is_fully_expanded() and not node.board.is_game_over():
            legal_moves = list(node.board.legal_moves)
            existing_moves = {child.move for child in node.children}
            
            for move in legal_moves:
                if move not in existing_moves:
                    new_board = node.board.copy()
                    new_board.push(move)
                    new_child = OptimizedMCTSNode(new_board, node, move)
                    node.children.append(new_child)
                    node = new_child
                    break

        # Fase de simulação otimizada (limitada)
        current_board = node.board.copy()
        moves_limit = 50  # Limitar simulações muito longas
        moves_count = 0
        
        while not current_board.is_game_over() and moves_count < moves_limit:
            legal_moves = list(current_board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            current_board.push(move)
            moves_count += 1

        # Fase de retropropagação
        result = current_board.result()
        reward = 1 if result == '1-0' else -1 if result == '0-1' else 0
        
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent
        
        simulations_done += 1

    # Retornar melhor movimento
    if root.children:
        best_child = root.best_child(exploration_weight=0)
        return best_child.move if best_child else random.choice(list(board.legal_moves))
    
    return random.choice(list(board.legal_moves))

# Classe de jogo otimizada
class OptimizedGame:
    def __init__(self):
        self.board = chess.Board()
        self.move_count = {'white': 0, 'black': 0}
        self.repetitive_moves = {'white': 0, 'black': 0}
        self.winner = None
        self.winner_points = 0
        self.points = {'white': 0, 'black': 0}
        self.last_cleanup = time.time()

    def cleanup_if_needed(self):
        # Limpeza periódica de memória
        if time.time() - self.last_cleanup > 60:  # A cada minuto
            gc.collect()
            board_to_input_cached.cache_clear()
            self.last_cleanup = time.time()

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

# Monitor de performance
class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0

    def update(self):
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_update >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time

    def get_stats(self):
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        cpu_usage = psutil.cpu_percent()
        return {
            'fps': self.current_fps,
            'memory_mb': memory_usage,
            'cpu_percent': cpu_usage,
            'uptime': time.time() - self.start_time
        }

# Inicializar componentes otimizados
agent_white = OptimizedDQNAgent()
agent_black = OptimizedDQNAgent()
performance_monitor = PerformanceMonitor()

# Carregar modelos se existirem
agent_white.load_optimized('ana_model.h5')
agent_black.load_optimized('pedro_model.h5')

# Função otimizada para atualizar o jogo
def update_game_optimized(game, agent_white, agent_black, num_simulations=500):
    if game.board.is_game_over():
        return

    state = board_to_input(game.board)
    legal_moves = list(game.board.legal_moves)

    if game.board.turn == chess.WHITE:
        # Usar MCTS otimizado com limite de tempo
        move = mcts_optimized(game.board, num_simulations, time_limit=0.5)
        reward = calculate_reward_optimized(game, move, chess.WHITE)
        game.board.push(move)
        next_state = board_to_input(game.board)
        done = game.board.is_game_over()
        agent_white.remember(state, move, reward, next_state, done)
        game.move_count['white'] += 1
    else:
        move = mcts_optimized(game.board, num_simulations, time_limit=0.5)
        reward = calculate_reward_optimized(game, move, chess.BLACK)
        game.board.push(move)
        next_state = board_to_input(game.board)
        done = game.board.is_game_over()
        agent_black.remember(state, move, reward, next_state, done)
        game.move_count['black'] += 1

    # Verificação otimizada de repetição
    if game.board.is_repetition():
        color_key = 'white' if game.board.turn == chess.BLACK else 'black'
        game.repetitive_moves[color_key] += 1

    if game.repetitive_moves['white'] > 10 or game.repetitive_moves['black'] > 10:
        game.board.reset()
        game.move_count = {'white': 0, 'black': 0}
        game.repetitive_moves = {'white': 0, 'black': 0}

    # Verificação de fim de jogo otimizada
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

# Função otimizada para calcular recompensa
def calculate_reward_optimized(game, move, color):
    reward = 1  # Recompensa base
    
    # Cache temporário do estado do tabuleiro
    original_board = game.board.copy()
    game.board.push(move)
    
    try:
        if game.board.is_checkmate():
            reward = 100
        elif game.board.is_check():
            reward = 10
        elif original_board.is_capture(move):
            reward = 5
        
        # Verificações otimizadas
        if game.board.is_repetition():
            reward -= 5
        
        # Controle do centro (cálculo direto)
        center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        if move.to_square in center_squares:
            reward += 2
        
        if game.board.is_game_over() and game.board.result() == '1/2-1/2':
            reward -= 50
            
    finally:
        # Restaurar estado original
        game.board = original_board
    
    return reward

# Função otimizada para exibir informações
def display_info_optimized(screen, game, performance_monitor, offset_x, offset_y, square_size):
    font = pygame.font.Font(None, 24)
    y_offset = offset_y + 8 * square_size + 10
    
    # Informações do jogo
    if game.winner:
        text = f'Vencedor: {game.winner} ({game.winner_points} pts)'
        text_surface = font.render(text, True, BLACK)
        screen.blit(text_surface, (offset_x + 10, y_offset))
        y_offset += 25
    
    # Informações de performance
    stats = performance_monitor.get_stats()
    perf_text = f'FPS: {stats["fps"]:.1f} | RAM: {stats["memory_mb"]:.1f}MB | CPU: {stats["cpu_percent"]:.1f}%'
    perf_surface = font.render(perf_text, True, BLACK)
    screen.blit(perf_surface, (offset_x + 10, y_offset))

# Configuração inicial otimizada
def main():
    # Configuração adaptável da tela
    square_size = min(monitor_width // 10, monitor_height // 10)
    screen_width = max(8 * square_size, 800)
    screen_height = max(8 * square_size + 100, 600)
    
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
    pygame.display.set_caption('Xadrez IA Otimizado')
    
    # Perguntar número de simulações (com valor padrão otimizado)
    try:
        num_simulations = int(input("Simulações por movimento (padrão 500): ") or "500")
        num_simulations = min(num_simulations, 2000)  # Limite máximo
    except ValueError:
        num_simulations = 500
    
    # Inicializar jogo
    main_game = OptimizedGame()
    clock = pygame.time.Clock()
    
    logger.info(f"Jogo iniciado com {num_simulations} simulações por movimento")
    
    # Loop principal otimizado
    running = True
    frame_skip = 0  # Para controlar rendering
    
    while running:
        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                screen_width, screen_height = event.w, event.h
                square_size = min(screen_width // 10, screen_height // 10)
                # Limpar cache do tabuleiro para nova resolução
                board_surface_cache.clear()
        
        # Lógica do jogo
        update_game_optimized(main_game, agent_white, agent_black, num_simulations)
        
        # Rendering otimizado (skip frames se necessário)
        frame_skip += 1
        if frame_skip >= 2:  # Renderizar a cada 2 frames para melhor performance
            frame_skip = 0
            
            screen.fill(WHITE)
            draw_board(screen, 0, 0, square_size)
            draw_pieces(screen, main_game.board, 0, 0, square_size)
            display_info_optimized(screen, main_game, performance_monitor, 0, 0, square_size)
            
            pygame.display.flip()
        
        # Treinamento em thread separada (não bloqueante)
        if frame_skip == 0:
            threading.Thread(target=agent_white.replay_optimized, daemon=True).start()
            threading.Thread(target=agent_black.replay_optimized, daemon=True).start()
        
        # Salvamento inteligente
        agent_white.save_optimized('ana_model.h5')
        agent_black.save_optimized('pedro_model.h5')
        
        # Limpeza periódica
        main_game.cleanup_if_needed()
        
        # Atualizar monitor de performance
        performance_monitor.update()
        
        # Controlar FPS
        clock.tick(60)  # 60 FPS máximo
    
    # Cleanup final
    pygame.quit()
    agent_white.save_optimized('ana_model.h5')
    agent_black.save_optimized('pedro_model.h5')
    logger.info("Jogo finalizado e modelos salvos")

if __name__ == "__main__":
    main()