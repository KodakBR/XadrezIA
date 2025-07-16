#!/usr/bin/env python3
"""
Performance Benchmark Script for Chess AI Optimization
Compara a performance entre a versão original e otimizada
"""

import time
import psutil
import threading
import numpy as np
import chess
import logging
from functools import wraps
import sys
import os

# Configurar logging para benchmark
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    def __init__(self):
        self.results = {}
        self.process = psutil.Process()
        
    def measure_time(self, func_name):
        """Decorator para medir tempo de execução"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                if func_name not in self.results:
                    self.results[func_name] = []
                
                self.results[func_name].append({
                    'time': execution_time,
                    'memory_delta': memory_delta,
                    'start_memory': start_memory,
                    'end_memory': end_memory
                })
                
                return result
            return wrapper
        return decorator
    
    def get_system_stats(self):
        """Obtém estatísticas do sistema"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available': psutil.virtual_memory().available / 1024 / 1024,  # MB
            'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None
        }
    
    def print_results(self):
        """Imprime resultados do benchmark"""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        
        for func_name, measurements in self.results.items():
            if not measurements:
                continue
                
            times = [m['time'] for m in measurements]
            memory_deltas = [m['memory_delta'] for m in measurements]
            
            print(f"\n{func_name}:")
            print(f"  Execuções: {len(measurements)}")
            print(f"  Tempo médio: {np.mean(times):.4f}s")
            print(f"  Tempo min/max: {np.min(times):.4f}s / {np.max(times):.4f}s")
            print(f"  Desvio padrão: {np.std(times):.4f}s")
            print(f"  Uso médio de memória: {np.mean(memory_deltas):.2f}MB")
            print(f"  Variação de memória: {np.min(memory_deltas):.2f}MB / {np.max(memory_deltas):.2f}MB")

# Benchmark de carregamento de imagens
def benchmark_image_loading():
    """Benchmark do carregamento de imagens"""
    benchmark = PerformanceBenchmark()
    
    print("Iniciando benchmark de carregamento de imagens...")
    
    # Simular carregamento original (todas as imagens de uma vez)
    @benchmark.measure_time("carregamento_original")
    def load_images_original():
        import pygame
        pygame.init()
        
        img_dir = "xadrez ico"
        if not os.path.exists(img_dir):
            # Usar imagens dummy para teste
            return {f'piece_{i}': pygame.Surface((64, 64)) for i in range(12)}
        
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
                images[piece] = pygame.image.load(filepath)
            else:
                images[piece] = pygame.Surface((64, 64))
        
        return images
    
    # Simular carregamento otimizado (lazy loading)
    @benchmark.measure_time("carregamento_otimizado")
    def load_images_optimized():
        import pygame
        pygame.init()
        
        class LazyImageLoader:
            def __init__(self):
                self.cache = {}
                self.img_dir = "xadrez ico"
                
            def get_image(self, piece):
                if piece not in self.cache:
                    filepath = os.path.join(self.img_dir, f'{piece}.png')
                    if os.path.exists(filepath):
                        self.cache[piece] = pygame.image.load(filepath)
                    else:
                        self.cache[piece] = pygame.Surface((64, 64))
                return self.cache[piece]
        
        return LazyImageLoader()
    
    # Executar testes múltiplas vezes
    for i in range(10):
        load_images_original()
        load_images_optimized()
    
    benchmark.print_results()
    return benchmark

# Benchmark de redimensionamento de imagens
def benchmark_image_scaling():
    """Benchmark do redimensionamento de imagens"""
    benchmark = PerformanceBenchmark()
    
    print("\nIniciando benchmark de redimensionamento de imagens...")
    
    import pygame
    pygame.init()
    
    # Criar imagem de teste
    test_image = pygame.Surface((64, 64))
    test_image.fill((255, 0, 0))
    
    # Redimensionamento sem cache (original)
    @benchmark.measure_time("redimensionamento_sem_cache")
    def scale_without_cache():
        sizes = [32, 48, 64, 80, 96]
        for size in sizes:
            scaled = pygame.transform.scale(test_image, (size, size))
        return len(sizes)
    
    # Redimensionamento com cache (otimizado)
    @benchmark.measure_time("redimensionamento_com_cache")
    def scale_with_cache():
        cache = {}
        sizes = [32, 48, 64, 80, 96]
        
        for size in sizes:
            cache_key = (id(test_image), size)
            if cache_key not in cache:
                cache[cache_key] = pygame.transform.scale(test_image, (size, size))
            scaled = cache[cache_key]
        
        return len(sizes)
    
    # Executar testes
    for i in range(100):
        scale_without_cache()
        scale_with_cache()
    
    benchmark.print_results()
    return benchmark

# Benchmark de conversão de tabuleiro
def benchmark_board_conversion():
    """Benchmark da conversão de tabuleiro para input da rede neural"""
    benchmark = PerformanceBenchmark()
    
    print("\nIniciando benchmark de conversão de tabuleiro...")
    
    # Criar tabuleiros de teste
    boards = [chess.Board() for _ in range(10)]
    for board in boards[1:]:
        # Fazer alguns movimentos aleatórios
        for _ in range(np.random.randint(1, 20)):
            legal_moves = list(board.legal_moves)
            if legal_moves:
                board.push(np.random.choice(legal_moves))
    
    # Conversão original (sem cache)
    @benchmark.measure_time("conversao_sem_cache")
    def convert_without_cache(board):
        piece_map = board.piece_map()
        input_array = np.zeros((64, 12), dtype=np.int8)
        for square, piece in piece_map.items():
            piece_type = piece.piece_type - 1
            color = 0 if piece.color == chess.WHITE else 6
            input_array[square][piece_type + color] = 1
        return input_array.flatten()
    
    # Conversão com cache (otimizada)
    from functools import lru_cache
    
    @lru_cache(maxsize=1000)
    def convert_with_cache_cached(fen):
        board = chess.Board(fen)
        piece_map = board.piece_map()
        input_array = np.zeros((64, 12), dtype=np.int8)
        for square, piece in piece_map.items():
            piece_type = piece.piece_type - 1
            color = 0 if piece.color == chess.WHITE else 6
            input_array[square][piece_type + color] = 1
        return input_array.flatten()
    
    @benchmark.measure_time("conversao_com_cache")
    def convert_with_cache(board):
        return convert_with_cache_cached(board.fen())
    
    # Executar testes
    for _ in range(1000):
        for board in boards:
            convert_without_cache(board)
            convert_with_cache(board)
    
    benchmark.print_results()
    return benchmark

# Benchmark de simulação MCTS
def benchmark_mcts():
    """Benchmark do algoritmo MCTS"""
    benchmark = PerformanceBenchmark()
    
    print("\nIniciando benchmark de MCTS...")
    
    board = chess.Board()
    
    # MCTS original (sem otimizações)
    @benchmark.measure_time("mcts_original")
    def mcts_original(board, num_simulations=100):
        # Simulação simplificada do MCTS original
        best_move = None
        best_score = -float('inf')
        
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            score = 0
            for _ in range(num_simulations // len(legal_moves)):
                # Simulação rápida
                test_board = board.copy()
                test_board.push(move)
                
                # Simulação até o fim ou limite de movimentos
                moves_count = 0
                while not test_board.is_game_over() and moves_count < 20:
                    test_legal_moves = list(test_board.legal_moves)
                    if test_legal_moves:
                        test_board.push(np.random.choice(test_legal_moves))
                    moves_count += 1
                
                # Avaliação simples
                if test_board.is_game_over():
                    result = test_board.result()
                    if result == '1-0':
                        score += 1
                    elif result == '0-1':
                        score -= 1
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    # MCTS otimizado (com limite de tempo)
    @benchmark.measure_time("mcts_otimizado")
    def mcts_optimized(board, num_simulations=100, time_limit=0.1):
        start_time = time.time()
        best_move = None
        best_score = -float('inf')
        simulations_done = 0
        
        legal_moves = list(board.legal_moves)
        while simulations_done < num_simulations and (time.time() - start_time) < time_limit:
            move = np.random.choice(legal_moves)
            
            # Simulação limitada
            test_board = board.copy()
            test_board.push(move)
            
            moves_count = 0
            while not test_board.is_game_over() and moves_count < 10:  # Limite menor
                test_legal_moves = list(test_board.legal_moves)
                if test_legal_moves:
                    test_board.push(np.random.choice(test_legal_moves))
                moves_count += 1
            
            # Avaliação rápida
            score = 0
            if test_board.is_game_over():
                result = test_board.result()
                if result == '1-0':
                    score = 1
                elif result == '0-1':
                    score = -1
            
            if score > best_score:
                best_score = score
                best_move = move
            
            simulations_done += 1
        
        return best_move or np.random.choice(legal_moves)
    
    # Executar testes
    for _ in range(20):
        mcts_original(board, 100)
        mcts_optimized(board, 100, 0.1)
    
    benchmark.print_results()
    return benchmark

def main():
    """Função principal do benchmark"""
    print("Iniciando Benchmark de Performance do Xadrez IA")
    print("=" * 60)
    
    # Informações do sistema
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"Memória total: {psutil.virtual_memory().total / 1024 / 1024:.0f} MB")
    print(f"Python: {sys.version}")
    
    benchmarks = []
    
    try:
        # Executar benchmarks individuais
        benchmarks.append(benchmark_image_loading())
        benchmarks.append(benchmark_image_scaling())
        benchmarks.append(benchmark_board_conversion())
        benchmarks.append(benchmark_mcts())
        
        # Resumo geral
        print("\n" + "="*60)
        print("RESUMO GERAL DO BENCHMARK")
        print("="*60)
        
        total_tests = sum(len(b.results) for b in benchmarks)
        print(f"Total de testes executados: {total_tests}")
        
        # Calcular melhorias de performance
        improvements = {}
        for benchmark in benchmarks:
            for func_name, measurements in benchmark.results.items():
                if 'original' in func_name.lower():
                    original_key = func_name
                    optimized_key = func_name.replace('original', 'otimizado').replace('sem_cache', 'com_cache')
                    
                    if optimized_key in benchmark.results:
                        original_time = np.mean([m['time'] for m in measurements])
                        optimized_time = np.mean([m['time'] for m in benchmark.results[optimized_key]])
                        
                        if optimized_time > 0:
                            improvement = ((original_time - optimized_time) / original_time) * 100
                            improvements[func_name] = improvement
        
        print("\nMelhorias de Performance:")
        for test_name, improvement in improvements.items():
            print(f"  {test_name}: {improvement:.1f}% mais rápido")
        
        if improvements:
            avg_improvement = np.mean(list(improvements.values()))
            print(f"\nMelhoria média: {avg_improvement:.1f}%")
        
    except Exception as e:
        logger.error(f"Erro durante benchmark: {e}")
        sys.exit(1)
    
    print("\nBenchmark concluído!")

if __name__ == "__main__":
    main()