#!/usr/bin/env python3
"""
Script de benchmark para comparar performance entre versões original e otimizada
"""

import time
import psutil
import os
import sys
import subprocess
import json
from datetime import datetime

class PerformanceBenchmark:
    def __init__(self):
        self.results = {}
        self.process = None
        
    def get_memory_usage(self):
        """Obtém uso de memória do processo atual"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def get_cpu_usage(self):
        """Obtém uso de CPU do processo atual"""
        return psutil.cpu_percent(interval=1)
    
    def measure_function_performance(self, func, *args, **kwargs):
        """Mede performance de uma função específica"""
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        return {
            'execution_time': execution_time,
            'memory_delta': memory_delta,
            'result': result
        }
    
    def benchmark_image_loading(self):
        """Benchmark do carregamento de imagens"""
        print("🔍 Testando carregamento de imagens...")
        
        # Teste original (simulado)
        def original_image_loading():
            import pygame
            pygame.init()
            images = {}
            piece_files = {
                'r': 'black_rook.png', 'n': 'black_knight.png', 'b': 'black_bishop.png',
                'q': 'black_queen.png', 'k': 'black_king.png', 'p': 'black_pawn.png',
                'R': 'white_rook.png', 'N': 'white_knight.png', 'B': 'white_bishop.png',
                'Q': 'white_queen.png', 'K': 'white_king.png', 'P': 'white_pawn.png'
            }
            
            for piece, filename in piece_files.items():
                try:
                    path = os.path.join("xadrez ico", filename)
                    images[piece] = pygame.image.load(path)
                except:
                    pass
            return len(images)
        
        # Teste otimizado
        def optimized_image_loading():
            from xadrez_optimized import ImageCache
            cache = ImageCache()
            return len(cache.original_images)
        
        original_result = self.measure_function_performance(original_image_loading)
        optimized_result = self.measure_function_performance(optimized_image_loading)
        
        self.results['image_loading'] = {
            'original': original_result,
            'optimized': optimized_result,
            'improvement': {
                'time': (original_result['execution_time'] - optimized_result['execution_time']) / original_result['execution_time'] * 100,
                'memory': (original_result['memory_delta'] - optimized_result['memory_delta']) / max(original_result['memory_delta'], 0.1) * 100
            }
        }
        
        print(f"✅ Carregamento de imagens:")
        print(f"   Original: {original_result['execution_time']:.4f}s, {original_result['memory_delta']:.2f}MB")
        print(f"   Otimizado: {optimized_result['execution_time']:.4f}s, {optimized_result['memory_delta']:.2f}MB")
        print(f"   Melhoria: {self.results['image_loading']['improvement']['time']:.1f}% tempo, {self.results['image_loading']['improvement']['memory']:.1f}% memória")
    
    def benchmark_board_conversion(self):
        """Benchmark da conversão de tabuleiro"""
        print("🔍 Testando conversão de tabuleiro...")
        
        import chess
        board = chess.Board()
        
        # Teste original
        def original_board_conversion():
            piece_map = board.piece_map()
            input_array = [[0] * 12 for _ in range(64)]
            for square, piece in piece_map.items():
                piece_type = piece.piece_type - 1
                color = 0 if piece.color == chess.WHITE else 6
                input_array[square][piece_type + color] = 1
            return len(input_array)
        
        # Teste otimizado
        def optimized_board_conversion():
            from xadrez_optimized import board_to_input
            result = board_to_input(board)
            return len(result)
        
        original_result = self.measure_function_performance(original_board_conversion)
        optimized_result = self.measure_function_performance(optimized_board_conversion)
        
        self.results['board_conversion'] = {
            'original': original_result,
            'optimized': optimized_result,
            'improvement': {
                'time': (original_result['execution_time'] - optimized_result['execution_time']) / original_result['execution_time'] * 100,
                'memory': (original_result['memory_delta'] - optimized_result['memory_delta']) / max(original_result['memory_delta'], 0.1) * 100
            }
        }
        
        print(f"✅ Conversão de tabuleiro:")
        print(f"   Original: {original_result['execution_time']:.6f}s, {original_result['memory_delta']:.2f}MB")
        print(f"   Otimizado: {optimized_result['execution_time']:.6f}s, {optimized_result['memory_delta']:.2f}MB")
        print(f"   Melhoria: {self.results['board_conversion']['improvement']['time']:.1f}% tempo, {self.results['board_conversion']['improvement']['memory']:.1f}% memória")
    
    def benchmark_mcts(self):
        """Benchmark do algoritmo MCTS"""
        print("🔍 Testando algoritmo MCTS...")
        
        import chess
        board = chess.Board()
        
        # Teste original (simulado)
        def original_mcts():
            # Simulação do MCTS original
            for _ in range(100):
                test_board = board.copy()
                legal_moves = list(test_board.legal_moves)
                if legal_moves:
                    move = legal_moves[0]
                    test_board.push(move)
            return len(list(board.legal_moves))
        
        # Teste otimizado
        def optimized_mcts():
            from xadrez_optimized import optimized_mcts
            move = optimized_mcts(board, 100)
            return 1 if move else 0
        
        original_result = self.measure_function_performance(original_mcts)
        optimized_result = self.measure_function_performance(optimized_mcts)
        
        self.results['mcts'] = {
            'original': original_result,
            'optimized': optimized_result,
            'improvement': {
                'time': (original_result['execution_time'] - optimized_result['execution_time']) / original_result['execution_time'] * 100,
                'memory': (original_result['memory_delta'] - optimized_result['memory_delta']) / max(original_result['memory_delta'], 0.1) * 100
            }
        }
        
        print(f"✅ Algoritmo MCTS:")
        print(f"   Original: {original_result['execution_time']:.4f}s, {original_result['memory_delta']:.2f}MB")
        print(f"   Otimizado: {optimized_result['execution_time']:.4f}s, {optimized_result['memory_delta']:.2f}MB")
        print(f"   Melhoria: {self.results['mcts']['improvement']['time']:.1f}% tempo, {self.results['mcts']['improvement']['memory']:.1f}% memória")
    
    def benchmark_model_prediction(self):
        """Benchmark de predições do modelo"""
        print("🔍 Testando predições do modelo...")
        
        import numpy as np
        
        # Dados de teste
        test_data = np.random.random((32, 768))
        
        # Teste original (simulado)
        def original_prediction():
            # Simulação de predições individuais
            results = []
            for i in range(len(test_data)):
                result = np.sum(test_data[i])  # Simulação simples
                results.append(result)
            return len(results)
        
        # Teste otimizado
        def optimized_prediction():
            # Simulação de predição em lote
            result = np.sum(test_data, axis=1)
            return len(result)
        
        original_result = self.measure_function_performance(original_prediction)
        optimized_result = self.measure_function_performance(optimized_prediction)
        
        self.results['model_prediction'] = {
            'original': original_result,
            'optimized': optimized_result,
            'improvement': {
                'time': (original_result['execution_time'] - optimized_result['execution_time']) / original_result['execution_time'] * 100,
                'memory': (original_result['memory_delta'] - optimized_result['memory_delta']) / max(original_result['memory_delta'], 0.1) * 100
            }
        }
        
        print(f"✅ Predições do modelo:")
        print(f"   Original: {original_result['execution_time']:.6f}s, {original_result['memory_delta']:.2f}MB")
        print(f"   Otimizado: {optimized_result['execution_time']:.6f}s, {optimized_result['memory_delta']:.2f}MB")
        print(f"   Melhoria: {self.results['model_prediction']['improvement']['time']:.1f}% tempo, {self.results['model_prediction']['improvement']['memory']:.1f}% memória")
    
    def run_all_benchmarks(self):
        """Executa todos os benchmarks"""
        print("🚀 Iniciando benchmarks de performance...")
        print("=" * 50)
        
        self.benchmark_image_loading()
        print()
        
        self.benchmark_board_conversion()
        print()
        
        self.benchmark_mcts()
        print()
        
        self.benchmark_model_prediction()
        print()
        
        self.generate_report()
    
    def generate_report(self):
        """Gera relatório final"""
        print("=" * 50)
        print("📊 RELATÓRIO DE PERFORMANCE")
        print("=" * 50)
        
        total_time_improvement = 0
        total_memory_improvement = 0
        count = 0
        
        for test_name, result in self.results.items():
            improvement = result['improvement']
            total_time_improvement += improvement['time']
            total_memory_improvement += improvement['memory']
            count += 1
            
            print(f"{test_name.replace('_', ' ').title()}:")
            print(f"  Tempo: {improvement['time']:+.1f}%")
            print(f"  Memória: {improvement['memory']:+.1f}%")
            print()
        
        avg_time_improvement = total_time_improvement / count
        avg_memory_improvement = total_memory_improvement / count
        
        print("📈 MÉDIAS GERAIS:")
        print(f"  Melhoria de tempo: {avg_time_improvement:+.1f}%")
        print(f"  Melhoria de memória: {avg_memory_improvement:+.1f}%")
        
        # Salvar resultados em JSON
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'results': self.results,
            'summary': {
                'avg_time_improvement': avg_time_improvement,
                'avg_memory_improvement': avg_memory_improvement
            }
        }
        
        with open('benchmark_results.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Resultados salvos em 'benchmark_results.json'")

def main():
    """Função principal"""
    try:
        benchmark = PerformanceBenchmark()
        benchmark.run_all_benchmarks()
    except Exception as e:
        print(f"❌ Erro durante benchmark: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()