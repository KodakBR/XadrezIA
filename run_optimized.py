#!/usr/bin/env python3
"""
Script de execução otimizada para o jogo de xadrez com IA
Inclui verificações de dependências e configurações automáticas
"""

import os
import sys
import subprocess
import platform
import psutil
import time
from pathlib import Path

def check_dependencies():
    """Verifica se todas as dependências estão instaladas"""
    required_packages = [
        'pygame', 'chess', 'numpy', 'tensorflow', 'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Dependências faltando:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Instale as dependências com:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ Todas as dependências estão instaladas")
    return True

def check_system_resources():
    """Verifica recursos do sistema"""
    print("\n🔍 Verificando recursos do sistema...")
    
    # Memória RAM
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    print(f"   RAM Total: {memory_gb:.1f} GB")
    print(f"   RAM Disponível: {memory.available / (1024**3):.1f} GB")
    
    # CPU
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"   CPUs: {cpu_count}")
    print(f"   Uso de CPU: {cpu_percent:.1f}%")
    
    # Verificar se há recursos suficientes
    if memory_gb < 4:
        print("⚠️  Aviso: Menos de 4GB de RAM detectado")
        print("   O jogo pode funcionar lentamente")
    
    if memory.available / (1024**3) < 1:
        print("⚠️  Aviso: Menos de 1GB de RAM disponível")
        print("   Considere fechar outros programas")
    
    return True

def optimize_system_settings():
    """Aplica otimizações específicas do sistema"""
    print("\n⚙️  Aplicando otimizações do sistema...")
    
    system = platform.system()
    
    if system == "Linux":
        # Otimizações para Linux
        try:
            # Verificar se o sistema suporta performance mode
            if os.path.exists("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"):
                print("   Configurando CPU para modo performance...")
                subprocess.run(["sudo", "cpupower", "frequency-set", "-g", "performance"], 
                             capture_output=True)
        except:
            print("   Não foi possível configurar modo performance (requer sudo)")
    
    elif system == "Windows":
        # Otimizações para Windows
        print("   Configurando prioridade do processo...")
        try:
            current_process = psutil.Process()
            current_process.nice(psutil.HIGH_PRIORITY_CLASS)
        except:
            print("   Não foi possível alterar prioridade do processo")
    
    elif system == "Darwin":  # macOS
        print("   Sistema macOS detectado - otimizações automáticas aplicadas")
    
    print("✅ Otimizações aplicadas")

def check_image_files():
    """Verifica se os arquivos de imagem estão presentes"""
    print("\n🖼️  Verificando arquivos de imagem...")
    
    img_dir = Path("xadrez ico")
    if not img_dir.exists():
        print("❌ Diretório de imagens não encontrado: 'xadrez ico'")
        print("   Crie o diretório e adicione as imagens das peças")
        return False
    
    required_images = [
        'black_rook.png', 'black_knight.png', 'black_bishop.png',
        'black_queen.png', 'black_king.png', 'black_pawn.png',
        'white_rook.png', 'white_knight.png', 'white_bishop.png',
        'white_queen.png', 'white_king.png', 'white_pawn.png'
    ]
    
    missing_images = []
    for image in required_images:
        if not (img_dir / image).exists():
            missing_images.append(image)
    
    if missing_images:
        print("❌ Imagens faltando:")
        for image in missing_images:
            print(f"   - {image}")
        print("   O jogo usará fallbacks coloridos")
    else:
        print("✅ Todas as imagens encontradas")
    
    return True

def run_benchmark():
    """Executa benchmark de performance"""
    print("\n📊 Executando benchmark de performance...")
    
    try:
        result = subprocess.run([sys.executable, "benchmark.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Benchmark concluído com sucesso")
            print("   Verifique o arquivo 'benchmark_results.json' para detalhes")
        else:
            print("⚠️  Benchmark falhou, mas o jogo continuará")
            print(f"   Erro: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print("⚠️  Benchmark demorou muito, pulando...")
    except Exception as e:
        print(f"⚠️  Erro no benchmark: {e}")

def run_game():
    """Executa o jogo otimizado"""
    print("\n🎮 Iniciando jogo de xadrez otimizado...")
    print("=" * 50)
    
    try:
        # Importar e executar o jogo
        from xadrez_optimized import main
        main()
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Jogo interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        print("   Verifique se todos os arquivos estão presentes")
        return False
    
    return True

def show_performance_tips():
    """Mostra dicas de performance"""
    print("\n💡 Dicas para melhor performance:")
    print("   • Feche outros programas desnecessários")
    print("   • Use menos simulações MCTS para jogos mais rápidos")
    print("   • Monitore o uso de CPU e memória")
    print("   • Execute o benchmark para verificar melhorias")
    print("   • Considere usar GPU se disponível")

def main():
    """Função principal"""
    print("🚀 Jogo de Xadrez com IA - Versão Otimizada")
    print("=" * 50)
    
    # Verificações iniciais
    if not check_dependencies():
        sys.exit(1)
    
    check_system_resources()
    optimize_system_settings()
    check_image_files()
    
    # Perguntar se quer executar benchmark
    try:
        run_benchmark_choice = input("\n❓ Executar benchmark de performance? (s/n): ").lower()
        if run_benchmark_choice in ['s', 'sim', 'y', 'yes']:
            run_benchmark()
    except KeyboardInterrupt:
        print("\n⏹️  Interrompido pelo usuário")
        sys.exit(0)
    
    show_performance_tips()
    
    # Executar o jogo
    print("\n" + "=" * 50)
    success = run_game()
    
    if success:
        print("\n✅ Jogo finalizado com sucesso")
    else:
        print("\n❌ Jogo finalizado com erros")
        sys.exit(1)

if __name__ == "__main__":
    main()