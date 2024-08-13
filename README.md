# Chess AI with Monte Carlo Tree Search and Deep Q-Learning

Este é um projeto de Xadrez com Inteligência Artificial utilizando uma combinação de **Monte Carlo Tree Search (MCTS)** e **Deep Q-Learning (DQN)**. O objetivo é criar agentes que possam jogar xadrez em nível competitivo, melhorando suas habilidades através de simulações e aprendizado por reforço.

## 🚀 Tecnologias Utilizadas

- **Python**: Linguagem principal do projeto.
- **Pygame**: Biblioteca usada para a interface gráfica do jogo.
- **TensorFlow**: Framework utilizado para a criação e treinamento da rede neural.
- **Keras**: API de alto nível para construção da rede neural.
- **Chess**: Biblioteca para a manipulação e controle das regras do xadrez.

## 🧠 Inteligência Artificial

### Deep Q-Learning (DQN)
O DQN é utilizado para treinar dois agentes: **Ana** (peças brancas) e **Pedro** (peças pretas). Cada agente utiliza uma rede neural para prever a melhor jogada com base no estado atual do tabuleiro.

### Monte Carlo Tree Search (MCTS)
O MCTS é usado para realizar simulações em segundo plano e ajudar os agentes a escolherem os melhores movimentos possíveis. Para cada jogada do jogo principal, várias simulações são realizadas para determinar o movimento com a maior pontuação.

## ⚙️ Como Funciona

1. **Tabuleiro Principal**: O jogo principal é exibido na tela e jogado entre os dois agentes treinados.
2. **Simulações em Background**: Para cada jogada no tabuleiro principal, o agente realiza várias simulações em segundo plano para determinar o melhor movimento.
3. **Treinamento Contínuo**: A cada jogada, os agentes são treinados com base nos resultados das simulações, e os modelos são salvos para uso posterior.

## 🛠 Como Executar

1. **Clone o Repositório**
    ```bash
    git clone https://github.com/KodakBR/XadrezIA)
    ```

2. **Instale as Dependências**
    ```bash
    pip install -r requirements.txt
    ```

3. **Execute o Jogo**
    ```bash
    python xadrez.py
    ```

4. **Escolha o Número de Simulações**
    Ao iniciar o programa, você será solicitado a inserir o número de simulações para cada jogada. Essas simulações são realizadas para ambos os agentes em segundo plano.

## 🖥 Interface Gráfica

A interface gráfica exibe apenas o tabuleiro principal onde o jogo acontece. As simulações em background são realizadas sem exibição gráfica para otimizar o desempenho.

### Controles
- O tabuleiro principal é atualizado automaticamente conforme os agentes fazem suas jogadas.
- O resultado de cada jogada e o movimento realizado são exibidos no terminal.

## 📂 Estrutura do Projeto

```plaintext
.
├── README.md           # Documentação do projeto
├── xadrez.py           # Código principal do jogo e inteligência artificial
├── requirements.txt    # Dependências do projeto
├── training.log        # Log do treinamento dos agentes
└── xadrez ico/         # Diretório contendo as imagens das peças

📈 Resultados
Modelos Treinados: Os modelos dos agentes são salvos em ana_model.h5 e pedro_model.h5, sendo continuamente aprimorados com o tempo.
Logs de Treinamento: Os resultados e o progresso do treinamento são registrados no arquivo training.log.

🤝 Contribuições
Sinta-se à vontade para contribuir com melhorias no código, relatórios de bugs ou novas funcionalidades através de pull requests.

📝 Créditos
Este projeto foi desenvolvido por Kein Soares.
