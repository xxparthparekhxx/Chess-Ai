# ♟️ Sigma-Go (ChessAI)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**Sigma-Go** is a modern, deep-learning-based Chess AI built from scratch. It combines a **ResNet (Residual Neural Network)** for intuition with **Monte Carlo Tree Search (MCTS)** for calculation, mimicking the architecture of DeepMind's AlphaZero.

It features a polished **Web Interface** with a premium dark-mode aesthetic, move hints, and analyzing tools.

---

## ✨ Features

*   **🧠 Advanced AI Architecture**:
    *   **ResNet Model**: A custom 10-block Residual Network that predicts move probabilities (Policy) and win chances (Value).
    *   **MCTS (Monte Carlo Tree Search)**: Simulates 800+ futures per move to find forced checkmates and tactical combinations.
*   **🖥️ Modern Web Interface**:
    *   **Glassmorphism Design**: Sleek, translucent UI with neon accents.
    *   **Interactive Board**: Drag-and-drop moves, move hints (dots), and instant evaluations.
    *   **Game Controls**: Play as White/Black, Undo moves, and Flip board.
    *   **Real-time Analysis**: Visual confidence bar and evaluation score.
*   **⚡ Optimized Performance**:
    *   **GPU Acceleration**: Fully utilizes CUDA for fast inference.
    *   **Smart Data Loading**: Processes millions of games using memory-efficient streaming.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.8+
*   NVidia GPU (Recommended for faster thinking)

### Setup
1.  **Clone the repository**
    ```bash
    git clone https://github.com/xxparthparekhxx/Chess-Ai.git
    cd Chess-Ai
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have the correct PyTorch version for your CUDA hardware)*

---

## 🚀 Usage

### 1. Play vs AI (Web Interface) - *Recommended*
Launch the full graphical interface.
```bash
python web/app.py
```
*   Open your browser to `http://127.0.0.1:5000`
*   **White Start**: You play White by default.
*   **Select Side**: Click "Play as Black" to switch sides (AI moves first).

### 2. Play vs AI (Command Line)
For a lightweight terminal experience.
```bash
python main.py
```

### 3. Train the Model
To train the AI on your own PGN datasets:
1.  Place `.pgn` files in `data/raw/`.
2.  Run preprocessing:
    ```bash
    python src/preprocess.py
    ```
3.  Start training:
    ```bash
    python src/train.py
    ```
    *   Checkpoints are saved to `models/chess_net.pth`.

---

## 🧠 Technical Architecture

**How it thinks:**

1.  **Input**: The board state is converted into a **14-plane tensor** (Piece types x Colors + Repetition planes).
2.  **ResNet Evaluator**:
    *   The board is fed into the Neural Network.
    *   **Output 1 (Policy)**: A probability distribution over all legal moves (Intuition).
    *   **Output 2 (Value)**: A scalar between -1 (Black Wins) and +1 (White Wins).
3.  **MCTS Simulation**:
    *   Instead of just taking the highest probability move, the AI runs **800 simulations**.
    *   It expands a search tree, balancing **Exploration** (trying new moves) vs **Exploitation** (playing known good moves).
    *   This allows it to "read" the future and solve complex endgame puzzles that the raw neural network might miss.

---

## 📂 Project Structure

```
Sigma-Go/
├── data/               # Raw PGNs and Processed Tensors
├── models/             # Saved PyTorch Models (.pth)
├── src/
│   ├── config.py       # Hyperparameters (Batch Size, LR)
│   ├── dataset.py      # Board encoding & logic
│   ├── mcts.py         # Monte Carlo Tree Search Logic
│   ├── model.py        # PyTorch ResNet Architecture
│   ├── preprocess.py   # Data pipeline
│   └── train.py        # Training loop
├── web/
│   ├── static/         # CSS, JS, and Assets
│   ├── templates/      # HTML Files
│   └── app.py          # Flask Backend
└── main.py             # CLI Entry Point
```

---

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a Pull Request.
