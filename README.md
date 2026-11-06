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

### 3. Play in the browser (WebGPU, no server)
`webgpu/` is a static web page that runs this same network and search on the visitor's own GPU through
WebGPU, with a WebAssembly (CPU) fallback. Nothing runs on a server except serving files. The Python code
is not changed; two scripts in `scripts/` export the network to ONNX and dump test fixtures from the Python
engine, and the TypeScript port is tested against those fixtures.
```bash
.venv/bin/python scripts/export_onnx.py     # writes webgpu/model/chess_net.onnx (git-ignored)
cd webgpu && npm install && npm run dev
```
Search runs for a time budget (default 2 s per move, adjustable) instead of a fixed 800 simulations. See
[`webgpu/README.md`](webgpu/README.md) for what runs where, build/test/deploy commands, browser support, parity
results with the Python engine, and known limits. The page has not been measured in a browser yet.

### 4. Train the Model
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

## 📏 Measured strength

`eval/run_match.py` plays the Python engine (`src/mcts.py` on the GPU, 800 simulations per move, temperature 0, no
Dirichlet noise, 8 leaves per network call) against Stockfish 19 limited with `UCI_LimitStrength` + `UCI_Elo` (Threads=1,
Hash=16, 100,000 nodes per move). Stockfish is only a test opponent; it is not part of the project. 24 openings
(2–4 random plies, level according to a full-strength Stockfish) are each played with both colours, so every level is 48 games.
A game is adjudicated when a full-strength Stockfish (20,000 nodes) holds a result beyond ±1000 cp for 6 plies in a row;
draw rules are applied and a game still going at 300 plies is drawn. Intervals are 95% Wilson on the score, draws counted
as half. PGNs, per-game JSON and summaries are in `eval/results/<tag>/`.

UCI_Elo is calibrated by Stockfish's developers against other engines at much longer time controls, so the Elo figures
below, from a node-limited match at fast settings, are approximate and say how strong the net is *against this
opponent*, not a rating on any human or federation scale.

Baseline, `models/chess_net.pth` (the net trained on human games only), tag `base`:

| Stockfish UCI_Elo | Games | W / D / L | Score | 95% CI | Elo diff | Implied Elo |
|---|---|---|---|---|---|---|
| 1320 | 48 | 35 / 7 / 6 | 0.802 | 0.67–0.89 | +243 (+122 to +364) | 1563 |
| 1600 | 48 | 23 / 16 / 9 | 0.646 | 0.50–0.77 | +104 (+3 to +206) | 1704 |
| 2000 | 48 | 3 / 18 / 27 | 0.250 | 0.15–0.39 | -191 (-302 to -79) | 1809 |

After two rounds of fine-tuning on Lichess's Stockfish evaluations (`scripts/finetune_lichess.py`; the second round is
`runs/ft2/best.pth`, 23.7M positions), tag `ft2`. This is the network the browser version plays (Hugging Face
`xxparthparekhxx/sigma-go`; the human-games net is its `base` revision):

| Stockfish UCI_Elo | Games | W / D / L | Score | 95% CI |
|---|---|---|---|---|
| 2000 | 48 | 27 / 13 / 8 | 0.698 | 0.56–0.81 |

---

## 🧠 Technical Architecture

**How it thinks:**

1.  **Input**: The board state is converted into a **14-plane tensor** (12 piece-placement planes + 2 side-to-move planes).
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
├── scripts/            # ONNX export and test-fixture generation (for webgpu/)
├── webgpu/             # Browser version: Vite + TypeScript, WebGPU / WASM
└── main.py             # CLI Entry Point
```

---

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a Pull Request.
