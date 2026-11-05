"""Throughput of ChessNet inference (fp32/fp16, batch sizes) and of the Python search's per-leaf CPU work."""
import os, sys, time
import numpy as np, torch, chess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import ChessNet
from dataset import board_to_tensor

torch.backends.cudnn.benchmark = True
net = ChessNet(10).cuda().eval()
net.load_state_dict(torch.load(os.path.join(ROOT, "models/chess_net.pth"), map_location="cuda"))
for dtype in (torch.float32, torch.float16):
    for cl in (False, True):
        m = net.to(dtype)
        if cl: m = m.to(memory_format=torch.channels_last)
        for bs in (64, 256, 512, 1024):
            x = torch.randn(bs, 14, 8, 8, device="cuda", dtype=dtype)
            if cl: x = x.contiguous(memory_format=torch.channels_last)
            with torch.no_grad():
                for _ in range(3): m(x)
                torch.cuda.synchronize(); t = time.time(); n = 0
                while time.time() - t < 3: m(x); n += bs
                torch.cuda.synchronize()
            print(f"{str(dtype):14s} channels_last={cl!s:5s} batch {bs:5d}: {n/(time.time()-t):8.0f} evals/s", flush=True)
net.float()
# CPU per-leaf work in the current search: copy, push, is_game_over, legal_moves, encode
b = chess.Board(); [b.push(m) for m in [chess.Move.from_uci(u) for u in "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8c5".split()]]
t = time.time(); N = 3000
for _ in range(N):
    c = b.copy(); c.push(next(iter(c.legal_moves))); c.is_game_over(); list(c.legal_moves); board_to_tensor(c)
print(f"python-chess leaf work: {(time.time()-t)/N*1e6:.0f} us/leaf on one core")
