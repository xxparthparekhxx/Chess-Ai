#!/usr/bin/env python
"""Export models/chess_net.pth to ONNX for the browser version (webgpu/).

What it does
  1. Loads the trained ChessNet in eval mode (CPU).
  2. Folds every BatchNorm into the Conv before it, so the exported graph has
     no BatchNorm nodes. The folded PyTorch model is compared with the original
     before export, and the ONNX file is compared with the original afterwards.
  3. Exports fp32 ONNX with a dynamic batch axis (input "planes", outputs
     "policy" and "value").
  4. Optionally (--fp16-weights) writes a second file whose weights are stored
     as fp16 and cast back to fp32 when the session starts. Compute stays fp32,
     the file is about half the size, and the parity error is measured and
     printed.

Nothing here changes src/. Run it on CPU, from the repo root:

    .venv/bin/python scripts/export_onnx.py
    .venv/bin/python scripts/export_onnx.py --fp16-weights

Extra packages (on top of requirements.txt and torch): see scripts/requirements-export.txt.
Uses torch's TorchScript-based exporter (dynamo=False), which is deprecated in recent
torch versions; if a future torch drops it, the export needs porting to torch.export.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import sys

# The GPU is usually busy with other work and the export does not need it.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import chess  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402
from torch.nn.utils.fusion import fuse_conv_bn_eval  # noqa: E402

from config import MODEL_PATH, NUM_RES_BLOCKS  # noqa: E402
from dataset import ACTION_SIZE, board_to_tensor  # noqa: E402
from model import ChessNet  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(ROOT, "webgpu", "model")
OPSET = 17


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(path=MODEL_PATH):
    model = ChessNet(num_res_blocks=NUM_RES_BLOCKS)
    state = torch.load(path, map_location="cpu")
    if "model" in state:  # a trainer state (model + optimizer)
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def fold_batchnorm(model):
    """Return a copy of `model` with each Conv+BatchNorm pair folded into one Conv."""
    m = copy.deepcopy(model).eval()

    def fold_seq(seq, conv_i, bn_i):
        seq[conv_i] = fuse_conv_bn_eval(seq[conv_i], seq[bn_i])
        seq[bn_i] = nn.Identity()

    fold_seq(m.conv_input, 0, 1)
    for block in m.res_tower:
        block.conv1 = fuse_conv_bn_eval(block.conv1, block.bn1)
        block.bn1 = nn.Identity()
        block.conv2 = fuse_conv_bn_eval(block.conv2, block.bn2)
        block.bn2 = nn.Identity()
    fold_seq(m.policy_head, 0, 1)
    fold_seq(m.value_head, 0, 1)
    return m


def sample_positions(n, seed=1234):
    """Real chess positions (random playouts) so the parity numbers are meaningful."""
    rng = random.Random(seed)
    planes = []
    while len(planes) < n:
        board = chess.Board()
        for _ in range(rng.randint(0, 120)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))
        if not board.is_game_over():
            planes.append(board_to_tensor(board))
    return np.stack(planes)


def torch_forward(model, x):
    with torch.no_grad():
        p, v = model(torch.from_numpy(x))
    return p.numpy(), v.numpy()


def max_err(a, b):
    return float(np.max(np.abs(a - b)))


def ort_forward(path, x):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    p, v = sess.run(["policy", "value"], {"planes": x})
    return p, v


def to_fp16_weights(model_proto, min_elements=1024):
    """Store large fp32 initializers as fp16 + a Cast back to fp32."""
    graph = model_proto.graph
    kept, casts = [], []
    converted = 0
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            if arr.size >= min_elements:
                half = numpy_helper.from_array(arr.astype(np.float16), init.name + "__fp16")
                kept.append(half)
                casts.append(
                    helper.make_node(
                        "Cast", [half.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"
                    )
                )
                converted += 1
                continue
        kept.append(init)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)  # casts first keeps the graph topologically sorted
    return converted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="where to write the .onnx files (default: webgpu/model)")
    ap.add_argument("--name", default="chess_net", help="base file name")
    ap.add_argument("--fp16-weights", action="store_true", help="also write <name>.fp16w.onnx (fp16 weights, fp32 compute)")
    ap.add_argument("--ckpt", default=MODEL_PATH, help="PyTorch checkpoint to export (default: models/chess_net.pth)")
    ap.add_argument("--parity-positions", type=int, default=96, help="number of real positions used for the parity check")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fp32_path = os.path.join(args.out_dir, args.name + ".onnx")

    torch.set_grad_enabled(False)
    print(f"loading {args.ckpt}")
    model = load_model(args.ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    x = sample_positions(args.parity_positions)
    ref_p, ref_v = torch_forward(model, x)

    folded = fold_batchnorm(model)
    fp, fv = torch_forward(folded, x)
    print(f"BN folded vs original (PyTorch): policy max abs err {max_err(ref_p, fp):.3e}, value {max_err(ref_v, fv):.3e}")

    dummy = torch.from_numpy(x[:1])
    torch.onnx.export(
        folded,
        (dummy,),
        fp32_path,
        input_names=["planes"],
        output_names=["policy", "value"],
        dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )

    proto = onnx.load(fp32_path)
    ops = sorted({n.op_type for n in proto.graph.node})
    onnx.checker.check_model(proto)
    # Stamp where the file came from so a stale export is easy to spot.
    for k, v in {
        "source": os.path.relpath(os.path.abspath(args.ckpt), ROOT),
        "source_sha256": sha256_of(args.ckpt),
        "input": "planes float32 [batch,14,8,8]",
        "outputs": f"policy float32 [batch,{ACTION_SIZE}] logits; value float32 [batch,1] tanh, White's point of view",
    }.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(proto, fp32_path)
    print(f"ops in graph: {', '.join(ops)}")
    assert "BatchNormalization" not in ops, "BatchNorm should be folded away"

    report = {
        "source_sha256": sha256_of(args.ckpt),
        "parameters": n_params,
        "opset": OPSET,
        "ops": ops,
        "parity_positions": int(x.shape[0]),
        "files": {},
    }

    def evaluate(path, label):
        size = os.path.getsize(path)
        entry = {"bytes": size, "mib": round(size / 2**20, 2)}
        for batch in (1, 8, 32):
            p, v = ort_forward(path, x[:batch])
            entry[f"batch{batch}_policy_max_abs_err"] = max_err(ref_p[:batch], p)
            entry[f"batch{batch}_value_max_abs_err"] = max_err(ref_v[:batch], v)
        p, v = ort_forward(path, x)
        entry["policy_max_abs_err"] = max_err(ref_p, p)
        entry["value_max_abs_err"] = max_err(ref_v, v)
        # Top-1 policy agreement is what search cares about most.
        entry["policy_argmax_agree"] = float(np.mean(np.argmax(ref_p, 1) == np.argmax(p, 1)))
        report["files"][os.path.basename(path)] = entry
        print(
            f"{label}: {size/2**20:.2f} MiB | vs PyTorch on {x.shape[0]} positions: "
            f"policy max abs err {entry['policy_max_abs_err']:.3e}, value {entry['value_max_abs_err']:.3e}, "
            f"policy argmax agreement {entry['policy_argmax_agree']*100:.1f}%"
        )

    evaluate(fp32_path, "fp32")

    if args.fp16_weights:
        fp16_path = os.path.join(args.out_dir, args.name + ".fp16w.onnx")
        half_proto = onnx.load(fp32_path)
        n = to_fp16_weights(half_proto)
        onnx.checker.check_model(half_proto)
        onnx.save(half_proto, fp16_path)
        print(f"fp16 weights: converted {n} initializers")
        evaluate(fp16_path, "fp16 weights")

    report_path = os.path.join(args.out_dir, "export_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
