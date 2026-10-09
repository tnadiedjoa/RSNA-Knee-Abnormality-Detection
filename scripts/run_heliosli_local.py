"""Run heliosli's compact pipeline (stage 2 of heliosli/rsna-knee-blend-gold) locally on a Kaggle-like test folder.

    python -m scripts.run_heliosli_local <public notebook .ipynb> <bundle dir> <code dir> <data dir> <out dir>
        [--arm E_lrprior] [--backbone convnext_small] [--env /tmp/heliosli_env]

Same calls as the notebook's cells 4-5: its embedded sources (SHA-256 checked) written to <out>/src, its pinned wheels
(pydicom 3.0.1, timm 1.0.25) installed in a separate target dir put first on PYTHONPATH, the controller probe_dual.py
run with the compact worker and the bundle manifest's settings. --arm / --backbone select the second model of a dual
bundle (g2_E_lrprior, vit_base_patch16_dinov3.lvd1689m). Used to time the pipeline and to score heliosli's models on
the 58 gold studies they never trained on (scripts/make_kaggle_sim.py builds the folder).
"""

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    for name in ("notebook", "bundle", "code", "data", "out"):
        p.add_argument(name)
    p.add_argument("--arm", default=None)
    p.add_argument("--backbone", default=None)
    p.add_argument("--env", default="/tmp/heliosli_env")
    p.add_argument("--chunk", type=int, default=None)
    args = p.parse_args()
    nb = json.loads(Path(args.notebook).read_text())
    cell = next("".join(c["source"]) for c in nb["cells"] if "SOURCES = " in "".join(c["source"]))
    consts = {}
    for line in cell.split("\n"):
        for key in ("SOURCES", "SOURCE_SHA", "ARM"):
            if line.startswith(key + " = "):
                consts[key] = ast.literal_eval(line[len(key) + 3:])
    out, bundle, code_src = Path(args.out), Path(args.bundle), Path(args.code)
    src = out / "src"
    src.mkdir(parents=True, exist_ok=True)
    for name, text in consts["SOURCES"].items():
        (src / name).write_bytes(text.encode("utf-8"))
        assert hashlib.sha256((src / name).read_bytes()).hexdigest() == consts["SOURCE_SHA"][name], name
    env_dir = Path(args.env)
    if not env_dir.is_dir():
        wheels = sorted(str(w) for w in (code_src / "dependencies").rglob("*.whl"))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", str(env_dir), *wheels], check=True)
    manifest = json.loads((bundle / "manifest.json").read_text())
    arm = args.arm or consts["ARM"]
    backbone = args.backbone or manifest["backbone"]
    worker_args = ["--backbone", backbone, "--size", str(manifest["size"]), "--crop", str(manifest["crop"]),
                   "--eval-stride", str(manifest.get("eval_stride", 3)), "--views", manifest.get("views", "v1")]
    if manifest.get("state_spec"):
        worker_args += ["--state-spec", str(bundle / manifest["state_spec"])]
    if manifest.get("side_model"):
        worker_args += ["--side-model", str(bundle / manifest["side_model"]), "--side-threshold", str(manifest.get("side_threshold", 2.0)),
                        "--side-position-mm", str(manifest.get("side_position_mm", 20.0)), "--side-max-x", str(manifest.get("side_max_x", 0.0))]
    worker_args += ["--slices", str(manifest.get("slices", 3)), "--phys-mm", str(manifest.get("phys_mm", 0.0))]
    if manifest.get("views") in ("v3k", "v4"):
        worker_args += ["--anchor-weights", str(code_src / "anatomy.pt")]
    if args.chunk:
        worker_args += ["--chunk", str(args.chunk)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(env_dir), str(code_src / "cloud_code")]), PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1")
    t0 = time.time()
    rc = subprocess.run([sys.executable, "-u", str(src / "probe_dual.py"), "--data", args.data, "--split", "test",
                         "--code", str(code_src / "cloud_code"), "--bundle", str(bundle), "--arm", arm, "--output", str(out / "pred"),
                         "--amp", "auto", "--loader-workers", "2", "--worker", "compact_infer.py",
                         "--worker-args", json.dumps(worker_args)], env=env).returncode
    print(f"rc {rc}, {time.time() - t0:.0f} s, arm {arm}, backbone {backbone}", flush=True)


if __name__ == "__main__":
    main()
