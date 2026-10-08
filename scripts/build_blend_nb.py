"""Build the blend notebook (kaggle/blend/): the public 0.943 pipeline + our ensemble, rank-averaged.

    python -m scripts.build_blend_nb <public.ipynb> --weight 0.3 [--models "['*e24*', ...]"] [--windows 12]

1. setup only: our DICOM decoders are installed with --target (the public environment stays untouched)
   and the clock of the whole run starts;
2. the public notebook's cells, verbatim (credited through --source: yamadan96/rsna-knee-d4-public0946, the 0.943
   stack, until blend 8; matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed, the 0.950 base, from blend 9),
   write its submission.csv. They run first
   because their own time budget counts from their own start (version 1 ran ours first: the public
   part then believed it had 8 h left and the run hit Kaggle's 9 h limit);
3. our models, in a separate process on every GPU with room left (grouped inference, both T4s since
   2026-10-06; before, the one with the most free memory), with whatever time is left before END_H:
   their own budget skips the last models, and a hard timeout kills the process group if the first
   ones alone would overrun. Too little time left: skipped;
4. per-label rank average (1 - w) * public + w * ours, or the public submission alone if ours is missing.
"""

import argparse
import json
from pathlib import Path

from knee.config import ROOT_DIR

SETUP = '''# Blend of a public pipeline with our ensemble (cells below: verbatim copy of {source}; credits there).
# Step 1: setup only. The public pipeline runs first, our models after it with the time left.
import glob, os, subprocess, sys, time
T0_BLEND = time.time()
W_OURS = {weight}  # rank weight of our ensemble in the final blend (0 = public pipeline alone)
OUR_MODELS = {models}
OUR_WINDOWS = {windows}
END_H = {end_h}  # our process must be done by then (Kaggle stops the notebook at 9 h)
MIN_LEFT_H = {min_left_h}  # below this, our models are skipped and the public submission goes alone
def _find(pattern):
    return next(p for depth in range(4) for p in glob.glob('/kaggle/input/' + '*/' * depth + pattern))
KNEE = os.path.dirname(os.path.dirname(_find('knee/infer.py')))
DATA = os.path.dirname(_find('test_series.csv'))
pip = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--target', '/tmp/knee_libs',
                      '--find-links', f'{{KNEE}}/wheels', 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg', 'python-gdcm'],
                     capture_output=True, text=True)
print(KNEE, DATA, sys.version, 'decoders:', 'ok' if pip.returncode == 0 else pip.stderr[-1000:])
'''

OURS = '''# Step 3: our models, after the public pipeline, in a separate process with the time left.
import gc, signal
_left = END_H * 3600 - (time.time() - T0_BLEND)
print(f'public pipeline done at {{(time.time() - T0_BLEND) / 3600:.2f}} h, {{_left / 3600:.2f}} h left for ours')
if W_OURS > 0 and _left > MIN_LEFT_H * 3600:
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception as e:
        print('empty_cache:', repr(e))
    _free = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True).stdout.split()
    _gpu = ','.join(str(i) for i, f in enumerate(_free) if int(f) >= 6000) or (
        str(max(range(len(_free)), key=lambda i: int(_free[i]))) if _free else '0')
    print('GPU free MiB:', _free, '-> ours on GPU(s)', _gpu)
    open('/tmp/knee_ours.py', 'w').write(f"""
import glob, shutil
from knee.infer import run
if __name__ == '__main__':
    models = list(dict.fromkeys(m for pat in {{OUR_MODELS!r}} for m in sorted(glob.glob('{{KNEE}}/models/' + pat))))
    print(models, flush=True)
    run('{{DATA}}', models, out_csv='/kaggle/working/sub_ours.csv', cache_dir='/tmp/knee_cache', workers=4,
        time_budget_s={{_left - 20 * 60:.0f}}, windows={{OUR_WINDOWS}}, grouped=True)
    shutil.rmtree('/tmp/knee_cache', ignore_errors=True)
""")
    _p = subprocess.Popen([sys.executable, '/tmp/knee_ours.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          start_new_session=True, env=dict(os.environ, PYTHONPATH=f'/tmp/knee_libs:{{KNEE}}', CUDA_VISIBLE_DEVICES=_gpu))
    try:
        _out, _ = _p.communicate(timeout=_left - 10 * 60)
        print(_out[-6000:]); print('our ensemble:', 'ok' if _p.returncode == 0 else f'FAILED ({{_p.returncode}})')
    except subprocess.TimeoutExpired:
        os.killpg(_p.pid, signal.SIGKILL); _p.communicate()
        if os.path.exists('/kaggle/working/sub_ours.csv'): os.remove('/kaggle/working/sub_ours.csv')
        print('our ensemble: TIMEOUT, killed')
else:
    print('our ensemble skipped')
print(f'{{(time.time() - T0_BLEND) / 3600:.2f}} h')
'''

BLEND = '''# Step 4: per-label rank average of the public submission and ours (public alone if ours is missing).
import pandas as pd
_pub = pd.read_csv('/kaggle/working/submission.csv', dtype={'StudyInstanceUID': str})
_pub.to_csv('/kaggle/working/sub_public.csv', index=False)
try:
    assert W_OURS > 0, 'weight 0'
    _labels = [c for c in _pub.columns if c != 'StudyInstanceUID']
    _ours = pd.read_csv('/kaggle/working/sub_ours.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    _ours = _ours.reindex(_pub['StudyInstanceUID'])[_labels]
    assert _ours.notna().all().all(), 'our predictions miss studies'
    _out = _pub.copy()
    for c in _labels:
        _out[c] = (1 - W_OURS) * _pub[c].rank(pct=True).to_numpy() + W_OURS * _ours[c].rank(pct=True).to_numpy()
    _out.to_csv('/kaggle/working/submission.csv', index=False)
    print(f'BLEND written: public {1 - W_OURS:.2f} + ours {W_OURS:.2f}', _out.shape)
except Exception as e:
    print('public submission kept alone:', repr(e))
print(f'total {(time.time() - T0_BLEND) / 3600:.2f} h')
'''


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("public_nb")
    p.add_argument("--weight", type=float, default=0.3)
    p.add_argument("--models", default="['*e24*', '*e23*', '*e25*']")
    p.add_argument("--windows", type=int, default=12)
    p.add_argument("--end-h", type=float, default=8.6)
    p.add_argument("--min-left-h", type=float, default=0.75)
    p.add_argument("--out", default=str(ROOT_DIR / "kaggle" / "blend" / "knee-blend.ipynb"))
    p.add_argument("--source", default="yamadan96/rsna-knee-d4-public0946, itself from ryokucha/rsna-knee-d4-blend-0946-ours10",
                   help="public notebook the cells come from, credited in the setup cell")
    args = p.parse_args()
    nb = json.loads(Path(args.public_nb).read_text())
    for c in nb["cells"]:
        c.pop("id", None)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    fmt = dict(source=args.source, weight=args.weight, models=args.models, windows=args.windows, end_h=args.end_h, min_left_h=args.min_left_h)
    nb["cells"] = [code(SETUP.format(**fmt))] + nb["cells"] + [code(OURS.format(**fmt)), code(BLEND)]
    Path(args.out).write_text(json.dumps(nb, indent=1))
    print(args.out, len(nb["cells"]), "cells, weight", args.weight)


if __name__ == "__main__":
    main()
