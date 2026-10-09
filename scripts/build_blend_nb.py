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
   With --reader-share s (0.950 base only, from blend 14): that base fuses per finding a CoAtNet and a 2.5D ConvNeXt
   reader, (1 - v) * CoAtNet + v * reader on ranks with hand-set v (Baker's 0.35 ... Synovitis 0.03), and leaves both
   arms in /kaggle/working; the fusion is redone with ours inside the reader's share,
   (1 - k v) * CoAtNet + k v * ((1 - s) * reader + s * ours) (k = --reader-scale), after checking that s = 0 gives
   the public submission back; uniform blend at --weight if the arms are missing.
   With --extra-arm-nb (0.950 base, from blend 17): a second public CoAtNet script (nartaa's 0.945 224 px model) runs
   in its own process after the public cells, its output path redirected to /kaggle/working/_r224.csv, and joins the
   base's CoAtNet arm at --extra-w inside the same per-finding fusion (with or without ours).
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
READER_SHARE = {reader_share}  # None: uniform blend at W_OURS; else ours' share of the public 2.5D reader's per-finding weight
READER_SCALE = {reader_scale}  # the readers' per-finding weights times this (with READER_SHARE)
EXTRA_ARM = {extra_arm!r}  # extra CoAtNet predictions averaged into the base's CoAtNet arm (None: no extra arm)
EXTRA_W = {extra_w}  # weight of EXTRA_ARM inside the CoAtNet arm
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

EXTRA_RUN = '''# Step 2b: the extra public arm (script written by the cell above), in its own process after the public pipeline.
if EXTRA_ARM:
    _t = time.time()
    try:
        _p = subprocess.run([sys.executable, '/kaggle/working/extra_arm.py'], capture_output=True, text=True,
                            timeout=max(600, END_H * 3600 - (time.time() - T0_BLEND) - 3 * 3600))
        print(_p.stdout[-2500:], _p.stderr[-1500:])
        print(f'extra arm: return code {_p.returncode}, {(time.time() - _t) / 60:.1f} min,',
              'written' if os.path.exists(EXTRA_ARM) else 'MISSING')
    except Exception as e:
        print('extra arm failed:', repr(e))
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

BLEND = '''# Step 4: the final submission. 0.950 base (READER_SHARE or EXTRA_ARM set): its per-finding fusion redone from its two
# arms (checked to give the public submission back), the CoAtNet arm averaged with EXTRA_ARM at EXTRA_W and the reader's
# share shared with ours at READER_SHARE (readers x READER_SCALE); otherwise per-label rank average with ours at W_OURS.
# Whatever is missing drops out; nothing usable left: the public submission alone.
import pandas as pd
_pub = pd.read_csv('/kaggle/working/submission.csv', dtype={'StudyInstanceUID': str})
_pub.to_csv('/kaggle/working/sub_public.csv', index=False)
_labels = [c for c in _pub.columns if c != 'StudyInstanceUID']
_rk = lambda x: x.rank(pct=True).to_numpy()
def _read(f):
    df = pd.read_csv(f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(_pub['StudyInstanceUID'])[_labels]
    assert df.notna().all().all(), f'{f} misses studies'
    return df
_ours = _arms = _extra = None
if W_OURS > 0:
    try:
        _ours = _read('/kaggle/working/sub_ours.csv')
    except Exception as e:
        print('ours missing:', repr(e))
if READER_SHARE is not None or EXTRA_ARM:
    try:  # CoAtNet and reader arms of the 0.950 base, and its per-finding reader weights
        _arms = [_read('/kaggle/working/_sota_0949.csv'), _read('/kaggle/working/_own.csv')]
        _v = {c: TARGET_WEIGHTS.get(c, 0.15) for c in _labels}
        _gap = max(abs(pd.Series((1 - _v[c]) * _rk(_arms[0][c]) + _v[c] * _rk(_arms[1][c])).rank(pct=True).to_numpy()
                       - _pub[c].to_numpy()).max() for c in _labels)
        assert _gap < 1e-6, f'public fusion not reproduced (max gap {_gap})'
        print(f'public fusion reproduced (max gap {_gap:.1e}); reader weights {_v}')
    except Exception as e:
        print('per-finding fusion impossible:', repr(e))
        _arms = None
if _arms is not None and EXTRA_ARM:
    try:
        _extra = _read(EXTRA_ARM)
    except Exception as e:
        print('extra arm missing:', repr(e))
_s = READER_SHARE if (_arms is not None and _ours is not None and READER_SHARE) else 0.0
_k = READER_SCALE if _s else 1.0
try:
    _out = _pub.copy()
    if _arms is not None and (_s or _extra is not None):
        for c in _labels:
            v = min(_k * _v[c], 0.9)
            co = _rk(_arms[0][c]) if _extra is None else (1 - EXTRA_W) * _rk(_arms[0][c]) + EXTRA_W * _rk(_extra[c])
            rd = (1 - _s) * _rk(_arms[1][c]) + (_s * _rk(_ours[c]) if _s else 0)
            _out[c] = pd.Series((1 - v) * co + v * rd).rank(pct=True).to_numpy()
        print(f'BLEND written: per finding, extra arm {EXTRA_W if _extra is not None else 0} of the CoAtNet arm,',
              f'ours {_s} of the reader share, readers x{_k};', {c: round(min(_k * _v[c], 0.9) * _s, 3) for c in _labels}, _out.shape)
    elif _ours is not None:
        for c in _labels:
            _out[c] = (1 - W_OURS) * _rk(_pub[c]) + W_OURS * _rk(_ours[c])
        print(f'BLEND written: public {1 - W_OURS:.2f} + ours {W_OURS:.2f}', _out.shape)
    else:
        print('public submission kept alone')
    _out.to_csv('/kaggle/working/submission.csv', index=False)
except Exception as e:
    _pub.to_csv('/kaggle/working/submission.csv', index=False)
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
    p.add_argument("--reader-share", type=float, default=None,
                   help="0.950 base: ours' share of its reader's per-finding weight (default: uniform blend at --weight)")
    p.add_argument("--reader-scale", type=float, default=1.0)
    p.add_argument("--extra-arm-nb", default=None, help="public notebook whose last code cell is the extra CoAtNet script")
    p.add_argument("--extra-w", type=float, default=0.33)
    p.add_argument("--extra-source", default="nartaa/rsna-knee-0945-efficient-224crop")
    p.add_argument("--out", default=str(ROOT_DIR / "kaggle" / "blend" / "knee-blend.ipynb"))
    p.add_argument("--source", default="yamadan96/rsna-knee-d4-public0946, itself from ryokucha/rsna-knee-d4-blend-0946-ours10",
                   help="public notebook the cells come from, credited in the setup cell")
    args = p.parse_args()
    nb = json.loads(Path(args.public_nb).read_text())
    for c in nb["cells"]:
        c.pop("id", None)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    fmt = dict(source=args.source, weight=args.weight, models=args.models, windows=args.windows, end_h=args.end_h, min_left_h=args.min_left_h,
               reader_share=args.reader_share, reader_scale=args.reader_scale,
               extra_arm='/kaggle/working/_r224.csv' if args.extra_arm_nb else None, extra_w=args.extra_w)
    extra = []
    if args.extra_arm_nb:  # its script verbatim, written to a file; only the output path changes
        src = [c for c in json.loads(Path(args.extra_arm_nb).read_text())["cells"] if c["cell_type"] == "code"][-1]
        src = "".join(src["source"])
        assert src.count('"/kaggle/working/submission.csv"') == 1, "output path of the extra arm not found once"
        src = src.replace('"/kaggle/working/submission.csv"', '"/kaggle/working/_r224.csv"')
        extra = [code(f"%%writefile /kaggle/working/extra_arm.py\n# Verbatim copy of {args.extra_source} (credits there); "
                      f"only its output path is changed.\n" + src), code(EXTRA_RUN)]
    nb["cells"] = [code(SETUP.format(**fmt))] + nb["cells"] + extra + [code(OURS.format(**fmt)), code(BLEND)]
    Path(args.out).write_text(json.dumps(nb, indent=1))
    print(args.out, len(nb["cells"]), "cells, weight", args.weight, "reader share", args.reader_share, "scale", args.reader_scale,
          "extra arm", bool(args.extra_arm_nb), args.extra_w)


if __name__ == "__main__":
    main()
