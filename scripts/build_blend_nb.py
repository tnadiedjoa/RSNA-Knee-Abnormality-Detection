"""Build the blend notebook (kaggle/blend/): our ensemble + the public 0.943 pipeline, rank-averaged.

    python -m scripts.build_blend_nb <public.ipynb> --weight 0.15 [--models "['*e13*', ...]"]

1. our models from the private knee-models dataset run first, in a separate process (GPU and RAM
   are free again afterwards, our DICOM decoders stay out of the public pipeline's environment);
2. the public notebook's cells, verbatim (yamadan96/rsna-knee-d4-public0946, from
   ryokucha/rsna-knee-d4-blend-0946-ours10; Apache 2.0), write its submission.csv;
3. per-label rank average: (1 - w) * public + w * ours. If our part failed, the public
   submission is kept as is.
"""

import argparse
import json
from pathlib import Path

from knee.config import ROOT_DIR

OURS = '''# Blend of our ensemble with the public 0.943 pipeline (cells below the next one: verbatim copy of
# yamadan96/rsna-knee-d4-public0946, itself from ryokucha/rsna-knee-d4-blend-0946-ours10; credits there).
# Step 1: our models, in a separate process, before the public pipeline.
import glob, os, subprocess, sys, time
T0_BLEND = time.time()
W_OURS = {weight}  # rank weight of our ensemble in the final blend (0 = public pipeline alone)
OUR_MODELS = {models}
OUR_BUDGET_S = {budget_h} * 3600
def _find(pattern):
    return next(p for depth in range(4) for p in glob.glob('/kaggle/input/' + '*/' * depth + pattern))
if W_OURS > 0:
    KNEE = os.path.dirname(os.path.dirname(_find('knee/infer.py')))
    DATA = os.path.dirname(_find('test_series.csv'))
    print(KNEE, DATA, sys.version)
    pip = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--target', '/tmp/knee_libs',
                          '--find-links', f'{{KNEE}}/wheels', 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg', 'python-gdcm'],
                         capture_output=True, text=True)
    print('decoders:', 'ok' if pip.returncode == 0 else pip.stderr[-1000:])
    script = f"""
import glob, shutil
from knee.infer import run
if __name__ == '__main__':
    models = list(dict.fromkeys(m for pat in {{OUR_MODELS!r}} for m in sorted(glob.glob('{{KNEE}}/models/' + pat))))
    print(models, flush=True)
    run('{{DATA}}', models, out_csv='/kaggle/working/sub_ours.csv', cache_dir='/tmp/knee_cache', workers=4,
        time_budget_s={{OUR_BUDGET_S}}, windows=24)
    shutil.rmtree('/tmp/knee_cache', ignore_errors=True)
"""
    open('/tmp/knee_ours.py', 'w').write(script)
    r = subprocess.run([sys.executable, '/tmp/knee_ours.py'], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=f'/tmp/knee_libs:{{KNEE}}'))
    print(r.stdout[-6000:]); print(r.stderr[-3000:])
    print('our ensemble:', 'ok' if r.returncode == 0 else f'FAILED ({{r.returncode}})', f'{{(time.time() - T0_BLEND) / 60:.1f}} min')
'''

BLEND = '''# Step 3: per-label rank average of the public submission and ours (public alone if ours is missing).
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
print(f'total {(time.time() - T0_BLEND) / 60:.1f} min')
'''


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("public_nb")
    p.add_argument("--weight", type=float, default=0.15)
    p.add_argument("--models", default="['*e13*', '*e16*', '*e18*', '*e11*', '*e07*', '*e12*', '*e09*']")
    p.add_argument("--budget-h", type=float, default=4.5)
    p.add_argument("--out", default=str(ROOT_DIR / "kaggle" / "blend" / "knee-blend.ipynb"))
    args = p.parse_args()
    nb = json.loads(Path(args.public_nb).read_text())
    for c in nb["cells"]:
        c.pop("id", None)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    nb["cells"] = ([code(OURS.format(weight=args.weight, models=args.models, budget_h=args.budget_h))]
                   + nb["cells"] + [code(BLEND)])
    Path(args.out).write_text(json.dumps(nb, indent=1))
    print(args.out, len(nb["cells"]), "cells, weight", args.weight)


if __name__ == "__main__":
    main()
