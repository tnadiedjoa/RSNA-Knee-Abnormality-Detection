import sys, subprocess, importlib, os, glob, time
print("python", sys.version)
for m in ["torch", "torchvision", "timm", "pydicom", "pylibjpeg", "libjpeg", "openjpeg", "gdcm", "cv2", "numpy", "pandas", "sklearn"]:
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(m, "MISSING", repr(e)[:80])
import torch
print("cuda", torch.cuda.is_available(), torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("cpus", os.cpu_count())
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; free -g; df -h /kaggle/working /tmp", shell=True, capture_output=True, text=True).stdout)
root = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
print(os.listdir(root) if os.path.exists(root) else os.listdir("/kaggle/input"))
import pydicom
t = time.time(); n = 0; ts = {}
for f in glob.glob(root + "/test_series/*/*/*.dcm"):
    ds = pydicom.dcmread(f); a = ds.pixel_array; n += 1
    ts[str(ds.file_meta.TransferSyntaxUID)] = ts.get(str(ds.file_meta.TransferSyntaxUID), 0) + 1
print("decoded", n, "test files in %.1fs" % (time.time() - t), ts)
