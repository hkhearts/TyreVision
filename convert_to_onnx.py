"""
Convert Wheelyze PyTorch models to single-file ONNX for browser inference.
Run: python convert_to_onnx.py
Output: models/ directory with .onnx files (each ~40MB, all data inline)
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add Wheelyze directory to path
WHEELYZE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Wheelyze-Tyre-Health-Detection')
sys.path.insert(0, WHEELYZE_DIR)
os.chdir(WHEELYZE_DIR)

import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cpu")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  TyreVision - PyTorch to ONNX Converter")
print("=" * 60)

dummy = torch.randn(1, 3, 300, 300)

def export_model_inline(model, out_path, model_name):
    """Export model to a single self-contained .onnx file."""
    model.eval()
    # Use legacy TorchScript-based exporter (dynamo=False) which
    # produces a single self-contained file without external .data
    torch.onnx.export(
        model,
        dummy,
        out_path,
        dynamo=False,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
    )
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"    OK: {model_name} saved ({size_mb:.1f}MB)")


# Binary model (Good vs Defective)
print("\n[1/4] Converting binary model...")
binary_model = models.efficientnet_b3(weights=None)
binary_model.classifier[1] = nn.Linear(binary_model.classifier[1].in_features, 2)
binary_model.load_state_dict(torch.load("binary_best.pth", map_location=device))
export_model_inline(binary_model, os.path.join(OUT_DIR, "binary_model.onnx"), "binary_model")

# Defective model (grades 1-5)
print("[2/4] Converting defective grades model...")
defective_model = models.efficientnet_b3(weights=None)
defective_model.classifier[1] = nn.Linear(defective_model.classifier[1].in_features, 5)
defective_model.load_state_dict(torch.load("defectiveFinal.pth", map_location=device))
export_model_inline(defective_model, os.path.join(OUT_DIR, "defective_model.onnx"), "defective_model")

# Good model (grades 6-10)
print("[3/4] Converting good grades model...")
good_model = models.efficientnet_b3(weights=None)
good_model.classifier[1] = nn.Linear(good_model.classifier[1].in_features, 5)
good_model.load_state_dict(torch.load("good_best.pth", map_location=device))
export_model_inline(good_model, os.path.join(OUT_DIR, "good_model.onnx"), "good_model")

# ResNet18 (full 1-10)
print("[4/4] Converting ResNet18 model...")
resnet = models.resnet18(weights=None)
resnet.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(resnet.fc.in_features, 10)
)
resnet.load_state_dict(torch.load("tyre_resnet18_best.pth", map_location=device))
export_model_inline(resnet, os.path.join(OUT_DIR, "resnet_model.onnx"), "resnet_model")

# Clean up the external data files from the previous run (if any)
for f in os.listdir(OUT_DIR):
    if f.endswith('.onnx.data'):
        os.remove(os.path.join(OUT_DIR, f))
        print(f"    Cleaned up: {f}")

print("\n" + "=" * 60)
print("  All 4 models converted to single-file ONNX!")
total_mb = sum(
    os.path.getsize(os.path.join(OUT_DIR, f)) / 1024 / 1024
    for f in os.listdir(OUT_DIR) if f.endswith('.onnx')
)
print(f"  Total size: {total_mb:.1f}MB")
print(f"  Output directory: {OUT_DIR}")
print("=" * 60)
