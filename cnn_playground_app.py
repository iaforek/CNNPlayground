import io
import json
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


# ============================================================
# Config
# ============================================================
ROOT = Path("./data")
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DRAWING_SAVE_DIR = Path("./saved_drawings")
DRAWING_SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


@dataclass
class DatasetInfo:
    name: str
    num_classes: int
    input_channels: int
    image_size: int
    class_names: List[str]
    mean: Tuple[float, ...]
    std: Tuple[float, ...]


DATASET_REGISTRY: Dict[str, DatasetInfo] = {
    "MNIST": DatasetInfo(
        name="MNIST",
        num_classes=10,
        input_channels=1,
        image_size=28,
        class_names=[str(i) for i in range(10)],
        mean=(0.1307,),
        std=(0.3081,),
    ),
    "FashionMNIST": DatasetInfo(
        name="FashionMNIST",
        num_classes=10,
        input_channels=1,
        image_size=28,
        class_names=[
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
        mean=(0.2860,),
        std=(0.3530,),
    ),
}


PRESET_MODELS: Dict[str, List[Dict[str, Any]]] = {
    "TinyCNN": [
        {
            "type": "conv",
            "out_channels": 8,
            "kernel_size": 3,
            "padding": "same",
            "activation": "relu",
            "batch_norm": True,
            "pool": "max",
            "pool_kernel_size": 2,
        },
        {
            "type": "conv",
            "out_channels": 16,
            "kernel_size": 3,
            "padding": "same",
            "activation": "relu",
            "batch_norm": True,
            "pool": "max",
            "pool_kernel_size": 2,
        },
        {"type": "flatten"},
        {"type": "linear", "out_features": 32, "activation": "relu", "dropout": 0.15},
        {"type": "linear", "out_features": "num_classes"},
    ],
    "LeNetLike": [
        {
            "type": "conv",
            "out_channels": 6,
            "kernel_size": 5,
            "padding": 0,
            "activation": "tanh",
            "pool": "avg",
            "pool_kernel_size": 2,
        },
        {
            "type": "conv",
            "out_channels": 16,
            "kernel_size": 5,
            "padding": 0,
            "activation": "tanh",
            "pool": "avg",
            "pool_kernel_size": 2,
        },
        {"type": "flatten"},
        {"type": "linear", "out_features": 120, "activation": "tanh"},
        {"type": "linear", "out_features": 84, "activation": "tanh"},
        {"type": "linear", "out_features": "num_classes"},
    ],
    "MediumCNN": [
        {
            "type": "conv",
            "out_channels": 16,
            "kernel_size": 3,
            "padding": "same",
            "activation": "relu",
            "batch_norm": True,
        },
        {
            "type": "conv",
            "out_channels": 32,
            "kernel_size": 3,
            "padding": "same",
            "activation": "relu",
            "batch_norm": True,
            "pool": "max",
            "pool_kernel_size": 2,
            "dropout": 0.1,
        },
        {
            "type": "conv",
            "out_channels": 64,
            "kernel_size": 3,
            "padding": "same",
            "activation": "relu",
            "batch_norm": True,
            "pool": "max",
            "pool_kernel_size": 2,
            "dropout": 0.15,
        },
        {"type": "flatten"},
        {"type": "linear", "out_features": 128, "activation": "relu", "dropout": 0.25},
        {"type": "linear", "out_features": "num_classes"},
    ],
    "MLPBaseline": [
        {"type": "flatten"},
        {"type": "linear", "out_features": 256, "activation": "relu", "dropout": 0.2},
        {"type": "linear", "out_features": 128, "activation": "relu", "dropout": 0.2},
        {"type": "linear", "out_features": "num_classes"},
    ],
}


def pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2)


DEFAULT_SPEC_TEXT = pretty_json(PRESET_MODELS["TinyCNN"])


# ============================================================
# Dataset helpers
# ============================================================

def build_transforms(info: DatasetInfo) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(info.mean, info.std),
    ])



def load_dataset(name: str, train: bool):
    info = DATASET_REGISTRY[name]
    tx = build_transforms(info)
    if name == "MNIST":
        return datasets.MNIST(ROOT, train=train, download=True, transform=tx)
    if name == "FashionMNIST":
        return datasets.FashionMNIST(ROOT, train=train, download=True, transform=tx)
    raise ValueError(f"Unsupported dataset: {name}")



def make_dataloaders(
    dataset_name: str,
    batch_size: int,
    val_ratio: float,
    max_train_samples: int,
    max_test_samples: int,
) -> Tuple[DatasetInfo, DataLoader, DataLoader, DataLoader]:
    info = DATASET_REGISTRY[dataset_name]
    full_train = load_dataset(dataset_name, train=True)
    test_ds = load_dataset(dataset_name, train=False)

    if 0 < max_train_samples < len(full_train):
        indices = np.random.permutation(len(full_train))[:max_train_samples]
        full_train = Subset(full_train, indices.tolist())

    if 0 < max_test_samples < len(test_ds):
        indices = np.random.permutation(len(test_ds))[:max_test_samples]
        test_ds = Subset(test_ds, indices.tolist())

    val_size = max(1, int(len(full_train) * val_ratio))
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    common_loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader_args)
    return info, train_loader, val_loader, test_loader


# ============================================================
# Dynamic model builder
# ============================================================

def activation_from_name(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "selu":
        return nn.SELU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class DynamicCNN(nn.Module):
    def __init__(self, spec: List[Dict[str, Any]], input_channels: int, num_classes: int):
        super().__init__()
        self.spec = deepcopy(spec)
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        current_channels = input_channels
        flattened_seen = False

        for idx, layer_cfg in enumerate(spec):
            layer_type = str(layer_cfg["type"]).lower()

            if layer_type == "conv":
                out_channels = int(layer_cfg["out_channels"])
                kernel_size = int(layer_cfg.get("kernel_size", 3))
                stride = int(layer_cfg.get("stride", 1))
                padding = layer_cfg.get("padding", "same")
                if padding == "same":
                    padding = kernel_size // 2
                padding = int(padding)

                block = []
                block.append(nn.Conv2d(current_channels, out_channels, kernel_size, stride=stride, padding=padding))
                if layer_cfg.get("batch_norm", False):
                    block.append(nn.BatchNorm2d(out_channels))
                activation = layer_cfg.get("activation")
                if activation:
                    block.append(activation_from_name(activation))
                pool = layer_cfg.get("pool")
                if pool:
                    pool_kernel = int(layer_cfg.get("pool_kernel_size", 2))
                    if str(pool).lower() == "max":
                        block.append(nn.MaxPool2d(pool_kernel))
                    elif str(pool).lower() == "avg":
                        block.append(nn.AvgPool2d(pool_kernel))
                    else:
                        raise ValueError(f"Unsupported pool type: {pool}")
                dropout = float(layer_cfg.get("dropout", 0.0))
                if dropout > 0:
                    block.append(nn.Dropout2d(dropout))
                self.layers.append(nn.Sequential(*block))
                current_channels = out_channels

            elif layer_type == "maxpool":
                self.layers.append(nn.MaxPool2d(int(layer_cfg.get("kernel_size", 2))))

            elif layer_type == "avgpool":
                self.layers.append(nn.AvgPool2d(int(layer_cfg.get("kernel_size", 2))))

            elif layer_type == "flatten":
                self.layers.append(Flatten())
                flattened_seen = True

            elif layer_type == "linear":
                out_features = layer_cfg.get("out_features")
                if out_features == "num_classes":
                    out_features = num_classes
                out_features = int(out_features)
                block = []
                block.append(nn.LazyLinear(out_features))
                if layer_cfg.get("batch_norm", False):
                    block.append(nn.LazyBatchNorm1d())
                activation = layer_cfg.get("activation")
                if activation:
                    block.append(activation_from_name(activation))
                dropout = float(layer_cfg.get("dropout", 0.0))
                if dropout > 0:
                    block.append(nn.Dropout(dropout))
                self.layers.append(nn.Sequential(*block))
                flattened_seen = True

            elif layer_type == "dropout":
                p = float(layer_cfg.get("p", 0.2))
                self.layers.append(nn.Dropout(p) if flattened_seen else nn.Dropout2d(p))

            elif layer_type == "activation":
                self.layers.append(activation_from_name(layer_cfg["name"]))

            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def named_submodules(self) -> List[Tuple[str, nn.Module]]:
        rows = []
        for i, layer in enumerate(self.layers):
            rows.append((f"layer_{i}", layer))
        return rows


# ============================================================
# Training, evaluation, summary
# ============================================================

def parse_spec(spec_text: str) -> List[Dict[str, Any]]:
    spec = json.loads(spec_text)
    if not isinstance(spec, list) or len(spec) == 0:
        raise ValueError("Architecture spec must be a non-empty JSON list.")
    return spec



def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total += xb.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(yb.cpu())

    return {
        "loss": total_loss / max(1, total),
        "acc": total_correct / max(1, total),
        "preds": torch.cat(all_preds) if all_preds else torch.tensor([]),
        "targets": torch.cat(all_targets) if all_targets else torch.tensor([]),
    }



def build_model_summary(model: nn.Module, input_shape: Tuple[int, int, int]) -> Tuple[str, List[str]]:
    rows = []
    hooks = []
    layer_names = []

    def register(name: str, module: nn.Module):
        def hook(_module, _inputs, output):
            shape = tuple(output.shape) if hasattr(output, "shape") else "?"
            params = sum(p.numel() for p in _module.parameters() if p.requires_grad)
            rows.append((name, _module.__class__.__name__, shape, params))
        hooks.append(module.register_forward_hook(hook))

    for name, module in model.named_submodules():
        register(name, module)
        layer_names.append(name)

    dummy = torch.zeros(1, *input_shape, device=DEVICE)
    model.eval()
    with torch.no_grad():
        _ = model(dummy)

    for h in hooks:
        h.remove()

    md = ["| Layer | Type | Output shape | Trainable params |", "|---|---|---:|---:|"]
    for name, cls_name, shape, params in rows:
        md.append(f"| {name} | {cls_name} | `{shape}` | {params:,} |")
    return "\n".join(md), layer_names



def benchmark_inference(model: nn.Module, sample_shape: Tuple[int, int, int], runs: int = 100) -> float:
    model.eval()
    x = torch.randn(1, *sample_shape, device=DEVICE)
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0



def make_history_plot(history: Dict[str, List[float]]) -> Image.Image:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = list(range(1, len(history["train_loss"]) + 1))

    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].plot(epochs, history["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train acc")
    axes[1].plot(epochs, history["val_acc"], label="val acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    img = fig_to_pil(fig)
    plt.close(fig)
    return img



def make_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, class_names: List[str]) -> Image.Image:
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(targets.numpy(), preds.numpy()):
        cm[int(t), int(p)] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    img = fig_to_pil(fig)
    plt.close(fig)
    return img



def fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).convert("RGB")



def export_checkpoint(state: Dict[str, Any], filename_prefix: str = "model") -> str:
    if not state or "model" not in state:
        raise ValueError("No trained model found in state.")
    model = state["model"]
    path = CHECKPOINT_DIR / f"{filename_prefix}_{int(time.time())}.pt"
    payload = {
        "model_state": model.state_dict(),
        "spec": state["spec"],
        "dataset_name": state["dataset_name"],
        "input_channels": state["input_channels"],
        "num_classes": state["num_classes"],
        "class_names": state["class_names"],
        "history": state["history"],
    }
    torch.save(payload, path)
    return str(path.resolve())



def train_experiment(
    dataset_name: str,
    preset_name: str,
    spec_text: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    weight_decay: float,
    val_ratio: float,
    max_train_samples: int,
    max_test_samples: int,
):
    try:
        info, train_loader, val_loader, test_loader = make_dataloaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
            val_ratio=val_ratio,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
        )
        spec = parse_spec(spec_text)
        model = DynamicCNN(spec, input_channels=info.input_channels, num_classes=info.num_classes).to(DEVICE)

        # Initialize lazy modules.
        with torch.no_grad():
            _ = model(torch.zeros(1, info.input_channels, info.image_size, info.image_size, device=DEVICE))

        criterion = nn.CrossEntropyLoss()
        if optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_state = None
        best_val_acc = -1.0
        train_start = time.perf_counter()

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            for xb, yb in train_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * xb.size(0)
                running_correct += (logits.argmax(dim=1) == yb).sum().item()
                running_total += xb.size(0)

            train_loss = running_loss / max(1, running_total)
            train_acc = running_correct / max(1, running_total)
            val_metrics = run_eval(model, val_loader, criterion)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["acc"])

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                best_state = deepcopy(model.state_dict())

        train_seconds = time.perf_counter() - train_start

        if best_state is not None:
            model.load_state_dict(best_state)

        test_metrics = run_eval(model, test_loader, criterion)
        param_count = count_parameters(model)
        summary_md, layer_names = build_model_summary(model, (info.input_channels, info.image_size, info.image_size))
        ms_per_sample = benchmark_inference(model, (info.input_channels, info.image_size, info.image_size), runs=100)
        history_img = make_history_plot(history)
        cm_img = make_confusion_matrix(test_metrics["preds"], test_metrics["targets"], info.class_names)

        metrics_md = f"""
### Experiment summary

- **Dataset:** {dataset_name}
- **Preset selected:** {preset_name}
- **Device:** `{DEVICE}`
- **Trainable parameters:** {param_count:,}
- **Approx. model size (fp32):** {param_count * 4 / (1024**2):.2f} MB
- **Best validation accuracy:** {best_val_acc:.4f}
- **Test accuracy:** {test_metrics['acc']:.4f}
- **Test loss:** {test_metrics['loss']:.4f}
- **Inference time:** {ms_per_sample:.3f} ms / sample
- **Training time:** {train_seconds:.2f} s

### Notes

- The layer table below shows **where parameters live** and **how tensor shapes change**.
- The feature map tab lets you inspect what a chosen layer is responding to.
- To compare model simplifications, keep the dataset and training settings fixed, then change only the architecture JSON.
"""

        # Cache a small test pool for later visualization.
        cached_samples = []
        for xb, yb in test_loader:
            for i in range(xb.size(0)):
                cached_samples.append((xb[i].cpu(), int(yb[i].cpu().item())))
                if len(cached_samples) >= 128:
                    break
            if len(cached_samples) >= 128:
                break

        state = {
            "model": model,
            "dataset_name": dataset_name,
            "spec": spec,
            "input_channels": info.input_channels,
            "num_classes": info.num_classes,
            "image_size": info.image_size,
            "class_names": info.class_names,
            "mean": info.mean,
            "std": info.std,
            "history": history,
            "layer_names": layer_names,
            "cached_samples": cached_samples,
        }

        feature_dropdown_update = gr.update(choices=layer_names, value=layer_names[0] if layer_names else None)
        return state, history_img, cm_img, metrics_md + "\n\n" + summary_md, feature_dropdown_update, "Training finished successfully."

    except Exception as exc:
        return None, None, None, f"Training failed: {exc}", gr.update(choices=[]), f"Error: {exc}"


# ============================================================
# Drawing, prediction, saving, batch testing
# ============================================================

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def denormalize_tensor(x: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> torch.Tensor:
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    return x * std_t + mean_t



def editor_value_to_numpy(image: Any) -> np.ndarray:
    if image is None:
        raise ValueError("Please provide an image first.")

    if isinstance(image, dict):
        if image.get("composite") is not None:
            image = image["composite"]
        elif image.get("background") is not None:
            image = image["background"]
        elif image.get("layers"):
            image = image["layers"][-1]
        else:
            raise ValueError("Could not read the editor image.")

    if isinstance(image, (str, Path)):
        pil = Image.open(image)
    elif isinstance(image, Image.Image):
        pil = image
    else:
        pil = Image.fromarray(np.array(image).astype(np.uint8))

    pil = pil.convert("RGBA") if pil.mode not in ("L", "RGB", "RGBA") else pil
    if pil.mode == "RGBA":
        white_bg = Image.new("RGBA", pil.size, (255, 255, 255, 255))
        pil = Image.alpha_composite(white_bg, pil).convert("L")
    else:
        pil = pil.convert("L")

    return np.array(pil)



def preprocess_image_for_model(image: Any, image_size: int) -> Tuple[Image.Image, torch.Tensor]:
    arr = editor_value_to_numpy(image)
    pil = Image.fromarray(arr).convert("L")

    # User drawings and batch images are expected to be dark strokes on a light background.
    # MNIST-style models expect the foreground digit to be bright on dark.
    pil = ImageOps.invert(pil)

    arr = np.array(pil)
    ys, xs = np.where(arr > 20)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Image appears empty after preprocessing.")

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pil = pil.crop((x0, y0, x1 + 1, y1 + 1))

    side = max(pil.width, pil.height)
    canvas = Image.new("L", (side, side), color=0)
    canvas.paste(pil, ((side - pil.width) // 2, (side - pil.height) // 2))

    resized = canvas.resize((20, 20), Image.Resampling.LANCZOS)
    final_canvas = Image.new("L", (image_size, image_size), color=0)
    offset = ((image_size - 20) // 2, (image_size - 20) // 2)
    final_canvas.paste(resized, offset)

    tensor = transforms.ToTensor()(final_canvas)
    return final_canvas, tensor



def make_probability_plot(probs: np.ndarray, class_names: List[str]) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(range(len(probs)), probs)
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Prediction probabilities")
    ax.set_ylabel("Probability")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    img = fig_to_pil(fig)
    plt.close(fig)
    return img



def run_model_on_image(state: Dict[str, Any], image: Any) -> Tuple[Image.Image, np.ndarray, str, float]:
    processed_pil, tensor = preprocess_image_for_model(image, state["image_size"])
    x = tensor.unsqueeze(0)
    mean = torch.tensor(state["mean"]).view(1, -1, 1, 1)
    std = torch.tensor(state["std"]).view(1, -1, 1, 1)
    x = (x - mean) / std
    x = x.to(DEVICE)

    model = state["model"]
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    pred_label = state["class_names"][pred_idx]
    confidence = float(probs[pred_idx])
    return processed_pil, probs, pred_label, confidence



def predict_from_canvas(state: Dict[str, Any], image: Any):
    if not state or "model" not in state:
        return None, None, "Train a model first."

    try:
        processed_pil, probs, pred_label, confidence = run_model_on_image(state, image)
        prob_img = make_probability_plot(probs, state["class_names"])
        top_lines = [
            f"### Prediction\n",
            f"- **Predicted class:** {pred_label}",
            f"- **Confidence:** {confidence:.4f}",
            "\nTop scores:",
        ]
        top_ids = np.argsort(probs)[::-1][:5]
        for idx in top_ids:
            top_lines.append(f"- {state['class_names'][int(idx)]}: {probs[int(idx)]:.4f}")

        return processed_pil, prob_img, "\n".join(top_lines)

    except Exception as exc:
        return None, None, f"Prediction failed: {exc}"



def save_drawing_action(image: Any, filename: str) -> str:
    try:
        arr = editor_value_to_numpy(image)
        pil = Image.fromarray(arr).convert("L")

        filename = filename.strip() if filename else ""
        if not filename:
            filename = f"drawing_{int(time.time())}.png"

        save_path = DRAWING_SAVE_DIR / filename
        if save_path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            save_path = save_path.with_suffix(".png")

        pil.save(save_path)
        return f"Saved drawing to: {save_path.resolve()}"
    except Exception as exc:
        return f"Save failed: {exc}"



def extract_expected_label(filename: str) -> str:
    stem = Path(filename).stem
    return stem.split("_")[0] if "_" in stem else stem



def load_batch_directory(directory_path: str):
    try:
        directory_path = (directory_path or "").strip()
        if not directory_path:
            return [], [], "Enter a directory path first."

        directory = Path(directory_path).expanduser()
        if not directory.exists() or not directory.is_dir():
            return [], [], f"Directory not found: {directory}"

        image_paths = sorted(
            [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS],
            key=lambda p: p.name.lower(),
        )
        if not image_paths:
            return [], [], "No supported image files were found in the directory."

        rows = [[p.name, extract_expected_label(p.name)] for p in image_paths]
        msg = f"Loaded **{len(image_paths)}** image(s) from `{directory.resolve()}`."
        return [str(p) for p in image_paths], rows, msg
    except Exception as exc:
        return [], [], f"Load failed: {exc}"



def run_batch_test(state: Dict[str, Any], batch_paths: List[str]):
    if not state or "model" not in state:
        return [], "Train a model first."
    if not batch_paths:
        return [], "Load a directory first."

    try:
        results = []
        scored_count = 0
        correct_count = 0

        for file_path in batch_paths:
            path = Path(file_path)
            expected_label = extract_expected_label(path.name)
            _processed_pil, _probs, pred_label, confidence = run_model_on_image(state, path)

            if expected_label in state["class_names"]:
                is_correct = pred_label == expected_label
                verdict = "correct" if is_correct else "not correct"
                scored_count += 1
                if is_correct:
                    correct_count += 1
            else:
                verdict = "expected label not in model classes"

            results.append([path.name, expected_label, pred_label, float(confidence), verdict])

        if scored_count > 0:
            accuracy = correct_count / scored_count
            summary = (
                f"### Batch test summary\n\n"
                f"- **Images loaded:** {len(batch_paths)}\n"
                f"- **Images scored against known classes:** {scored_count}\n"
                f"- **Correct:** {correct_count}\n"
                f"- **Batch accuracy:** {accuracy:.4f}"
            )
        else:
            summary = (
                f"### Batch test summary\n\n"
                f"- **Images loaded:** {len(batch_paths)}\n"
                f"- No filenames matched the current model classes, so accuracy was not computed."
            )

        return results, summary
    except Exception as exc:
        return [], f"Batch test failed: {exc}"


# ============================================================
# Feature map visualization
# ============================================================

def get_module_by_name(model: DynamicCNN, layer_name: str) -> nn.Module:
    for name, module in model.named_submodules():
        if name == layer_name:
            return module
    raise KeyError(f"Layer not found: {layer_name}")



def plot_feature_maps(activation: torch.Tensor, title: str) -> Image.Image:
    activation = activation.detach().cpu()
    if activation.ndim == 4:
        activation = activation[0]
        channels = activation.shape[0]
        show_n = min(16, channels)
        cols = 4
        rows = math.ceil(show_n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows))
        axes = np.array(axes).reshape(-1)
        for i in range(show_n):
            axes[i].imshow(activation[i].numpy(), cmap="viridis")
            axes[i].set_title(f"ch {i}")
            axes[i].axis("off")
        for j in range(show_n, len(axes)):
            axes[j].axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        img = fig_to_pil(fig)
        plt.close(fig)
        return img

    vec = activation.flatten().numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(vec)), vec)
    ax.set_title(title)
    ax.set_xlabel("Unit")
    ax.set_ylabel("Activation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = fig_to_pil(fig)
    plt.close(fig)
    return img



def visualize_feature_maps(state: Dict[str, Any], layer_name: str, sample_index: int):
    if not state or "model" not in state:
        return None, None, "Train a model first."
    if not state["cached_samples"]:
        return None, None, "No cached test samples are available."

    sample_index = int(sample_index) % len(state["cached_samples"])
    x_cpu, target = state["cached_samples"][sample_index]
    x = x_cpu.unsqueeze(0).to(DEVICE)

    model = state["model"]
    target_module = get_module_by_name(model, layer_name)
    captured = {}

    def hook(_module, _inputs, output):
        captured["activation"] = output

    handle = target_module.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
    handle.remove()

    activation = captured["activation"]
    fmap_img = plot_feature_maps(activation, f"Feature maps: {layer_name}")

    original = denormalize_tensor(x_cpu, state["mean"], state["std"]).clamp(0, 1)
    original_pil = transforms.ToPILImage()(original)

    md = f"""
### Sample inspection

- **Layer:** {layer_name}
- **Sample index:** {sample_index}
- **True label:** {state['class_names'][target]}
- **Predicted label:** {state['class_names'][pred]}
- **Confidence:** {float(probs[0, pred].cpu().item()):.4f}

Use this view to compare how earlier layers detect simple local patterns while later layers become more class-specific.
"""
    return original_pil, fmap_img, md


# ============================================================
# UI helpers
# ============================================================

def load_preset(name: str) -> str:
    return pretty_json(PRESET_MODELS[name])



def save_model_action(state: Dict[str, Any], model_name: str):
    try:
        if not model_name.strip():
            model_name = "cnn_experiment"
        path = export_checkpoint(state, model_name)
        return f"Saved checkpoint to:\n{path}"
    except Exception as exc:
        return f"Save failed: {exc}"


DESCRIPTION = """
# CNN Playground for digits and simple image classification

This app is intentionally small and hackable:

- pick a **preset** or edit the **JSON architecture** directly,
- train on **MNIST** or **FashionMNIST**,
- compare **parameter count, speed, and accuracy**,
- inspect **layer-by-layer shapes and feature maps**,
- test the trained model by **drawing on the canvas**.

## Architecture JSON format

Supported layer types in the JSON list:

- `conv`
- `maxpool`
- `avgpool`
- `flatten`
- `linear`
- `dropout`
- `activation`

Example conv block:
```json
{
  "type": "conv",
  "out_channels": 16,
  "kernel_size": 3,
  "padding": "same",
  "activation": "relu",
  "batch_norm": true,
  "pool": "max",
  "pool_kernel_size": 2,
  "dropout": 0.1
}
```

Example linear block:
```json
{
  "type": "linear",
  "out_features": 128,
  "activation": "relu",
  "dropout": 0.2
}
```

Final classifier layer can use:
```json
{"type": "linear", "out_features": "num_classes"}
```
"""


with gr.Blocks(title="CNN Playground") as demo:
    gr.Markdown(DESCRIPTION)
    app_state = gr.State(None)
    batch_state = gr.State([])

    with gr.Tab("Build & Train"):
        with gr.Row():
            with gr.Column(scale=1):
                dataset_name = gr.Dropdown(list(DATASET_REGISTRY.keys()), value="MNIST", label="Dataset")
                preset_name = gr.Dropdown(list(PRESET_MODELS.keys()), value="TinyCNN", label="Preset model")
                load_preset_btn = gr.Button("Load preset into editor")
                spec_text = gr.Code(value=DEFAULT_SPEC_TEXT, language="json", label="Architecture JSON")

            with gr.Column(scale=1):
                epochs = gr.Slider(1, 20, value=3, step=1, label="Epochs")
                batch_size = gr.Slider(16, 256, value=64, step=16, label="Batch size")
                learning_rate = gr.Number(value=1e-3, label="Learning rate")
                optimizer_name = gr.Dropdown(["AdamW", "SGD"], value="AdamW", label="Optimizer")
                weight_decay = gr.Number(value=1e-4, label="Weight decay")
                val_ratio = gr.Slider(0.05, 0.3, value=0.1, step=0.05, label="Validation ratio")
                max_train_samples = gr.Slider(1000, 60000, value=12000, step=1000, label="Max train samples (for fast experiments)")
                max_test_samples = gr.Slider(1000, 10000, value=3000, step=1000, label="Max test samples")
                train_btn = gr.Button("Train model", variant="primary")

        status_text = gr.Markdown()
        with gr.Row():
            history_image = gr.Image(label="Training curves")
            confusion_image = gr.Image(label="Confusion matrix")
        summary_md = gr.Markdown()

        with gr.Row():
            model_save_name = gr.Textbox(value="cnn_experiment", label="Checkpoint name prefix")
            save_btn = gr.Button("Save trained model")
        save_status = gr.Textbox(label="Save status")

    with gr.Tab("Test by Drawing"):
        with gr.Row():
            with gr.Column(scale=1):
                sketch = gr.Sketchpad(
                    label="Draw a digit",
                    image_mode="L",
                    canvas_size=(280, 280),
                    brush=gr.Brush(colors=["#000000", "#FFFFFF"], default_size=18, color_mode="fixed"),
                )
                with gr.Row():
                    predict_btn = gr.Button("Predict drawing", variant="primary")
                    save_drawing_btn = gr.Button("Save drawing")
                drawing_filename = gr.Textbox(value="7_1.png", label="Save drawing as", placeholder="Example: 7_1.png")
                drawing_save_status = gr.Textbox(label="Drawing save status")
            with gr.Column(scale=1):
                processed_image = gr.Image(label="Preprocessed 28x28 input")
                probability_image = gr.Image(label="Class probabilities")
                prediction_md = gr.Markdown()

    with gr.Tab("Batch Test"):
        with gr.Row():
            batch_directory = gr.Textbox(
                label="Image directory",
                placeholder="Example: C:/Users/you/Documents/my_digit_dataset",
            )
            load_batch_btn = gr.Button("Load images")
            run_batch_btn = gr.Button("Run batch test", variant="primary")
        batch_load_md = gr.Markdown(
            "Use a directory of images named like `1_1.jpg`, `7_2.png`, `a_1.png`, or `tree_1.png`."
        )
        batch_loaded_df = gr.Dataframe(
            headers=["filename", "expected_label"],
            datatype=["str", "str"],
            label="Loaded files",
        )
        batch_results_df = gr.Dataframe(
            headers=["filename", "expected_label", "predicted_label", "confidence", "result"],
            datatype=["str", "str", "str", "number", "str"],
            label="Batch results",
        )
        batch_summary_md = gr.Markdown()

    with gr.Tab("Feature Maps"):
        with gr.Row():
            feature_layer = gr.Dropdown(choices=[], label="Layer")
            feature_index = gr.Slider(0, 127, value=0, step=1, label="Cached test sample index")
            feature_btn = gr.Button("Visualize features", variant="primary")
        with gr.Row():
            sample_image = gr.Image(label="Input sample")
            feature_image = gr.Image(label="Feature maps / activations")
        feature_md = gr.Markdown()

    load_preset_btn.click(load_preset, inputs=preset_name, outputs=spec_text)
    train_btn.click(
        train_experiment,
        inputs=[
            dataset_name,
            preset_name,
            spec_text,
            epochs,
            batch_size,
            learning_rate,
            optimizer_name,
            weight_decay,
            val_ratio,
            max_train_samples,
            max_test_samples,
        ],
        outputs=[app_state, history_image, confusion_image, summary_md, feature_layer, status_text],
    )
    predict_btn.click(predict_from_canvas, inputs=[app_state, sketch], outputs=[processed_image, probability_image, prediction_md])
    save_drawing_btn.click(save_drawing_action, inputs=[sketch, drawing_filename], outputs=drawing_save_status)
    load_batch_btn.click(load_batch_directory, inputs=[batch_directory], outputs=[batch_state, batch_loaded_df, batch_load_md])
    run_batch_btn.click(run_batch_test, inputs=[app_state, batch_state], outputs=[batch_results_df, batch_summary_md])
    feature_btn.click(visualize_feature_maps, inputs=[app_state, feature_layer, feature_index], outputs=[sample_image, feature_image, feature_md])
    save_btn.click(save_model_action, inputs=[app_state, model_save_name], outputs=save_status)


if __name__ == "__main__":
    demo.launch()
