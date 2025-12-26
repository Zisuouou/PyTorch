# 디렉터리 구조 (권장)
# D:\AI_SVT_Training_mk\
#   ├─ images\                 # 원본/증강 이미지 (확장자 섞여도 OK)
#   ├─ annotations\annos\      # VOC XML
#   ├─ labels.json              # 클래스 매핑 (background:0 필수)
#   ├─ train_result_pt\         # (자동 생성) PyTorch 텐서보드 로그/체크포인트
#   ├─ output_inference_graph\saved_model\   # (내보내기 산출물)
#   └─ object_detection_pt\     # 아래 파이썬 스크립트들 위치

# ---------------------------------------------------------------------------
# object_detection_pt/voc_dataset.py
# ---------------------------------------------------------------------------
import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    labels = []
    # support either <name> text string labels with a mapping provided by labels.json
    for obj in root.findall("object"):
        name = obj.find("name").text
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(name)
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)
    return boxes, labels, (width, height)

class VOCDataset(Dataset):
    """
    Minimal Pascal VOC-style dataset that reads:
      - images_dir: folder of images (jpg/png)
      - annos_dir:  folder of VOC XML files, one per image
    Label mapping comes from a 'labels.json' file at --labels_json path:
        {
          "background": 0,
          "Dent": 1,
          "Scratch": 2,
          "Crack": 3,
          "Other": 4
        }
    (background MUST be 0; class ids must be contiguous 1..N)
    """
    def __init__(self, images_dir, annos_dir, labels_json, transforms=None, image_exts=(".jpg",".jpeg",".png",".bmp")):
        self.images_dir = images_dir
        self.annos_dir  = annos_dir
        self.labels = json_load(labels_json)
        self.name2id = self.labels
        # reverse map id->name (excluding background 0 for convenience)
        self.id2name = {v:k for k,v in self.name2id.items()}
        self.transforms = transforms
        # index XMLs, keep only those with matching image
        xmls = sorted(glob.glob(os.path.join(self.annos_dir, "*.xml")))
        self.samples = []
        for x in xmls:
            stem = os.path.splitext(os.path.basename(x))[0]
            img_path = None
            for ext in image_exts:
                p = os.path.join(self.images_dir, stem + ext)
                if os.path.isfile(p):
                    img_path = p
                    break
            if img_path:
                self.samples.append((img_path, x))
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No (image, xml) pairs found under {images_dir} / {annos_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, xml_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        boxes_xyxy, label_names, _ = parse_voc_xml(xml_path)

        # map label names to ids
        labels = []
        for n in label_names:
            if n not in self.name2id:
                raise KeyError(f"Label '{n}' not found in labels.json")
            labels.append(self.name2id[n])
        # empty boxes edge-case
        if len(boxes_xyxy) == 0:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels_t,
            "image_id": torch.tensor([idx])
        }
        if self.transforms:
            img = self.transforms(img)
        # torchvision models expect C,H,W tensor float [0..1]
        import torchvision.transforms as T
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)
        return img, target

def json_load(p):
    import json
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def collate_fn(batch):
    # batch is list of (img, target). Need to return tuple(list[Tensor], list[Dict])
    imgs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return imgs, targets

# ---------------------------------------------------------------------------
# object_detection_pt/utils.py
# ---------------------------------------------------------------------------
import os
import shutil
import time
import torch

class SmoothedValue:
    def __init__(self, momentum=0.98):
        self.momentum = momentum
        self.val = None
    
    def update(self, x):
        x = float(x)
        if self.val is None:
            self.val = x
        else:
            self.val = self.momentum * self.val + (1 - self.momentum) * x
        return self.val

def save_checkpoint(state, ckpt_dir, step, is_best=False, best_at_07=False):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_step{step}.pth")
    torch.save(state, path)
    # update pointers
    latest = os.path.join(ckpt_dir, "latest.pth")
    shutil.copy2(path, latest)
    if is_best:
        bestp = os.path.join(ckpt_dir, "best.pth")
        shutil.copy2(path, bestp)
    if best_at_07:
        best07 = os.path.join(ckpt_dir, "best_at_0.7.pth")
        shutil.copy2(path, best07)

def format_seconds(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ---------------------------------------------------------------------------
# object_detection_pt/model_main_pt_FRCNN_res50.py
# ---------------------------------------------------------------------------
# PyTorch Faster R-CNN ResNet50-FPN training loop
# Mirrors TF-style workflow: TensorBoard logs, step-based training, checkpoints,
# and "0.7 point" best checkpoint snapshot (best_at_0.7.pth).

import os
import json
import time
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from voc_dataset import VOCDataset, collate_fn
from utils import SmoothedValue, save_checkpoint, format_seconds
from utils.voc_dataset import VOCDataset        # +

def build_model(num_classes, pretrained=True):
    # Load COCO-pretrained backbone+head, then replace predictor with our num_classes
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--images_dir", required=True, help="Path to images folder")
    p.add_argument("--annos_dir", required=True, help="Path to VOC XML folder")
    p.add_argument("--labels_json", required=True, help="labels.json path (with background:0)")
    # Training schedule (step-based to mirror TF behavior)
    p.add_argument("--max_steps", type=int, default=10000, help="Total optimizer steps")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ckpt_interval", type=int, default=1000, help="Save ckpt every N steps")
    # Optimizer / HPs
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--lr_step", type=int, default=8000, help="StepLR milestone in steps")
    # I/O
    p.add_argument("--train_result", default="train_result_pt", help="TensorBoard logdir")
    p.add_argument("--ckpt_dir", default="train_result_pt/ckpts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default="", help="Path to ckpt to resume from")
    p.add_argument("--save_every_image_vis", action="store_true", help="Log images with boxes intermittently")
    # 0.7 checkpoint logic
    p.add_argument("--save_best_at_0p7", action="store_true", help="At step>=0.7*max_steps, save best_at_0.7.pth on improvement")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.train_result, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}   (CUDA available: {torch.cuda.is_available()})")

    dataset = VOCDataset(args.images_dir, args.annos_dir, args.labels_json)
    num_classes = max(dataset.id2name.keys()) + 1  # background=0, classes 1..N
    print(f"[Dataset] {len(dataset)} images  /  classes={num_classes-1}  (labels.json background+{num_classes-1} classes)")

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    model = build_model(num_classes=num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.lr_step], gamma=args.lr_gamma)

    # TensorBoard
    writer = SummaryWriter(log_dir=args.train_result)
    # try to add graph once
    try:
        import torchvision.transforms as T
        from PIL import Image
        img0, _ = dataset[0]
        if img0.ndim == 3:
            writer.add_graph(model, (img0.unsqueeze(0).to(device),))
    except Exception as e:
        print(f"[WARN] add_graph failed: {e}")

    step = 0
    best_loss = float("inf")
    loss_ema = SmoothedValue()
    start_time = time.time()
    model.train()

    # If resume
    if args.resume and os.path.isfile(args.resume):
        print(f"[Resume] Loading: {args.resume}")
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        step = state.get("step", 0)
        best_loss = state.get("best_loss", float("inf"))
        print(f"[Resume] step={step}, best_loss={best_loss:.6f}")

    dl_iter = iter(data_loader)
    while step < args.max_steps:
        try:
            imgs, targets = next(dl_iter)
        except StopIteration:
            dl_iter = iter(data_loader)
            imgs, targets = next(dl_iter)

        imgs = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()

        step += 1
        loss_val = float(losses.detach().cpu())
        ema = loss_ema.update(loss_val)
        lr_now = optimizer.param_groups[0]["lr"]

        # TB logs
        if step % 10 == 0 or step == 1:
            writer.add_scalar("train/loss_total", loss_val, step)
            writer.add_scalar("train/loss_ema", ema, step)
            writer.add_scalar("train/lr", lr_now, step)

        if step % 100 == 0 or step == 1:
            elapsed = time.time() - start_time
            eta = (args.max_steps - step) * (elapsed / max(step,1))
            print(f"[{step:6d}/{args.max_steps}] loss={loss_val:.4f} ema={ema:.4f} lr={lr_now:.5f} | elapsed {format_seconds(elapsed)} ETA {format_seconds(eta)}")

        if step % args.ckpt_interval == 0 or step == args.max_steps:
            is_best = ema < best_loss
            if is_best:
                best_loss = ema
            # 0.7 logic
            best_at_07 = False
            if args.save_best_at_0p7 and step >= int(0.7 * args.max_steps) and is_best:
                best_at_07 = True
            save_checkpoint({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best_loss": best_loss,
                "labels_json": os.path.abspath(args.labels_json),
                "num_classes": num_classes
            }, args.ckpt_dir, step, is_best=is_best, best_at_07=best_at_07)

        # LR step on schedule
        if step == args.lr_step:
            scheduler.step()

    writer.close()
    print("[DONE] Training finished.")
    # final save (alias)
    save_checkpoint({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_loss": best_loss,
        "labels_json": os.path.abspath(args.labels_json),
        "num_classes": num_classes
    }, args.ckpt_dir, step, is_best=True, best_at_07=True)

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# object_detection_pt/export_pt.py
# ---------------------------------------------------------------------------
# Export checkpoints to a "saved_model" style folder hierarchy
# to mirror TF: output_inference_graph/saved_model/*
import os
import json
import argparse
import shutil
import torch

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def load_labels(labels_json_path):
    with open(labels_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_model(num_classes):
    m = fasterrcnn_resnet50_fpn(weights=None)
    in_features = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return m

def export_from_ckpt(ckpt_path, output_dir="output_inference_graph/saved_model", script_input_size=800):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    num_classes = int(state.get("num_classes", 2))
    labels_json = state.get("labels_json", None)
    if labels_json is None or not os.path.isfile(labels_json):
        raise FileNotFoundError("labels_json path missing inside checkpoint; set when training or pass --labels_json to re-export.")

    labels = load_labels(labels_json)
    os.makedirs(output_dir, exist_ok=True)

    # build & load
    model = build_model(num_classes)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    # save raw state_dict
    state_path = os.path.join(output_dir, "model_state_dict.pth")
    torch.save(model.state_dict(), state_path)

    # TorchScript (trace with dummy input)
    dummy = torch.randn(1, 3, script_input_size, script_input_size)
    with torch.no_grad():
        scripted = torch.jit.trace(model, dummy)
    script_path = os.path.join(output_dir, "model_scripted.pt")
    scripted.save(script_path)

    # copy labels + minimal pipeline meta
    shutil.copy2(labels_json, os.path.join(output_dir, "labels.json"))
    meta = {
        "framework": "pytorch",
        "arch": "fasterrcnn_resnet50_fpn",
        "script_input_size": script_input_size,
        "num_classes": num_classes
    }
    with open(os.path.join(output_dir, "pipeline.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # write a tiny inference helper next to the artifacts
    infer_py = os.path.join(output_dir, "inference.py")
    with open(infer_py, "w", encoding="utf-8") as f:
        f.write(f"""# Inference helper for exported PyTorch Faster R-CNN
import os, json
import torch
from PIL import Image
from torchvision.transforms import functional as F

BASE = os.path.dirname(__file__)
SCRIPTED = os.path.join(BASE, "model_scripted.pt")
LABELS  = os.path.join(BASE, "labels.json")

with open(LABELS, "r", encoding="utf-8") as f:
    labels = json.load(f)
id2name = {{v:k for k,v in labels.items()}}

model = torch.jit.load(SCRIPTED, map_location="cpu").eval()

def predict(image_path, score_thresh=0.5):
    img = Image.open(image_path).convert("RGB")
    timg = F.to_tensor(img).unsqueeze(0)
    with torch.no_grad():
        out = model(timg)[0]
    boxes = out["boxes"].tolist()
    scores = out["scores"].tolist()
    labs = out["labels"].tolist()
    results = []
    for b,s,l in zip(boxes, scores, labs):
        if s < score_thresh: 
            continue
        results.append({"box": b, "score": float(s), "label_id": int(l), "label_name": id2name.get(int(l), str(l))})
    return results

if __name__ == "__main__":
    import sys, pprint
    img = sys.argv[1]
    print(predict(img))
""")

    # readme
    with open(os.path.join(output_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("This folder mirrors TF's saved_model style for convenience. Use inference.py for quick tests.\n")

    print(f"[EXPORTED] {ckpt_path} -> {output_dir}")
    print(f"  - state_dict : {state_path}")
    print(f"  - scripted   : {script_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="Path to checkpoint .pth (e.g., train_result_pt/ckpts/best.pth)")
    ap.add_argument("--use_best", action="store_true", help="Shortcut for 'best.pth' under --ckpt_dir")
    ap.add_argument("--use_best07", action="store_true", help="Shortcut for 'best_at_0.7.pth' under --ckpt_dir")
    ap.add_argument("--ckpt_dir", default="train_result_pt/ckpts")
    ap.add_argument("--output_dir", default="output_inference_graph/saved_model")
    ap.add_argument("--script_input_size", type=int, default=800)
    args = ap.parse_args()

    if args.ckpt:
        ck = args.ckpt
    else:
        if args.use_best:
            ck = os.path.join(args.ckpt_dir, "best.pth")
        elif args.use_best07:
            ck = os.path.join(args.ckpt_dir, "best_at_0.7.pth")
        else:
            ck = os.path.join(args.ckpt_dir, "latest.pth")
    export_from_ckpt(ck, output_dir=args.output_dir, script_input_size=args.script_input_size)

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# object_detection_pt/inference_demo.py  (폴더 단위 테스트용)
# ---------------------------------------------------------------------------
# Quick demo script to run inference on a folder of images using the exported artifacts.
import os, glob, argparse, json, pprint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--export_dir", default="output_inference_graph/saved_model")
    ap.add_argument("--score_thresh", type=float, default=0.5)
    args = ap.parse_args()

    infer_py = os.path.join(args.export_dir, "inference.py")
    if not os.path.isfile(infer_py):
        raise FileNotFoundError(f"inference.py not found under {args.export_dir}; run export first.")

    import importlib.util
    spec = importlib.util.spec_from_file_location("pt_infer", infer_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    images = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
        images.extend(glob.glob(os.path.join(args.images_dir, ext)))
    images = sorted(images)
    print(f"[INFO] Found {len(images)} images")

    for p in images:
        res = mod.predict(p, score_thresh=args.score_thresh)
        print(os.path.basename(p))
        pprint.pprint(res[:5])

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# object_detection_pt/configs/pipeline_pt.json (예시 하이퍼파라미터)
# ---------------------------------------------------------------------------
# {
#   "framework": "pytorch",
#   "arch": "fasterrcnn_resnet50_fpn",
#   "hyperparams": {
#     "batch_size": 2,
#     "lr": 0.005,
#     "momentum": 0.9,
#     "weight_decay": 0.0005,
#     "lr_step": 8000,
#     "lr_gamma": 0.1,
#     "max_steps": 10000,
#     "ckpt_interval": 1000
#   },
#   "notes": "TF의 step 기반 루프 및 0.7 체크포인트 컨벤션을 반영"
# }

# ---------------------------------------------------------------------------
# labels.json (예시)
# ---------------------------------------------------------------------------
# {
#   "background": 0,
#   "Dent": 1,
#   "Scratch": 2,
#   "Crack": 3,
#   "Other": 4
# }

# ---------------------------------------------------------------------------
# Windows 배치 예시 (루트에 두고 실행)
# ---------------------------------------------------------------------------
# Train_PyTorch.bat
# @echo off
# REM PyTorch training starter (mirrors TF flow)
# SET ROOT=D:\AI_SVT_Training_mk
# SET IMAGES=%ROOT%\images
# SET ANNOS=%ROOT%\annotations\annos
# SET LABELS=%ROOT%\labels.json
#
# cd /d %ROOT%
# REM TensorBoard on port 6008 to avoid clashing with TF on 6007
# start "" tensorboard --logdir=train_result_pt --port=6008
#
# REM Activate env if needed (uncomment and adapt)
# REM call C:\Users\SVT\anaconda3\Scripts\activate.bat Pytorch
#
# python "%~dp0object_detection_pt\model_main_pt_FRCNN_res50.py" ^
#   --images_dir "%IMAGES%" ^
#   --annos_dir "%ANNOS%" ^
#   --labels_json "%LABELS%" ^
#   --max_steps 10000 ^
#   --batch_size 2 ^
#   --ckpt_interval 1000 ^
#   --train_result "train_result_pt" ^
#   --ckpt_dir "train_result_pt\ckpts" ^
#   --save_best_at_0p7
#
# Export_From_TB_Step.bat
# @echo off
# REM Export using best_at_0.7 if present, otherwise best, otherwise latest
# SET ROOT=D:\AI_SVT_Training_mk
# cd /d %ROOT%
# python "%~dp0object_detection_pt\export_pt.py" --use_best07 --ckpt_dir "train_result_pt\ckpts" --output_dir "output_inference_graph\saved_model"
