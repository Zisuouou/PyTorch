import os
import glob
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision
from torchvision.transforms import functional as F

from torch.utils.tensorboard import SummaryWriter   # TensofBoard 연결

# 12.22_ 1epoch당 걸리는 시간
import time

import re

########################################
# 1. 설정 부분 
########################################

# 내가 만든 데이터셋 루트
DATA_ROOT = r"C:\Users\SVT\Desktop\PyTorch\annos"      
IMG_DIR   = os.path.join(DATA_ROOT, "image")
ANN_DIR   = os.path.join(DATA_ROOT, "xml")

# 클래스 이름 (0번은 background 고정)
# 내 라벨 이름 넣기
CLASS_NAMES = [
    "__background__",  # 0
    "MilPinContamin"
]


########################################
# 2. 유틸 함수: XML 파싱해서 box/label 뽑기
########################################

def parse_voc_xml(xml_path, class_to_idx):
    """
    Pascal VOC 형식의 XML에서
    boxes(Tensor[N,4]), labels(Tensor[N]) 를 뽑아서 리턴
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.iter("object"):
        name = obj.find("name").text
        if name not in class_to_idx:
            # labelimg에서 엉뚱한 클래스 들어간 경우 스킵
            print(f"[WARN] Unknown class '{name}' in {xml_path} -> skip")
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # 잘못된 박스(좌표가 같은 경우 등) 스킵
        if xmax <= xmin or ymax <= ymin:
            print(f"[WARN] invalid box in {os.path.basename(xml_path)}:"
                  f"[{xmin}, {ymin}, {xmax}, {ymax}] -> skip this object")
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_to_idx[name])

    if len(boxes) == 0:
        # 박스 없으면 dummy 하나 넣고 background로 처리해도 괜찮은데 걍 스킵
        pass

    if len(boxes) == 0:
        # 박스가 하나도 없으면 모델이 학습하기 힘듦
        # 나중에 Dataset 에서 이미지 자체를 스킵하도록 처리 가능, 여기서는 그대로 리턴(빈 리스트)
        pass

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)

    return boxes, labels


########################################
# 3. 커스텀 Dataset
########################################

class CustomVOCDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms

        # 이미지 리스트 수집 (jpg, jpeg, png 등 필요시 추가)
        self.img_paths = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg"))
        )

        # 클래스 이름 -> 인덱스 매핑
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1- 이미지 불러오기
        img_path = self.img_paths[idx]
        file_name = os.path.basename(img_path)
        name_no_ext, _ = os.path.splitext(file_name)
        xml_path = os.path.join(self.ann_dir, name_no_ext + ".xml")

        img = Image.open(img_path).convert("RGB")

        # 2- XML 파싱
        boxes, labels = parse_voc_xml(xml_path, self.class_to_idx)

        # 박스가 하나도 없으면(라벨이 없는 이미지) -> 간단히 dummy 하나 넣거나,
        # 여기선 에러 대신 배경만 있는 이미지로 취급
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # 3- target dict 구성 (Faster R-CNN 형식)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        # area, iscrowd 는 필수는 아님, 근데 넣어주는게 좋음
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["area"] = area
        target["iscrowd"] = torch.zeros((labels.shape[0],), dtype=torch.int64)

        # 4- 변환 적용 (Tensor 변환 + augmentation)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


########################################
# 4. Transform 정의 (ToTensor + RandomHorizontalFlip)
########################################

class ComposeTransforms:
    """torchvision detection 튜토리얼 스타일 transform compose"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)  # [C,H,W], 0~1
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            _, h, w = image.shape
            image = image.flip(-1)  # 좌우 반전

            boxes = target["boxes"]
            if boxes.numel() > 0:
                xmin = w - boxes[:, 2]
                xmax = w - boxes[:, 0]
                boxes[:, 0] = xmin
                boxes[:, 2] = xmax
                target["boxes"] = boxes

        return image, target


def get_transform(train=True):
    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return ComposeTransforms(transforms)


########################################
# 5. 모델: Faster R-CNN (ResNet50 FPN)
########################################

def get_model(num_classes):
    # torchvision에서 제공하는 pre-trained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"  # torchvision>=0.13
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 새로운 헤드로 교체 (num_classes 개의 클래스)
    model.roi_heads.box_predictor = \
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            num_classes
        )
    return model


########################################
# 6. collate_fn (detection 튜토리얼 기본)
########################################

def collate_fn(batch):
    return tuple(zip(*batch))


########################################
# 7. 간단한 학습 루프

# TensorBoard에 잘 기록하려면 epoch별 평균 train loss를 리턴받는 게 편함
########################################

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1):
    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        if (i + 1) % print_freq == 0:
            loss_value = losses.item()
            print(f"[Epoch {epoch}] Step {i+1}/{len(data_loader)} "
                  f"loss: {loss_value:.4f}")
    
    # 에포크 평균 loss 리턴
    epoch_loss = running_loss / max(1, len(data_loader))
    return epoch_loss

########################################
# 8. 메인: 데이터 로더 + 학습 실행
########################################

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    # TensorBoard writer 생성
    writer = SummaryWriter(log_dir="runs/custom_voc")

    # 12.26
    CKPT_DIR = "checkpoints"
    os.makedirs(CKPT_DIR, exist_ok=True)
    FINAL_WEIGHTS_PATH = os.path.join(CKPT_DIR, "fasterrcnn_custom_final.pth")
    CKPT_PATH = os.path.join(CKPT_DIR, "checkpoint_last.pth")   # 마지막 저장(덮어쓰기)

    # 1- Dataset 생성
    full_dataset = CustomVOCDataset(IMG_DIR, ANN_DIR, transforms=get_transform(train=True))

    # 2- train/valid 나누기 (ex: 8:2)
    n_total = len(full_dataset)
    n_train = int(n_total * 0.8)
    n_valid = n_total - n_train
    train_dataset, valid_dataset = random_split(full_dataset, [n_train, n_valid])

    # valid 에는 augmentation 없게 ToTensor만 적용
    valid_dataset.dataset.transforms = get_transform(train=False)

    print(f"[DATASET] total={n_total}, train={len(train_dataset)}, valid={len(valid_dataset)}")

    # 3- DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # 4- 모델 준비
    num_classes = len(CLASS_NAMES)  # background 포함 개수
    model = get_model(num_classes)
    model.to(device)

    # 5- 옵티마이저
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # 12.26
    start_epoch = 1
    if os.path.exists(CKPT_PATH):
        print(f"[RESUME] found {CKPT_PATH} -> loading")
        ckpt = torch.load(CKPT_PATH, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1

        print(f"[RESUME] start_epoch={start_epoch}")
    else:
        print("[RESUME] no checkpoint -> start from scratch")

    # 6- 학습   __TF랑 속도 비교 / step 기준 튜닝이 목적 (훈련만 시간 체크)
    num_epochs = 10
    
    for epoch in range(start_epoch, num_epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.time()

        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1)
        writer.add_scalar("Loss/train", train_loss, epoch)

        model.train()
        valid_loss_sum = 0.0
        avg_valid_loss = 0.0    # 12.26
        valid_steps = 0

        with torch.no_grad():
            for images, targets in valid_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                valid_loss_sum += losses.item()
                valid_steps += 1

        if valid_steps > 0:
                avg_valid_loss = valid_loss_sum / valid_steps
                writer.add_scalar("Loss/valid", avg_valid_loss, epoch)  # 추가
        if torch.cuda.is_available():
                torch.cuda.synchronize()
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        writer.add_scalar("Time/epoch_total_sec", epoch_time, epoch)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"valid_loss={avg_valid_loss:.4f} "
            f"time(total)={epoch_time:.2f}s"
        )
        # 12.26 - epoch checkpoint 저장
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": avg_valid_loss
        }, CKPT_PATH)

    # 7- 최종 모델 저장
    os.makedirs("checkpoints", exist_ok=True)
    # save_path = os.path.join("checkpoints", "fasterrcnn_custom.pth")
    torch.save(model.state_dict(), FINAL_WEIGHTS_PATH)
    print(f"[FINAL SAVED] {FINAL_WEIGHTS_PATH}")


    writer.close()


if __name__ == "__main__":
    main()
