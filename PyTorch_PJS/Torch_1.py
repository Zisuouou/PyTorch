import os
import glob
import xml.etree.ElementTree as ET      # 1~3줄 파일/XML 처리

import torch
from torch.utils.data import Dataset, DataLoader, random_split  # 5~6줄 PyTorch 기본
from PIL import Image   # 이미지
import torchvision
from torchvision.transforms import functional as F      # Object Detection(Faster R-CNN 사용)

from torch.utils.tensorboard import SummaryWriter   # TensofBoard 연결

# 25.12.22_ 1epoch당 걸리는 시간
import time

# 26.02.27 
from engine import train_one_epoch, evaluate
import utils # 공식 utils.py에서 collate_fn 가져오기 위해서 utils.py도 같은 디렉토리에 있어야 함

# 26.03.04 : pth to pt
from torchvision.models.detection.anchor_utils import AnchorGenerator

########################################
# 1. 설정 부분 
########################################

# 내가 만든 데이터셋 루트
DATA_ROOT = r"C:\Users\SVT\Desktop\PyTorch_PJS\annos"      # XML과 이미지 경로 설정
IMG_DIR   = os.path.join(DATA_ROOT, "image")
ANN_DIR   = os.path.join(DATA_ROOT, "xml")

# 클래스 이름 (0번은 background 고정)
# 내 라벨 이름 넣기
CLASS_NAMES = [
    "__background__",  # 0
    "crack",   # 1
]

########################################
# 2. 유틸 함수: XML 파싱해서 box/label 뽑기
########################################

def parse_voc_xml(xml_path, class_to_idx):      # Pascal VOC 형식 XML 파일을 읽어서 박스와 라벨을 추출하는 함수
    """
    Pascal VOC 형식의 XML에서
    boxes(Tensor[N,4]), labels(Tensor[N]) 를 뽑아서 리턴
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []      # 각 객체의 박스 좌표 추출
    labels = []     # 클래스 레이블 추출

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

    return boxes, labels        # 결과 : Tensor 형태로 boxes([N,4]), labels([N]) 반환


########################################
# 3. 커스텀 Dataset
########################################

class CustomVOCDataset(Dataset):        # PyTorch Dataset 상속, VOC 형식의 XML 어노테이션을 가진 커스텀 데이터셋 클래스
    def __init__(self, img_dir, ann_dir, transforms=None):      # 이미지 경로 수집, 클래스 -> 인덱스 매핑
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms

        # 이미지 리스트 수집 (jpg, jpeg, png 등 필요시 추가)
        self.img_paths = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg"))
        )

        # 클래스 이름 -> 인덱스 매핑
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    def __len__(self):      # 데이터셋 크기 반환
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1- 이미지 불러오기 (RGB로 변환)
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
        # Tensor가 아닌 파이썬 int로 전달
        target["image_id"] = torch.tensor([idx]).item()

        # area, iscrowd 는 필수는 아님, 근데 넣어주는게 좋음
        if boxes.numel() > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.zeros((0,), dtype=torch.float32)

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


class ToTensor:     # PIL 이미지를 Tensor로 변환하는 transform 클래스
    def __call__(self, image, target):
        image = F.to_tensor(image)  # [C,H,W], 0~1
        return image, target


class RandomHorizontalFlip:     # 학습데이터 좌우반전 / 그에 맞춰 박스 좌표도 반전
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
        transforms.append(RandomHorizontalFlip(0.5))        # 확률 0.5
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

def train_one_epoch(model, optimizer, data_loader, device, epoch, writer, print_freq=1):        # 1epoch 학습
    model.train()
    running_loss = 0.0
            # 모든 배치에 대해 순전파 → 손실 계산 → 역전파 → 가중치 업데이트
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        # TensorFlow 처럼 상세 Loss 기록 추가_26.02.27
        # 1에포크 내의 진행도를 반영하기 위해서 global_step 계산
        global_step = (epoch - 1) * len(data_loader) + i
        for loss_name, loss_value in loss_dict.items():
            writer.add_scalar(f"Loss_Detail/{loss_name}", loss_value.item(), global_step)   # 02.27_ 상세 Loss 기록 추가

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       # GPU / CPU 자동선택
    print("[DEVICE]", device)

    # TensorBoard writer 생성
    writer = SummaryWriter(log_dir="runs/custom_voc")   

    # 26.02.27_수정 (random_split 대신 리스트 슬라이싱 방식 사용)
    # 1. 전체 이미지 경로 리스트로 가져와 섞기
    all_img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    n_total = len(all_img_paths)

    # 랜덤 시드를 고정
    indices = torch.randperm(n_total).tolist()
    n_train = int(n_total * 0.8)

    train_indices = indices[:n_train]
    valid_indices = indices[n_train:]

    # 2. Train 데이터셋 생성
    train_dataset = CustomVOCDataset(IMG_DIR, ANN_DIR, transforms=get_transform(train=True))
    # 전체 경로 중 train 인덱스에 해당하는 것만 필터링
    train_dataset.img_paths = [all_img_paths[i] for i in train_indices[:n_train]]

    # 3. Valid 데이터셋 생성 (Transforms를 다르게 적용하기 위해 별도 생성)
    valid_dataset = CustomVOCDataset(IMG_DIR, ANN_DIR, transforms=get_transform(train=False))
    valid_dataset.img_paths = [all_img_paths[i] for i in valid_indices[:n_train]]


    print(f"[DATASET] total={n_total}, train={len(train_dataset)}, valid={len(valid_dataset)}")

    # 3- DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,   #(훈련)
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,   # 검증
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
    # 26.03.03_추가 : 체크포인트 설정
    os.makedirs("checkpoints", exist_ok=True)   # 루프 안에서 저장하기 위해 폴더를 미리 생성
    best_map = 0.0  # 최고 mAP 기록용 변수 초기화

    # 6- 학습_ 26.02.27_ 공식 engine.py의 train_one_epoch, evaluate 사용(mAP) 변경
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        # 시간 측정 시작
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # GPU 연산 완료 대기
        epoch_start = time.time()

        # 1. 학습 (상세 Loss는 함수 내부에서 writer에 기록)
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, writer, print_freq=1)
        writer.add_scalar("Loss/train", train_loss, epoch)

        # 2. 검증 mAP 측정(공식 engine.py의 evaluate 사용)
        # 이 함수가 실행되면 터미널에 IoU 표가 출력
        from engine import evaluate
        coco_evaluator = evaluate(model, valid_loader, device=device)

        # mAP 지표 기록
        mAP = coco_evaluator.coco_eval['bbox'].stats[0] # AP @[ IoU=0.50:0.95 ]
        mAP_50 = coco_evaluator.coco_eval['bbox'].stats[1] # AP @[ IoU=0.50 ]
        writer.add_scalar("Metrics/mAP", mAP, epoch)
        writer.add_scalar("Metrics/mAP_50", mAP_50, epoch)

        # 3. 검증 Loss 측정(그래프 유지용)
        model.train() # Faster R-CNN은 train 모드여야 Loss를 반환함
        valid_loss_sum = 0.0
        valid_steps = 0
        with torch.no_grad():
            for images, targets in valid_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                valid_loss_sum += losses.item()
                valid_steps += 1
        
        if valid_steps > 0:
            valid_loss = valid_loss_sum / valid_steps
            writer.add_scalar("Loss/valid", valid_loss, epoch)

        # 4. 시간 측정 종료
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # GPU 연산 완료 대기
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        writer.add_scalar("Time/epoch_time_sec", epoch_time, epoch)

        print(f"[Epoch {epoch}] train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, mAP: {mAP:.4f}, mAP_50: {mAP_50:.4f}, epoch_time: {epoch_time:.2f} sec")

        # 전체 에포크의 70% 이상 진행, 현재 mAP 가 이전의 최고 기록(best_map)보다 높으면 체크포인트 저장 _26.03.03
        if (epoch / num_epochs) >= 0.7 and mAP > best_map:
            best_map = mAP # 최고 mAP 기록 업뎃
            best_save_path = os.path.join("checkpoints", "fasterrcnn_best.pth")     # 최고 성능 모델 저장
            torch.save(model.state_dict(), best_save_path)
            print(f"New Best Model Saved at Epoch {epoch} (mAP: {best_map:.4f})")
    
    # 7- 최종 모델 저장 _학습이 완전히 끝난 후의 마지막 상태도 저장
    os.makedirs("checkpoints", exist_ok=True)
    save_path = os.path.join("checkpoints", "fasterrcnn_custom_last.pth")       # 마지막 모델 저장
    torch.save(model.state_dict(), save_path)
    print(f"[SAVED LAST] {save_path}")

    # 8. pth to pt 저장 방식 변경 (TorchScript 형식으로 저장) _26.03.04
    print("\n[INFO] Saving model in .pt format...")

    # (1) 성능이 가장 좋았던 모델을 .pt 형식으로 저장
    best_model_path = os.path.join("checkpoints", "fasterrcnn_best.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"{best_model_path} 가중치 로드 완료. TorchScript 형식으로 저장 중...")
    else:
        print("[WARN] Best model checkpoint not found. 현재 모델 상태로 저장합니다.")

    # (2) 모델은 평가(추론) 모드로 변경
    model.eval()

    # (3) TorchScript 변환 및 저장
    # FasterRCNN 모델은 일반적으로 입력이 리스트 형태이므로, 예시 입력을 만들어서 트레이싱 방식으로 저장
    try:
        scripted_model = torch.jit.script(model)  # 스크립팅 방식으로 저장 시도
        pt_save_path = os.path.join("checkpoints", "fasterrcnn_best.pt")
        scripted_model.save(pt_save_path)
        print(f"{pt_save_path} 형식으로 모델 저장 완료.")
    except Exception as e:
        print(f"[ERROR] 모델을 .pt 형식으로 저장하는 중 오류 발생: {e}")

    writer.close()


if __name__ == "__main__":
    main()
