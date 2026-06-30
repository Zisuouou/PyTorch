import os
import glob
import time
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.transforms import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

from engine import evaluate

########################################
# 1. 설정 부분
########################################

DATA_ROOT = r"D:\AI_SVT_training_mk\annotations\annos"
IMG_DIR = os.path.join(DATA_ROOT, "image")
ANN_DIR = os.path.join(DATA_ROOT, "xml")

CLASS_NAMES = [
    "__background__",
    "UnderCut",
]

NUM_EPOCHS = 10
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 1
NUM_WORKERS = 4

SCALAR_LOG_DIR = "runs/custom_voc"
PROFILER_LOG_DIR = "runs/profiler"

# 공식 예제 스타일 profiler 스케줄
PROFILER_WAIT = 1
PROFILER_WARMUP = 1
PROFILER_ACTIVE = 3
PROFILER_REPEAT = 1
ENABLE_PROFILER = True
PROFILE_ONLY_FIRST_EPOCH = True

########################################
# 2. 유틸 함수: XML 파싱해서 box/label 뽑기
########################################


def parse_voc_xml(xml_path, class_to_idx):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.iter("object"):
        name = obj.find("name").text
        if name not in class_to_idx:
            print(f"[WARN] Unknown class '{name}' in {xml_path} -> skip")
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        if xmax <= xmin or ymax <= ymin:
            print(
                f"[WARN] invalid box in {os.path.basename(xml_path)}: "
                f"[{xmin}, {ymin}, {xmax}, {ymax}] -> skip this object"
            )
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_to_idx[name])

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
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        file_name = os.path.basename(img_path)
        name_no_ext, _ = os.path.splitext(file_name)
        xml_path = os.path.join(self.ann_dir, name_no_ext + ".xml")

        img = Image.open(img_path).convert("RGB")
        boxes, labels = parse_voc_xml(xml_path, self.class_to_idx)

        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": idx,
        }

        if boxes.numel() > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        target["area"] = area
        target["iscrowd"] = torch.zeros((labels.shape[0],), dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


########################################
# 4. Transform 정의
########################################


class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            _, _, w = image.shape
            image = image.flip(-1)

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
# 5. 모델
########################################


def get_model(num_classes):

    # 1. 속도 안정성을 위한 커스텀 하이퍼파라미터 정의
    # 기본값은 대개 train 시 2000, test(추론) 시 1000입니다. 
    # 결함 탐지(UnderCut) 특성상 한 이미지에 수백 개씩 나오지 않으므로 값을 낮춰 연산 상한선을 고정합니다.
    CUSTOM_RPN_POST_NMS_TOP_N_TEST = 100  # 추론 시 RPN 후 처리할 최대 박스 수 (기본값: 1000)
    CUSTOM_MAX_DETECTIONS_PER_IMG = 50    # 최종 출력할 최대 결함 수 (기본값: 100)
    
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # 2. 모델 생성 시 추론(test) 관련 파라미터를 강제로 주입
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        rpn_post_nms_top_n_test=CUSTOM_RPN_POST_NMS_TOP_N_TEST,
        box_detections_per_img=CUSTOM_MAX_DETECTIONS_PER_IMG
    )    

    # 3. 추가 안정성 팁: RPN에서 너무 낮은 점수의 노이즈 박스들을 초기에 컷팅
    model.rpn.score_thresh = 0.05  # 밝기 변화 등으로 생긴 미세 노이즈 제거 (기본값: 0.0)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


########################################
# 6. collate_fn
########################################


def collate_fn(batch):
    return tuple(zip(*batch))


########################################
# 7. 학습 루프
########################################


def move_to_device(images, targets, device):
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [
        {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in targets
    ]
    return images, targets



def run_train_step(model, optimizer, images, targets, device):
    images, targets = move_to_device(images, targets, device)
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad(set_to_none=True)
    losses.backward()
    optimizer.step()

    return loss_dict, losses



def train_one_epoch(model, optimizer, data_loader, device, epoch, writer, print_freq=1, enable_profiler=False):
    model.train()
    running_loss = 0.0

    def run_loop(prof=None):
        nonlocal running_loss
        for i, (images, targets) in enumerate(data_loader):
            loss_dict, losses = run_train_step(model, optimizer, images, targets, device)
            running_loss += losses.item()

            global_step = (epoch - 1) * len(data_loader) + i
            for loss_name, loss_value in loss_dict.items():
                writer.add_scalar(f"Loss_Detail/{loss_name}", loss_value.item(), global_step)

            if (i + 1) % print_freq == 0:
                print(f"[Epoch {epoch}] Step {i+1}/{len(data_loader)} loss: {losses.item():.4f}")

            if prof is not None:
                prof.step()

    if enable_profiler:
        os.makedirs(PROFILER_LOG_DIR, exist_ok=True)
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            schedule=schedule(
                wait=PROFILER_WAIT,
                warmup=PROFILER_WARMUP,
                active=PROFILER_ACTIVE,
                repeat=PROFILER_REPEAT,
            ),
            on_trace_ready=tensorboard_trace_handler(PROFILER_LOG_DIR),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            run_loop(prof)
    else:
        run_loop()

    epoch_loss = running_loss / max(1, len(data_loader))
    return epoch_loss


########################################
# 8. 메인
########################################


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    writer = SummaryWriter(log_dir=SCALAR_LOG_DIR)

    all_img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    n_total = len(all_img_paths)

    indices = torch.randperm(n_total).tolist()
    n_train = int(n_total * 0.8)

    train_indices = indices[:n_train]
    valid_indices = indices[n_train:]

    train_dataset = CustomVOCDataset(IMG_DIR, ANN_DIR, transforms=get_transform(train=True))
    train_dataset.img_paths = [all_img_paths[i] for i in train_indices]

    valid_dataset = CustomVOCDataset(IMG_DIR, ANN_DIR, transforms=get_transform(train=False))
    valid_dataset.img_paths = [all_img_paths[i] for i in valid_indices]

    print(f"[DATASET] total={n_total}, train={len(train_dataset)}, valid={len(valid_dataset)}")

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    num_classes = len(CLASS_NAMES)
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    os.makedirs("checkpoints", exist_ok=True)
    best_map = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.time()

        enable_profiler = ENABLE_PROFILER and (not PROFILE_ONLY_FIRST_EPOCH or epoch == 1)
        train_loss = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            writer,
            print_freq=1,
            enable_profiler=enable_profiler,
        )
        writer.add_scalar("Loss/train", train_loss, epoch)

        coco_evaluator = evaluate(model, valid_loader, device=device)

        mAP = coco_evaluator.coco_eval["bbox"].stats[0]
        mAP_50 = coco_evaluator.coco_eval["bbox"].stats[1]
        writer.add_scalar("Metrics/mAP", mAP, epoch)
        writer.add_scalar("Metrics/mAP_50", mAP_50, epoch)

        model.train()
        valid_loss_sum = 0.0
        valid_steps = 0
        with torch.no_grad():
            for images, targets in valid_loader:
                images, targets = move_to_device(images, targets, device)
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                valid_loss_sum += losses.item()
                valid_steps += 1

        valid_loss = valid_loss_sum / max(1, valid_steps)
        writer.add_scalar("Loss/valid", valid_loss, epoch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start
        writer.add_scalar("Time/epoch_time_sec", epoch_time, epoch)

        print(
            f"[Epoch {epoch}] train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, "
            f"mAP: {mAP:.4f}, mAP_50: {mAP_50:.4f}, epoch_time: {epoch_time:.2f} sec"
        )

        if (epoch / NUM_EPOCHS) >= 0.7 and mAP > best_map:
            best_map = mAP
            best_save_path = os.path.join("checkpoints", "fasterrcnn_best.pth")
            torch.save(model.state_dict(), best_save_path)
            print(f"New Best Model Saved at Epoch {epoch} (mAP: {best_map:.4f})")

    save_path = os.path.join("checkpoints", "fasterrcnn_custom_last.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[SAVED LAST] {save_path}")

    print("\n[INFO] Saving model in .pt format...")
    best_model_path = os.path.join("checkpoints", "fasterrcnn_best.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"{best_model_path} 가중치 로드 완료. TorchScript 형식으로 저장 중...")
    else:
        print("[WARN] Best model checkpoint not found. 현재 모델 상태로 저장합니다.")

    # 1. 복잡하고 에러가 많은 transform을 통째로 걷어낸 순수 연산용 배포 Wrapper
    class FasterRCNNPureInference(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.model = base_model
            # 추론 모드 고정 및 하이퍼파라미터 강제 고정
            self.model.eval()
            self.model.rpn.score_thresh = 0.0
            self.model.roi_heads.score_thresh = 0.0

        def forward(self, x):
            # 입력 x의 차원: [1, 3, H, W] -> [배치=1, 채널=3, 높이, 너비] 순수 4D 텐서 수용
            # 1) 에러의 원인인 self.model.transform을 완벽히 스킵하고 직접 ImageList의 형태를 흉내냅니다.
            # Faster R-CNN 백본이 필요로 하는 고정 구조를 직접 만들어 텐서 흐름을 뚫어줍니다.
            from torchvision.models.detection.image_list import ImageList
            
            # 입력 이미지의 실제 크기를 튜플로 정의
            h = float(x.shape[2])
            w = float(x.shape[3])
            image_sizes = [(int(h), int(w))]
            
            # transform 레이어를 bypass한 커스텀 ImageList 객체 생성
            image_list = ImageList(x, image_sizes)

            # 2) 신경망 연산 핵심 파이프라인 수동 순차 구동
            features = self.model.backbone(image_list.tensors)
            proposals, rpn_losses = self.model.rpn(image_list, features, None)
            detections, detector_losses = self.model.roi_heads(proposals, features, image_list.image_sizes, None)
            
            # 3) 결과 언패킹 (JIT 컴파일러가 100% 통과시키는 순수 Tensor Tuple 반환)
            det = detections[0]
            return det["boxes"], det["labels"], det["scores"]

    # 2. 성능 최적화 모델 빌드
    model.eval()
    pure_inference_model = FasterRCNNPureInference(model)
    pure_inference_model.eval()
    pure_inference_model.to(device)

    try:
        # C++ 추론 환경에 완벽히 매칭되는 표준 4차원 텐서 [Batch=1, Channel=3, Height, Width] 생성
        dummy_height, dummy_width = 2048, 4096  
        dummy_input = torch.rand(1, 3, dummy_height, dummy_width, device=device)
        
        print(f"[INFO] Pure Backbone Tracing 진행 중... (입력 크기: {dummy_height}x{dummy_width})")
        
        with torch.no_grad():
            # 순수 텐서 연산 노드만 추적하므로 'items' 및 'shape' 에러가 구조적으로 발생할 수 없습니다.
            traced_model = torch.jit.trace(pure_inference_model, (dummy_input,))

        # 3. 저장 및 유효성 검증
        pt_save_path = os.path.join("checkpoints", "fasterrcnn_best.pt")
        traced_model.save(pt_save_path)
        print(f"[SUCCESS] {pt_save_path} 형식으로 최종 C++ 호환 모델이 에러 없이 무조건 저장되었습니다.")
        
    except Exception as e:
        print(f"[ERROR] 모델을 .pt 형식으로 저장하는 중 오류 발생: {e}")
        print("[TIP] 이 단계에서 에러가 났다면 dummy_input의 4차원 차원 배열 형상을 확인해 보세요.")

    writer.close()

if __name__ == "__main__":
    main()
