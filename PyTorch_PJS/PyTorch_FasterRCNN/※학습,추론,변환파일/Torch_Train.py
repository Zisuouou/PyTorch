import os
import glob
import time
import xml.etree.ElementTree as ET
import json
import multiprocessing
import random
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
import torchvision
from torchvision.transforms import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

from engine import evaluate

# 전체적인 구조
# 1. 모델 종류 : Object Detection (Faster R-CNN 모델)
# 2. Backbone : ResNet50
# 3. Feature Pyramid : FPN

########################################
# 1. 설정 부분
########################################

DATA_ROOT = r"C:\Users\SVT\Desktop\PyTorch_PJS\annos"
IMG_DIR = os.path.join(DATA_ROOT, "image")
ANN_DIR = os.path.join(DATA_ROOT, "xml")

# PJS: annos/image, annos/xml 아래에 train/valid 폴더를 분리해서 사용
TRAIN_IMG_DIR = os.path.join(IMG_DIR, "train")
VALID_IMG_DIR = os.path.join(IMG_DIR, "valid")
TRAIN_ANN_DIR = os.path.join(ANN_DIR, "train")
VALID_ANN_DIR = os.path.join(ANN_DIR, "valid")

# 기존처럼 annos/image/*.jpg, annos/xml/*.xml 형태로만 파일이 있으면
# 최초 실행 시 train/valid 폴더를 만들고 복사 방식으로 자동 분리
AUTO_SPLIT_IF_NEEDED = True
TRAIN_RATIO = 0.8
SPLIT_SEED = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

CLASS_NAMES = [
    "__background__",
    "102embossed",
]

NUM_EPOCHS = 10
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 1
NUM_WORKERS = 2

SCALAR_LOG_DIR = "runs/custom_voc"
PROFILER_LOG_DIR = "runs/profiler"

# 공식 예제 스타일 profiler 스케줄
PROFILER_WAIT = 1
PROFILER_WARMUP = 1
PROFILER_ACTIVE = 3
PROFILER_REPEAT = 1
ENABLE_PROFILER = True
PROFILE_ONLY_FIRST_EPOCH = True

def _as_bool(value, default=False):
    """GUI/환경변수에서 넘어온 bool 값을 안전하게 변환"""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


# 26.06.04_PJS 
def apply_gui_overrides_from_env():
    """
    GUI에서 전달한 설정값을 환경변수 PYTORCH_GUI_CONFIG로 받아서
    이 훈련 파일의 전역 설정값에 반영
    """
    global DATA_ROOT, IMG_DIR, ANN_DIR
    global TRAIN_IMG_DIR, VALID_IMG_DIR, TRAIN_ANN_DIR, VALID_ANN_DIR
    global AUTO_SPLIT_IF_NEEDED, TRAIN_RATIO, SPLIT_SEED
    global ENABLE_OFFLINE_AUGMENTATION, OFFLINE_AUGMENTATION_OVERWRITE, USE_ONLINE_AUGMENTATION
    global OFFLINE_BRIGHTNESS_FACTOR, OFFLINE_CONTRAST_FACTOR, OFFLINE_SHIFT_X_RATIO, OFFLINE_SHIFT_Y_RATIO
    global CLASS_NAMES
    global NUM_EPOCHS, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, NUM_WORKERS
    global SCALAR_LOG_DIR, PROFILER_LOG_DIR
    global PROFILER_WAIT, PROFILER_WARMUP, PROFILER_ACTIVE, PROFILER_REPEAT
    global ENABLE_PROFILER, PROFILE_ONLY_FIRST_EPOCH

    raw = os.environ.get("PYTORCH_GUI_CONFIG", "").strip()
    if not raw:
        return
    
    try:
        cfg = json.loads(raw)
    except Exception as e:
        print(f"[WARN] PYTORCH_GUI_CONFIG 파싱 실패: {e}")
        return
    
    overrides = cfg.get("overrides", cfg)

    DATA_ROOT = overrides.get("DATA_ROOT", DATA_ROOT)
    IMG_DIR = overrides.get("IMG_DIR", os.path.join(DATA_ROOT, "image"))
    ANN_DIR = overrides.get("ANN_DIR", os.path.join(DATA_ROOT, "xml"))

    TRAIN_IMG_DIR = overrides.get("TRAIN_IMG_DIR", os.path.join(IMG_DIR, "train"))
    VALID_IMG_DIR = overrides.get("VALID_IMG_DIR", os.path.join(IMG_DIR, "valid"))
    TRAIN_ANN_DIR = overrides.get("TRAIN_ANN_DIR", os.path.join(ANN_DIR, "train"))
    VALID_ANN_DIR = overrides.get("VALID_ANN_DIR", os.path.join(ANN_DIR, "valid"))

    AUTO_SPLIT_IF_NEEDED = _as_bool(overrides.get("AUTO_SPLIT_IF_NEEDED", AUTO_SPLIT_IF_NEEDED), AUTO_SPLIT_IF_NEEDED)
    TRAIN_RATIO = float(overrides.get("TRAIN_RATIO", TRAIN_RATIO))
    SPLIT_SEED = int(overrides.get("SPLIT_SEED", SPLIT_SEED))

    ENABLE_OFFLINE_AUGMENTATION = _as_bool(
        overrides.get("ENABLE_OFFLINE_AUGMENTATION", ENABLE_OFFLINE_AUGMENTATION),
        ENABLE_OFFLINE_AUGMENTATION,
    )
    OFFLINE_AUGMENTATION_OVERWRITE = _as_bool(
        overrides.get("OFFLINE_AUGMENTATION_OVERWRITE", OFFLINE_AUGMENTATION_OVERWRITE),
        OFFLINE_AUGMENTATION_OVERWRITE,
    )
    USE_ONLINE_AUGMENTATION = _as_bool(
        overrides.get("USE_ONLINE_AUGMENTATION", USE_ONLINE_AUGMENTATION),
        USE_ONLINE_AUGMENTATION,
    )
    OFFLINE_BRIGHTNESS_FACTOR = float(overrides.get("OFFLINE_BRIGHTNESS_FACTOR", OFFLINE_BRIGHTNESS_FACTOR))
    OFFLINE_CONTRAST_FACTOR = float(overrides.get("OFFLINE_CONTRAST_FACTOR", OFFLINE_CONTRAST_FACTOR))
    OFFLINE_SHIFT_X_RATIO = float(overrides.get("OFFLINE_SHIFT_X_RATIO", OFFLINE_SHIFT_X_RATIO))
    OFFLINE_SHIFT_Y_RATIO = float(overrides.get("OFFLINE_SHIFT_Y_RATIO", OFFLINE_SHIFT_Y_RATIO))

    CLASS_NAMES = overrides.get("CLASS_NAMES", CLASS_NAMES)
    if CLASS_NAMES and CLASS_NAMES[0] != "__background__":
        CLASS_NAMES = ["__background__"] + [c for c in CLASS_NAMES if c != "__background__"]
    
    NUM_EPOCHS = int(overrides.get("NUM_EPOCHS", NUM_EPOCHS))
    TRAIN_BATCH_SIZE = int(overrides.get("TRAIN_BATCH_SIZE", TRAIN_BATCH_SIZE))
    VALID_BATCH_SIZE = int(overrides.get("VALID_BATCH_SIZE", VALID_BATCH_SIZE))
    NUM_WORKERS = int(overrides.get("NUM_WORKERS", NUM_WORKERS))

    SCALAR_LOG_DIR = overrides.get("SCALAR_LOG_DIR", SCALAR_LOG_DIR)
    PROFILER_LOG_DIR = overrides.get("PROFILER_LOG_DIR", PROFILER_LOG_DIR)

    PROFILER_WAIT = int(overrides.get("PROFILER_WAIT", PROFILER_WAIT))
    PROFILER_WARMUP = int(overrides.get("PROFILER_WARMUP", PROFILER_WARMUP))
    PROFILER_ACTIVE = int(overrides.get("PROFILER_ACTIVE", PROFILER_ACTIVE))
    PROFILER_REPEAT = int(overrides.get("PROFILER_REPEAT", PROFILER_REPEAT))
    ENABLE_PROFILER = bool(overrides.get("ENABLE_PROFILER", ENABLE_PROFILER))
    PROFILE_ONLY_FIRST_EPOCH = bool(overrides.get("PROFILE_ONLY_FIRST_EPOCH", PROFILE_ONLY_FIRST_EPOCH))

    print("[GUI CONFIG APPLIED]")
    print(f" - DATA_ROOT = {DATA_ROOT}")
    print(f" - IMG_DIR = {IMG_DIR}")
    print(f" - ANN_DIR = {ANN_DIR}")
    print(f" - TRAIN_IMG_DIR = {TRAIN_IMG_DIR}")
    print(f" - VALID_IMG_DIR = {VALID_IMG_DIR}")
    print(f" - TRAIN_ANN_DIR = {TRAIN_ANN_DIR}")
    print(f" - VALID_ANN_DIR = {VALID_ANN_DIR}")
    print(f" - AUTO_SPLIT_IF_NEEDED = {AUTO_SPLIT_IF_NEEDED}")
    print(f" - TRAIN_RATIO = {TRAIN_RATIO}")
    print(f" - ENABLE_OFFLINE_AUGMENTATION = {ENABLE_OFFLINE_AUGMENTATION}")
    print(f" - OFFLINE_AUGMENTATION_OVERWRITE = {OFFLINE_AUGMENTATION_OVERWRITE}")
    print(f" - USE_ONLINE_AUGMENTATION = {USE_ONLINE_AUGMENTATION}")
    print(f" - CLASS_NAMES = {CLASS_NAMES}")
    print(f" - NUM_EPOCHS = {NUM_EPOCHS}")
    print(f" - TRAIN_BATCH_SIZE = {TRAIN_BATCH_SIZE}")
    print(f" - VALID_BATCH_SIZE = {VALID_BATCH_SIZE}")
    print(f" - NUM_WORKERS = {NUM_WORKERS}")
    print(f" - SCALAR_LOG_DIR = {SCALAR_LOG_DIR}")
    print(f" - PROFILER_LOG_DIR = {PROFILER_LOG_DIR}")

########################################
# 2. 유틸 함수: XML 파싱해서 box/label 뽑기
########################################

def list_image_files(img_dir):
    """지정 폴더 바로 아래의 이미지 파일만 수집합니다. 하위 폴더는 포함하지 않음"""
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext.upper()}")))
    return sorted(set(paths))


def find_xml_for_image(img_path, ann_dir):
    """이미지 파일명과 같은 VOC XML 파일을 찾습니다. .xml/.XML 둘 다 확인"""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    for ext in (".xml", ".XML"):
        xml_path = os.path.join(ann_dir, stem + ext)
        if os.path.exists(xml_path):
            return xml_path
    return None


def prepare_train_valid_folders():
    """
    annos/image/train, annos/image/valid, annos/xml/train, annos/xml/valid 폴더를 준비

    - 이미 train/valid 폴더에 데이터가 있으면 그대로 사용
    - train/valid 폴더가 비어 있고 기존 flat 구조(annos/image/*.jpg, annos/xml/*.xml)만 있으면
      원본을 삭제하지 않고 copy2 방식으로 train/valid에 자동 분리
    - valid는 학습용으로 사용하지 않고, evaluate/inference 단계에서만 읽음
    """
    for folder in (TRAIN_IMG_DIR, VALID_IMG_DIR, TRAIN_ANN_DIR, VALID_ANN_DIR):
        os.makedirs(folder, exist_ok=True)

    train_imgs = list_image_files(TRAIN_IMG_DIR)
    valid_imgs = list_image_files(VALID_IMG_DIR)

    if train_imgs and valid_imgs:
        print("[DATA SPLIT] 기존 train/valid 폴더 데이터를 사용")
        return

    if train_imgs or valid_imgs:
        print("[WARN] train 또는 valid 폴더 중 한쪽에만 데이터가 있음")
        print("       현재 폴더 상태를 유지, 비어 있는 쪽은 직접 채우거나 AUTO_SPLIT_IF_NEEDED를 확인 필요")
        return

    if not AUTO_SPLIT_IF_NEEDED:
        print("[DATA SPLIT] AUTO_SPLIT_IF_NEEDED=False 이므로 자동 분리를 하지 않음")
        return

    root_imgs = list_image_files(IMG_DIR)
    pairs = []
    for img_path in root_imgs:
        xml_path = find_xml_for_image(img_path, ANN_DIR)
        if xml_path is None:
            print(f"[WARN] XML 없음 -> split 제외: {os.path.basename(img_path)}")
            continue
        pairs.append((img_path, xml_path))

    if not pairs:
        print("[WARN] 자동 분리할 원본 이미지/XML 쌍이 없음")
        print(f"       확인 위치: {IMG_DIR}, {ANN_DIR}")
        return

    rng = random.Random(SPLIT_SEED)
    rng.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * TRAIN_RATIO)
    if n_total >= 2:
        n_train = max(1, min(n_train, n_total - 1))
    else:
        n_train = 1

    train_pairs = pairs[:n_train]
    valid_pairs = pairs[n_train:]

    def copy_pairs(split_name, split_pairs, dst_img_dir, dst_ann_dir):
        for src_img, src_xml in split_pairs:
            img_name = os.path.basename(src_img)
            stem = os.path.splitext(img_name)[0]
            dst_img = os.path.join(dst_img_dir, img_name)
            dst_xml = os.path.join(dst_ann_dir, stem + ".xml")
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)
            if not os.path.exists(dst_xml):
                shutil.copy2(src_xml, dst_xml)
        print(f"[DATA SPLIT] {split_name}: {len(split_pairs)}개 복사 완료")

    copy_pairs("train", train_pairs, TRAIN_IMG_DIR, TRAIN_ANN_DIR)
    copy_pairs("valid", valid_pairs, VALID_IMG_DIR, VALID_ANN_DIR)

    print("[DATA SPLIT] 원본 flat 데이터는 삭제하지 않았음")
    print(f" - train image: {TRAIN_IMG_DIR}")
    print(f" - train xml  : {TRAIN_ANN_DIR}")
    print(f" - valid image: {VALID_IMG_DIR}")
    print(f" - valid xml  : {VALID_ANN_DIR}")


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
        self.img_paths = list_image_files(self.img_dir)
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        file_name = os.path.basename(img_path)
        name_no_ext, _ = os.path.splitext(file_name)
        xml_path = find_xml_for_image(img_path, self.ann_dir)
        if xml_path is None:
            raise FileNotFoundError(f"XML 파일을 찾을 수 없습니다: {os.path.join(self.ann_dir, name_no_ext + '.xml')}")

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
# PJS: 증강은 train_dataset 에만 적용
#      valid_dataset 은 get_transform(train=False)에서 ToTensor만 적용

# 증강 강도/확률 설정값
AUG_BRIGHTNESS_PROB = 0.5
AUG_BRIGHTNESS_DELTA = 0.20      # 밝기 factor: 0.8 ~ 1.2

AUG_CONTRAST_PROB = 0.5
AUG_CONTRAST_DELTA = 0.20        # 대비 factor: 0.8 ~ 1.2

AUG_CLAHE_PROB = 0.3
AUG_CLAHE_CLIP_LIMIT = 2.0
AUG_CLAHE_TILE_GRID_SIZE = (8, 8)

AUG_GAUSSIAN_NOISE_PROB = 0.3
AUG_GAUSSIAN_NOISE_STD = 0.03    # 0~1 기준 표준편차. 0.03이면 약한 노이즈

AUG_TRANSLATE_PROB = 0.5
AUG_TRANSLATE_X_RATIO = 0.05     # 이미지 너비의 ±5% 범위 좌우 이동
AUG_TRANSLATE_Y_RATIO = 0.05     # 이미지 높이의 ±5% 범위 상하 이동

# PJS: 증강 이미지를 실제 파일로 저장하는 offline augmentation 설정
# True이면 학습 시작 전에 annos/image/train, annos/xml/train 안에
# 원본 train 데이터의 증강 이미지와 XML을 생성 valid 폴더는 절대 증강하지 않움
ENABLE_OFFLINE_AUGMENTATION = True
OFFLINE_AUGMENTATION_OVERWRITE = False   # 이미 같은 증강 파일이 있으면 다시 만들지 않음
OFFLINE_AUGMENT_MARK = "_aug_"          # 이 문자열이 들어간 파일은 다시 증강하지 않음
OFFLINE_AUG_IMAGE_EXT = ".jpg"           # 저장할 증강 이미지 확장자

# offline 저장형 증강을 사용할 때는 기본적으로 학습 중 랜덤 증강을 끔
# 이유: 저장된 증강 이미지에 또 랜덤 증강이 겹치면 과한 변형이 될 수 있음
# 학습 중에도 매 epoch 랜덤 증강을 추가하고 싶으면 True로 변경
USE_ONLINE_AUGMENTATION = False

OFFLINE_BRIGHTNESS_FACTOR = 1.20
OFFLINE_CONTRAST_FACTOR = 1.20
OFFLINE_SHIFT_X_RATIO = 0.05
OFFLINE_SHIFT_Y_RATIO = 0.05


class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def _refresh_area(target):
    """bbox 변경 후 area 값을 다시 계산"""
    boxes = target["boxes"]
    if boxes.numel() > 0:
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    else:
        target["area"] = torch.zeros((0,), dtype=torch.float32)
    return target


def _clip_boxes_and_remove_invalid(target, image_width, image_height):
    """
    이동/변형 후 이미지 밖으로 나간 bbox를 이미지 범위 안으로 자르고,
    완전히 사라진 bbox는 labels/area/iscrowd와 함께 제거
    """
    boxes = target["boxes"]

    if boxes.numel() == 0:
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.zeros((0,), dtype=torch.int64)
        target["area"] = torch.zeros((0,), dtype=torch.float32)
        target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
        return target

    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=image_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=image_height)

    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

    target["boxes"] = boxes[keep]
    target["labels"] = target["labels"][keep]

    if "iscrowd" in target:
        target["iscrowd"] = target["iscrowd"][keep]
    else:
        target["iscrowd"] = torch.zeros((target["labels"].shape[0],), dtype=torch.int64)

    target = _refresh_area(target)
    return target


def _translate_pil_image(image, dx, dy, fill=(0, 0, 0)):
    """PIL 이미지를 wrap-around 없이 dx/dy만큼 이동"""
    width, height = image.size
    translated = Image.new(image.mode, (width, height), fill)

    src_left = max(0, -dx)
    src_top = max(0, -dy)
    src_right = min(width, width - dx)
    src_bottom = min(height, height - dy)

    if src_right <= src_left or src_bottom <= src_top:
        return translated

    dst_left = max(0, dx)
    dst_top = max(0, dy)

    crop = image.crop((src_left, src_top, src_right, src_bottom))
    translated.paste(crop, (dst_left, dst_top))
    return translated


class RandomBrightness:
    def __init__(self, prob=0.5, delta=0.2):
        self.prob = prob
        self.delta = delta

    def __call__(self, image, target):
        if random.random() < self.prob:
            factor = random.uniform(1.0 - self.delta, 1.0 + self.delta)
            image = ImageEnhance.Brightness(image).enhance(factor)
        return image, target


class RandomContrast:
    def __init__(self, prob=0.5, delta=0.2):
        self.prob = prob
        self.delta = delta

    def __call__(self, image, target):
        if random.random() < self.prob:
            factor = random.uniform(1.0 - self.delta, 1.0 + self.delta)
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image, target


class RandomCLAHE:
    _warned = False

    def __init__(self, prob=0.3, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.prob = prob
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target

        try:
            import cv2

            image_np = np.array(image)
            lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size,
            )
            l_channel = clahe.apply(l_channel)

            lab = cv2.merge((l_channel, a_channel, b_channel))
            image_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            image = Image.fromarray(image_np)

        except Exception as e:
            # opencv-python이 없는 환경에서도 학습이 멈추지 않도록 1회만 경고 출력 후 원본 사용
            if not RandomCLAHE._warned:
                print(f"[WARN] CLAHE 적용 실패. opencv-python 설치 필요 가능성 있음: {e}")
                RandomCLAHE._warned = True

        return image, target


class RandomGaussianNoise:
    def __init__(self, prob=0.3, mean=0.0, std=0.03):
        self.prob = prob
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        if random.random() < self.prob:
            image_np = np.asarray(image).astype(np.float32) / 255.0
            noise = np.random.normal(self.mean, self.std, image_np.shape).astype(np.float32)
            image_np = np.clip(image_np + noise, 0.0, 1.0)
            image = Image.fromarray((image_np * 255.0).astype(np.uint8))
        return image, target


class RandomTranslate:
    def __init__(self, prob=0.5, max_dx_ratio=0.05, max_dy_ratio=0.05, fill=(0, 0, 0)):
        self.prob = prob
        self.max_dx_ratio = max_dx_ratio
        self.max_dy_ratio = max_dy_ratio
        self.fill = fill

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target

        width, height = image.size
        max_dx = int(width * self.max_dx_ratio)
        max_dy = int(height * self.max_dy_ratio)

        if max_dx == 0 and max_dy == 0:
            return image, target

        dx = random.randint(-max_dx, max_dx) if max_dx > 0 else 0
        dy = random.randint(-max_dy, max_dy) if max_dy > 0 else 0

        if dx == 0 and dy == 0:
            return image, target

        image = _translate_pil_image(image, dx, dy, fill=self.fill)

        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            boxes[:, [0, 2]] += dx
            boxes[:, [1, 3]] += dy
            target["boxes"] = boxes
            target = _clip_boxes_and_remove_invalid(target, width, height)

        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            width, height = image.size
            image = F.hflip(image)

            boxes = target["boxes"].clone()
            if boxes.numel() > 0:
                xmin = width - boxes[:, 2]
                xmax = width - boxes[:, 0]
                boxes[:, 0] = xmin
                boxes[:, 2] = xmax
                target["boxes"] = boxes
                target = _clip_boxes_and_remove_invalid(target, width, height)

        return image, target



def _set_xml_text(parent, tag, value):
    elem = parent.find(tag)
    if elem is None:
        elem = ET.SubElement(parent, tag)
    elem.text = str(value)
    return elem


def _update_voc_xml_basic_info(root, new_filename, image_width, image_height, image_depth=3):
    """VOC XML의 filename/path/size 정보를 새 증강 이미지 기준으로 갱신"""
    _set_xml_text(root, "filename", new_filename)

    path_elem = root.find("path")
    if path_elem is not None:
        path_elem.text = os.path.join(TRAIN_IMG_DIR, new_filename)

    size = root.find("size")
    if size is None:
        size = ET.SubElement(root, "size")

    _set_xml_text(size, "width", int(image_width))
    _set_xml_text(size, "height", int(image_height))
    _set_xml_text(size, "depth", int(image_depth))


def _write_xml_tree(tree, save_xml_path):
    os.makedirs(os.path.dirname(save_xml_path), exist_ok=True)
    try:
        ET.indent(tree, space="    ", level=0)  # Python 3.9+
    except Exception:
        pass
    tree.write(save_xml_path, encoding="utf-8", xml_declaration=True)


def _copy_xml_for_augmented_image(src_xml_path, dst_xml_path, new_filename, image_width, image_height):
    """밝기/대비/CLAHE/노이즈처럼 bbox가 변하지 않는 증강용 XML 저장"""
    tree = ET.parse(src_xml_path)
    root = tree.getroot()
    _update_voc_xml_basic_info(root, new_filename, image_width, image_height)
    _write_xml_tree(tree, dst_xml_path)


def _save_shifted_xml_for_augmented_image(src_xml_path, dst_xml_path, new_filename, image_width, image_height, dx, dy):
    """좌우/상하 이동 증강용 XML 저장, bbox도 dx/dy만큼 이동하고 이미지 밖 좌표는 clip"""
    tree = ET.parse(src_xml_path)
    root = tree.getroot()
    _update_voc_xml_basic_info(root, new_filename, image_width, image_height)

    remove_objects = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            remove_objects.append(obj)
            continue

        try:
            xmin = float(bndbox.find("xmin").text) + dx
            ymin = float(bndbox.find("ymin").text) + dy
            xmax = float(bndbox.find("xmax").text) + dx
            ymax = float(bndbox.find("ymax").text) + dy
        except Exception:
            remove_objects.append(obj)
            continue

        xmin = max(0.0, min(float(image_width), xmin))
        xmax = max(0.0, min(float(image_width), xmax))
        ymin = max(0.0, min(float(image_height), ymin))
        ymax = max(0.0, min(float(image_height), ymax))

        # 이동 후 박스가 완전히 사라지면 해당 object 제거
        if xmax <= xmin or ymax <= ymin:
            remove_objects.append(obj)
            continue

        bndbox.find("xmin").text = str(int(round(xmin)))
        bndbox.find("ymin").text = str(int(round(ymin)))
        bndbox.find("xmax").text = str(int(round(xmax)))
        bndbox.find("ymax").text = str(int(round(ymax)))

    for obj in remove_objects:
        root.remove(obj)

    # 모든 bbox가 사라진 증강 이미지는 저장하지 않음
    if len(root.findall("object")) == 0:
        return False

    _write_xml_tree(tree, dst_xml_path)
    return True


def _apply_clahe_pil(image):
    """PIL RGB 이미지에 CLAHE 적용, opencv-python 필요"""
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(f"opencv-python 설치 필요: {e}")

    image_np = np.array(image)
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=AUG_CLAHE_CLIP_LIMIT,
        tileGridSize=AUG_CLAHE_TILE_GRID_SIZE,
    )
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    image_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(image_np)


def _apply_gaussian_noise_pil(image, rng):
    image_np = np.asarray(image).astype(np.float32) / 255.0
    noise = rng.normal(0.0, AUG_GAUSSIAN_NOISE_STD, image_np.shape).astype(np.float32)
    image_np = np.clip(image_np + noise, 0.0, 1.0)
    return Image.fromarray((image_np * 255.0).astype(np.uint8))


def _save_aug_image(image, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ext = os.path.splitext(save_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        image.save(save_path, quality=95)
    else:
        image.save(save_path)


def generate_train_augmented_files():
    """
    train 폴더에만 증강 이미지/XML을 실제 파일로 저장

    저장 위치:
      - annos/image/train/*_aug_*.jpg
      - annos/xml/train/*_aug_*.xml

    valid 폴더는 건드리지 않음
    이미 _aug_가 들어간 파일은 다시 증강하지 않아서 무한 증식을 방지
    """
    if not ENABLE_OFFLINE_AUGMENTATION:
        print("[AUG SAVE] ENABLE_OFFLINE_AUGMENTATION=False 이므로 증강 파일 저장을 건너뜁니다.")
        return

    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_ANN_DIR, exist_ok=True)

    train_img_paths = list_image_files(TRAIN_IMG_DIR)
    base_img_paths = [
        p for p in train_img_paths
        if OFFLINE_AUGMENT_MARK not in os.path.splitext(os.path.basename(p))[0]
    ]

    if not base_img_paths:
        print("[AUG SAVE] 증강할 원본 train 이미지가 없습니다.")
        return

    rng = np.random.default_rng(SPLIT_SEED)
    created = 0
    skipped = 0
    failed = 0
    clahe_warned = False

    for img_path in base_img_paths:
        xml_path = find_xml_for_image(img_path, TRAIN_ANN_DIR)
        if xml_path is None:
            print(f"[WARN] train XML 없음 -> 증강 제외: {os.path.basename(img_path)}")
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] 이미지 열기 실패 -> 증강 제외: {img_path} / {e}")
            failed += 1
            continue

        width, height = image.size
        stem = os.path.splitext(os.path.basename(img_path))[0]
        save_ext = OFFLINE_AUG_IMAGE_EXT if OFFLINE_AUG_IMAGE_EXT.startswith(".") else ".jpg"

        # 1) bbox 변화 없는 증강들
        no_bbox_augments = []
        no_bbox_augments.append(("brightness", ImageEnhance.Brightness(image).enhance(OFFLINE_BRIGHTNESS_FACTOR)))
        no_bbox_augments.append(("contrast", ImageEnhance.Contrast(image).enhance(OFFLINE_CONTRAST_FACTOR)))

        try:
            no_bbox_augments.append(("clahe", _apply_clahe_pil(image)))
        except Exception as e:
            if not clahe_warned:
                print(f"[WARN] CLAHE 증강 저장 실패. opencv-python 설치 필요 가능성 있음: {e}")
                clahe_warned = True

        no_bbox_augments.append(("gaussian_noise", _apply_gaussian_noise_pil(image, rng)))

        for aug_name, aug_img in no_bbox_augments:
            new_filename = f"{stem}{OFFLINE_AUGMENT_MARK}{aug_name}{save_ext}"
            dst_img_path = os.path.join(TRAIN_IMG_DIR, new_filename)
            dst_xml_path = os.path.join(TRAIN_ANN_DIR, os.path.splitext(new_filename)[0] + ".xml")

            if (not OFFLINE_AUGMENTATION_OVERWRITE) and os.path.exists(dst_img_path) and os.path.exists(dst_xml_path):
                skipped += 1
                continue

            try:
                _save_aug_image(aug_img, dst_img_path)
                _copy_xml_for_augmented_image(xml_path, dst_xml_path, new_filename, width, height)
                created += 1
            except Exception as e:
                print(f"[WARN] 증강 저장 실패: {new_filename} / {e}")
                failed += 1

        # 2) bbox 좌표가 같이 바뀌는 이동 증강들
        shift_x = max(1, int(width * OFFLINE_SHIFT_X_RATIO))
        shift_y = max(1, int(height * OFFLINE_SHIFT_Y_RATIO))
        shift_augments = [
            ("shift_left", -shift_x, 0),
            ("shift_right", shift_x, 0),
            ("shift_up", 0, -shift_y),
            ("shift_down", 0, shift_y),
        ]

        for aug_name, dx, dy in shift_augments:
            new_filename = f"{stem}{OFFLINE_AUGMENT_MARK}{aug_name}{save_ext}"
            dst_img_path = os.path.join(TRAIN_IMG_DIR, new_filename)
            dst_xml_path = os.path.join(TRAIN_ANN_DIR, os.path.splitext(new_filename)[0] + ".xml")

            if (not OFFLINE_AUGMENTATION_OVERWRITE) and os.path.exists(dst_img_path) and os.path.exists(dst_xml_path):
                skipped += 1
                continue

            try:
                aug_img = _translate_pil_image(image, dx, dy, fill=(0, 0, 0))
                xml_ok = _save_shifted_xml_for_augmented_image(
                    xml_path,
                    dst_xml_path,
                    new_filename,
                    width,
                    height,
                    dx,
                    dy,
                )
                if not xml_ok:
                    skipped += 1
                    continue

                _save_aug_image(aug_img, dst_img_path)
                created += 1
            except Exception as e:
                print(f"[WARN] 이동 증강 저장 실패: {new_filename} / {e}")
                failed += 1

    print("[AUG SAVE] train 전용 증강 파일 저장 완료")
    print(f" - 원본 train 이미지 수: {len(base_img_paths)}")
    print(f" - 새로 생성된 증강 이미지/XML 쌍: {created}")
    print(f" - 이미 존재/제외: {skipped}")
    print(f" - 실패: {failed}")
    print(f" - 저장 위치 image: {TRAIN_IMG_DIR}")
    print(f" - 저장 위치 xml  : {TRAIN_ANN_DIR}")


def get_transform(train=True):
    if train and USE_ONLINE_AUGMENTATION:
        # train에만 online 랜덤 증강 적용
        # 기본값은 False. 저장형 offline 증강과 중복 적용되는 것을 막기 위함
        transforms = [
            RandomBrightness(prob=AUG_BRIGHTNESS_PROB, delta=AUG_BRIGHTNESS_DELTA),
            RandomContrast(prob=AUG_CONTRAST_PROB, delta=AUG_CONTRAST_DELTA),
            RandomCLAHE(
                prob=AUG_CLAHE_PROB,
                clip_limit=AUG_CLAHE_CLIP_LIMIT,
                tile_grid_size=AUG_CLAHE_TILE_GRID_SIZE,
            ),
            RandomGaussianNoise(
                prob=AUG_GAUSSIAN_NOISE_PROB,
                std=AUG_GAUSSIAN_NOISE_STD,
            ),
            RandomTranslate(
                prob=AUG_TRANSLATE_PROB,
                max_dx_ratio=AUG_TRANSLATE_X_RATIO,
                max_dy_ratio=AUG_TRANSLATE_Y_RATIO,
            ),
            RandomHorizontalFlip(0.5),
            ToTensor(),
        ]
    else:
        # valid에는 증강 적용 금지
        # train도 offline 저장형 증강을 기본으로 쓰므로 여기서는 Tensor 변환만 수행
        transforms = [ToTensor()]

    return ComposeTransforms(transforms)


########################################
# 5. 모델
########################################
# Faster R-CNN ResNet50 FPN 모델 사용

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
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
    apply_gui_overrides_from_env()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    writer = SummaryWriter(log_dir=SCALAR_LOG_DIR)

    prepare_train_valid_folders()

    # PJS: train 폴더에만 증강 이미지/XML을 실제 파일로 저장
    # valid 폴더는 증강하지 않고 evaluate/inference 용도로만 사용
    generate_train_augmented_files()

    # PJS: 이제 랜덤 인메모리 split이 아니라 폴더 기준으로 train/valid를 분리
    # train: annos/image/train + annos/xml/train  -> 증강 적용 O
    # valid: annos/image/valid + annos/xml/valid  -> 증강 적용 X, evaluate/inference 때만 사용
    train_dataset = CustomVOCDataset(TRAIN_IMG_DIR, TRAIN_ANN_DIR, transforms=get_transform(train=True))
    valid_dataset = CustomVOCDataset(VALID_IMG_DIR, VALID_ANN_DIR, transforms=get_transform(train=False))

    if len(train_dataset) == 0:
        raise RuntimeError(
            "train 데이터가 없습니다. annos/image/train 및 annos/xml/train 폴더를 확인하세요."
        )

    print(f"[DATASET] train={len(train_dataset)}, valid={len(valid_dataset)}")
    print(f"[DATASET] train image dir = {TRAIN_IMG_DIR}")
    print(f"[DATASET] train xml dir   = {TRAIN_ANN_DIR}")
    print(f"[DATASET] valid image dir = {VALID_IMG_DIR}")
    print(f"[DATASET] valid xml dir   = {VALID_ANN_DIR}")

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    valid_loader = None
    if len(valid_dataset) > 0:
        # valid는 학습에는 사용하지 않고, evaluate/inference 단계에서만 순회
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
    else:
        print("[WARN] valid 데이터가 없습니다. valid inference/evaluation은 건너뜁니다.")

    num_classes = len(CLASS_NAMES)
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    os.makedirs("checkpoints", exist_ok=True)
    best_map = -1.0

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

        # valid는 학습 loss 계산에 사용하지 않음
        # evaluate() 내부에서 model.eval() 기반 inference/evaluation 용도로만 사용
        if valid_loader is not None:
            coco_evaluator = evaluate(model, valid_loader, device=device)

            mAP = float(coco_evaluator.coco_eval["bbox"].stats[0])
            mAP_50 = float(coco_evaluator.coco_eval["bbox"].stats[1])
            writer.add_scalar("Metrics/mAP", mAP, epoch)
            writer.add_scalar("Metrics/mAP_50", mAP_50, epoch)
        else:
            mAP = 0.0
            mAP_50 = 0.0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start
        writer.add_scalar("Time/epoch_time_sec", epoch_time, epoch)

        print(
            f"[Epoch {epoch}] train_loss: {train_loss:.4f}, "
            f"valid_inference_mAP: {mAP:.4f}, mAP_50: {mAP_50:.4f}, "
            f"epoch_time: {epoch_time:.2f} sec"
        )

        if valid_loader is not None and (epoch / NUM_EPOCHS) >= 0.7 and mAP > best_map:
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

    model.eval()
    try:
        scripted_model = torch.jit.script(model)
        pt_save_path = os.path.join("checkpoints", "fasterrcnn_best.pt")
        scripted_model.save(pt_save_path)
        print(f"{pt_save_path} 형식으로 모델 저장 완료.")
    except Exception as e:
        print(f"[ERROR] 모델을 .pt 형식으로 저장하는 중 오류 발생: {e}")

    writer.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows에서 DataLoader의 num_workers > 0 사용 시 필요
    main()
