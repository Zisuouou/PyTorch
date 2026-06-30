import os
import glob
import json
import warnings

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F

warnings.filterwarnings("ignore")


# ============================================================
# 1. 설정
#    - Torch_2_TensorBoard_Profiler.py에서 train/valid 분리 후
#      valid 폴더만 추론 대상으로 사용
# ============================================================

DATA_ROOT = r"C:\Users\SVT\Desktop\PyTorch_PJS\annos"

# 학습 때 쓴 CLASS_NAMES와 반드시 동일해야 함
CLASS_NAMES = [
    "__background__",
    "102embossed",
]

# 학습 스크립트에서 저장한 TorchScript .pt 경로
CKPT_PATH = r"C:\Users\SVT\Desktop\PyTorch_PJS\checkpoints\fasterrcnn_best.pt"

# 중요:
# train에서 학습한 모델로, valid 이미지만 검증/추론
# train 폴더와 증강 이미지는 inference에서 안 읽음
VALID_IMG_DIR = os.path.join(DATA_ROOT, "image", "valid")
TEST_IMG_DIR = VALID_IMG_DIR

# 결과 저장 폴더
OUT_DIR = r"C:\Users\SVT\Desktop\PyTorch_PJS\results\valid_inference"

# 이 값 이상인 box만 검출로 인정
SCORE_THRESH = 0.7

# 지원 이미지 확장자
IMAGE_EXTENSIONS = [
    "*.jpg", "*.JPG",
    "*.jpeg", "*.JPEG",
    "*.png", "*.PNG",
    "*.bmp", "*.BMP",
    "*.tif", "*.tiff", "*.TIF", "*.TIFF",
]


def apply_gui_overrides_from_env():
    """
    GUI/launcher에서 PYTORCH_GUI_CONFIG 환경변수로 전달한 설정을 반영

    사용 가능한 override 예:
    {
        "DATA_ROOT": ".../annos",
        "CLASS_NAMES": ["__background__", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "CKPT_PATH": ".../checkpoints/fasterrcnn_best.pt",
        "OUT_DIR": ".../results/valid_inference",
        "SCORE_THRESH": 0.7
    }

    주의:
    - IMG_DIR가 전달되어도 inference는 image/valid만 사용
    - TEST_IMG_DIR 또는 VALID_IMG_DIR를 직접 넘기면 그 경로를 사용하되,
      기본 원칙은 valid 폴더만 추론하는 것
    """
    global DATA_ROOT, CLASS_NAMES, CKPT_PATH
    global VALID_IMG_DIR, TEST_IMG_DIR, OUT_DIR, SCORE_THRESH

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

    CLASS_NAMES = overrides.get("CLASS_NAMES", CLASS_NAMES)
    if CLASS_NAMES and CLASS_NAMES[0] != "__background__":
        CLASS_NAMES = ["__background__"] + [
            c for c in CLASS_NAMES if c != "__background__"
        ]

    CKPT_PATH = overrides.get("CKPT_PATH", CKPT_PATH)
    OUT_DIR = overrides.get("OUT_DIR", OUT_DIR)
    SCORE_THRESH = float(overrides.get("SCORE_THRESH", SCORE_THRESH))

    # 기본은 DATA_ROOT/image/valid
    VALID_IMG_DIR = os.path.join(DATA_ROOT, "image", "valid")

    # 명시적으로 TEST_IMG_DIR 또는 VALID_IMG_DIR를 넘겼을 때만 사용
    TEST_IMG_DIR = overrides.get(
        "TEST_IMG_DIR",
        overrides.get("VALID_IMG_DIR", VALID_IMG_DIR)
    )

    print("[GUI CONFIG APPLIED - INFERENCE]")
    print(f" - DATA_ROOT = {DATA_ROOT}")
    print(f" - CLASS_NAMES = {CLASS_NAMES}")
    print(f" - CKPT_PATH = {CKPT_PATH}")
    print(f" - TEST_IMG_DIR = {TEST_IMG_DIR}")
    print(f" - OUT_DIR = {OUT_DIR}")
    print(f" - SCORE_THRESH = {SCORE_THRESH}")


# ============================================================
# 2. 모델 구조 정의
#    - .pth state_dict를 읽을 때만 사용
#    - .pt TorchScript는 torch.jit.load로 바로 로드
# ============================================================

def get_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            num_classes
        )
    )
    return model


def load_model(device):
    """
    .pt TorchScript 또는 .pth state_dict 모델을 로드합니다.

    반환값:
      - model
      - model_format: "pt" 또는 "pth"

    중요:
    convert_pth_to_pt.py로 변환한 .pt 모델은 FasterRCNNWrapper가 적용되어
    입력은 img_tensor 1개, 출력은 boxes, scores, labels 3개입니다.
    """
    assert os.path.isfile(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}"

    ext = os.path.splitext(CKPT_PATH)[1].lower()

    if ext == ".pt":
        try:
            model = torch.jit.load(CKPT_PATH, map_location=device)
            print(f"[LOADED TorchScript .pt] {CKPT_PATH}")
            model.to(device)
            model.eval()
            return model, "pt"
        except Exception as e:
            print(f"[ERROR] TorchScript .pt 로드 실패: {e}")
            raise

    if ext == ".pth":
        model = get_model(len(CLASS_NAMES))
        state_dict = torch.load(CKPT_PATH, map_location=device)

        # checkpoint dict 형태로 저장된 경우도 대응
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict)
        print(f"[LOADED state_dict .pth] {CKPT_PATH}")
        model.to(device)
        model.eval()
        return model, "pth"

    raise ValueError(f"지원하지 않는 체크포인트 확장자입니다: {CKPT_PATH}")


def _empty_detection_tensors():
    boxes = torch.empty((0, 4), dtype=torch.float32)
    scores = torch.empty((0,), dtype=torch.float32)
    labels = torch.empty((0,), dtype=torch.long)
    return boxes, scores, labels


def unpack_detection_results(results):
    """
    모델 출력 형태를 boxes, scores, labels 3개 Tensor로 통일합니다.

    지원 출력:
      1) convert_pth_to_pt.py 변환 .pt:
         results = (boxes, scores, labels)
      2) 일반 Faster R-CNN .pth:
         results = [{"boxes": ..., "scores": ..., "labels": ...}]
      3) TorchScript Faster R-CNN 일부 형태:
         results = (losses, detections)
    """
    if isinstance(results, tuple):
        # convert_pth_to_pt.py의 FasterRCNNWrapper 출력
        if len(results) == 3 and all(torch.is_tensor(x) for x in results):
            boxes, scores, labels = results
            return boxes.detach().cpu(), scores.detach().cpu(), labels.detach().cpu()

        # 일부 TorchScript Faster R-CNN 출력: (losses, detections)
        if len(results) >= 2:
            detections = results[1]
            if isinstance(detections, (list, tuple)) and len(detections) > 0:
                output = detections[0]
                if isinstance(output, dict):
                    boxes = output.get("boxes", torch.empty((0, 4)))
                    scores = output.get("scores", torch.empty((0,)))
                    labels = output.get("labels", torch.empty((0,), dtype=torch.long))
                    return boxes.detach().cpu(), scores.detach().cpu(), labels.detach().cpu()

    # 일반 Faster R-CNN 출력: list[dict]
    if isinstance(results, (list, tuple)) and len(results) > 0 and isinstance(results[0], dict):
        output = results[0]
        boxes = output.get("boxes", torch.empty((0, 4)))
        scores = output.get("scores", torch.empty((0,)))
        labels = output.get("labels", torch.empty((0,), dtype=torch.long))
        return boxes.detach().cpu(), scores.detach().cpu(), labels.detach().cpu()

    print(f"[WARN] 지원하지 않는 모델 출력 형식입니다: {type(results)}")
    return _empty_detection_tensors()


# ============================================================
# 3. 이미지 수집
# ============================================================

def collect_valid_images():
    """
    TEST_IMG_DIR 안의 이미지만 수집
    기본 TEST_IMG_DIR는 annos/image/valid 
    """
    if not os.path.isdir(TEST_IMG_DIR):
        raise FileNotFoundError(
            "[ERROR] valid 이미지 폴더가 없습니다.\n"
            f"        현재 경로: {TEST_IMG_DIR}\n"
            "        먼저 Torch_2_TensorBoard_Profiler.py에서 "
            "annos/image/train, annos/image/valid 구조를 만들어 주세요."
        )

    # 안전장치: 기본 경로가 아닌 경우 경고
    normalized = os.path.normpath(TEST_IMG_DIR).lower()
    if os.path.basename(normalized) != "valid":
        print(
            "[WARN] TEST_IMG_DIR의 마지막 폴더명이 'valid'가 아닙니다.\n"
            "       지수님 요청 기준으로는 valid 폴더만 추론하는 것이 맞습니다.\n"
            f"       현재 TEST_IMG_DIR = {TEST_IMG_DIR}"
        )

    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        img_paths.extend(glob.glob(os.path.join(TEST_IMG_DIR, ext)))

    img_paths = sorted(set(img_paths))

    # 혹시라도 증강 파일이 valid 폴더에 들어간 경우 제외
    img_paths = [
        p for p in img_paths
        if "_aug_" not in os.path.basename(p).lower()
    ]

    return img_paths


# ============================================================
# 4. 한 장 추론 + DETECTED / MISSED 분리 저장
# ============================================================

def run_inference_on_image(
    model,
    model_format: str,
    device,
    img_path: str,
    detected_dir: str,
    missed_dir: str
):
    img = Image.open(img_path).convert("RGB")
    img_tensor = F.to_tensor(img).to(device)

    model.eval()
    with torch.no_grad():
        if model_format == "pt":
            # convert_pth_to_pt.py로 변환한 .pt는 wrapper 구조라서
            # 입력: img_tensor 1개
            # 출력: boxes, scores, labels
            try:
                results = model(img_tensor)
            except Exception as e:
                # 혹시 Torch_Train.py에서 torch.jit.script(model) 방식으로 만든 .pt라면
                # 기존 Faster R-CNN 방식인 list 입력을 한 번 더 시도합니다.
                print(f"[WARN] .pt single tensor 입력 실패, list 입력으로 재시도: {e}")
                results = model([img_tensor])
        else:
            # .pth state_dict 모델은 일반 Faster R-CNN 방식
            results = model([img_tensor])

    boxes, scores, labels = unpack_detection_results(results)

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    detected = False
    kept_count = 0

    for box, label, score in zip(boxes, labels, scores):
        score_value = float(score)
        if score_value < SCORE_THRESH:
            continue

        detected = True
        kept_count += 1

        x1, y1, x2, y2 = box.tolist()

        li = int(label)
        cls_name = CLASS_NAMES[li] if 0 <= li < len(CLASS_NAMES) else f"label_{li}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        text = f"{cls_name} {score_value:.2f}"
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # 이미지 밖으로 텍스트가 나가지 않도록 보정
        text_y1 = max(0, y1 - text_h)
        text_y2 = max(text_h, y1)

        draw.rectangle([x1, text_y1, x1 + text_w, text_y2], fill="red")
        draw.text((x1, text_y1), text, fill="white", font=font)

    file_name = os.path.basename(img_path)
    out_path = os.path.join(detected_dir if detected else missed_dir, file_name)

    img.save(out_path)
    print(
        f"[SAVED] {out_path} "
        f"(detected={detected}, boxes_above_thresh={kept_count})"
    )

    return detected, kept_count


# ============================================================
# 5. 메인
# ============================================================

def main():
    apply_gui_overrides_from_env()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    detected_dir = os.path.join(OUT_DIR, "DETECTED")
    missed_dir = os.path.join(OUT_DIR, "MISSED")
    os.makedirs(detected_dir, exist_ok=True)
    os.makedirs(missed_dir, exist_ok=True)

    model, model_format = load_model(device)

    img_paths = collect_valid_images()

    print(f"[INFO] valid images only: {len(img_paths)}")
    print(f"[VALID IMG DIR] {TEST_IMG_DIR}")
    print(f"[OUT] DETECTED -> {detected_dir}")
    print(f"[OUT] MISSED   -> {missed_dir}")
    print(f"[THRESH] SCORE_THRESH = {SCORE_THRESH}")

    if not img_paths:
        print("[WARN] valid 폴더에 추론할 이미지가 없습니다.")
        return

    detected_images = 0
    missed_images = 0
    total_boxes = 0

    for img_path in img_paths:
        detected, kept_count = run_inference_on_image(
            model,
            model_format,
            device,
            img_path,
            detected_dir,
            missed_dir
        )
        if detected:
            detected_images += 1
        else:
            missed_images += 1
        total_boxes += kept_count

    print("\n[DONE]")
    print(f" - total valid images : {len(img_paths)}")
    print(f" - detected images    : {detected_images}")
    print(f" - missed images      : {missed_images}")
    print(f" - total kept boxes   : {total_boxes}")


if __name__ == "__main__":
    main()
