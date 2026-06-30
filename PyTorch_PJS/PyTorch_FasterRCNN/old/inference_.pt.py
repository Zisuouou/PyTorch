import os
import glob

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

import warnings
warnings.filterwarnings("ignore")   

# -----------------
# 1. 설정 (학습 때랑 반드시 동일 해야함)
# -----------------

# 클래스 이름 (학습 때 쓴 것과 완전히 동일 해야함)
CLASS_NAMES = [
    "__background__",  # 0
    "crack",   # 1
]


# 체크포인트 경로 (학습 스크립트에서 저장한 위치)
CKPT_PATH = r"C:\Users\SVT\Desktop\PyTorch_PJS\checkpoints\fasterrcnn_best.pt"

# 예측 돌릴 이미지 폴더
TEST_IMG_DIR = r"C:\Users\SVT\Desktop\PyTorch_PJS\annos\image"

# 결과 저장 폴더 (하위에 DETECTED / MISSED 자동 생성)
OUT_DIR = r"C:\Users\SVT\Desktop\PyTorch_PJS\results"

# 신뢰도 threshold (이 값 이상인 것만 "검출"로 인정 + 그림)
SCORE_THRESH = 0.5


# -----------------
# 2. 모델 구조 정의 (학습 때와 동일)
# -----------------
def get_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        num_classes
    )
    return model


# -----------------
# 3. 한 장 추론 + "검출/미검출" 폴더 분리 저장
# -----------------
def run_inference_on_image(model, device, img_path: str, detected_dir: str, missed_dir: str):
    # 1) 이미지 로드
    img = Image.open(img_path).convert("RGB")

    # 2) Tensor로 변환 (학습 때 ToTensor()와 동일)
    import torchvision.transforms.functional as F
    img_tensor = F.to_tensor(img).to(device)  # [C,H,W], 0~1

    # 3) 모델 추론
    model.eval()
    with torch.no_grad():
        # TorchScript 모델은 (losses, detections) 튜플 반환 _26.03.04
        results = model([img_tensor])
        # 일반 모델은 detections 리스트만 주지만, Scripting 모델은 튜플의 두 번째가 결과임
        if isinstance(results, tuple):
            output = results[1][0]  # (losses, detections) 중 detections의 첫 번째 이미지 결과
        else:
            output = results[0]  # 일반 모델은 그냥 리스트이므로 첫 번째 이미지 결과

    boxes = output.get("boxes", torch.empty((0, 4))).detach().cpu()
    labels = output.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu()
    scores = output.get("scores", torch.empty((0,))).detach().cpu()

    # 4) PIL 이미지에 그리기
    draw = ImageDraw.Draw(img)

    # 폰트 설정 (없어도 동작, 없으면 기본 폰트)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    detected = False  # 핵심: score_thresh 이상 박스가 하나라도 있으면 True

    for box, label, score in zip(boxes, labels, scores):
        if float(score) < SCORE_THRESH:
            continue

        detected = True

        x1, y1, x2, y2 = box.tolist()

        # label이 범위를 벗어나는 예외 방지
        li = int(label)
        cls_name = CLASS_NAMES[li] if 0 <= li < len(CLASS_NAMES) else f"label_{li}"

        # 박스 그리기
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # 라벨 + 점수 텍스트
        text = f"{cls_name} {float(score):.2f}"

        bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill="red")
        draw.text((x1, y1 - text_h), text, fill="white", font=font)

    # 5) 저장 경로 분기
    file_name = os.path.basename(img_path)
    if detected:
        out_path = os.path.join(detected_dir, file_name)
    else:
        out_path = os.path.join(missed_dir, file_name)

    img.save(out_path)
    print(f"[SAVED] {out_path}  (detected={detected})")


# -----------------
# 4. 메인 : 폴더 돌면서 전부 추론
# -----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    # 결과 폴더 생성
    detected_dir = os.path.join(OUT_DIR, "DETECTED")
    missed_dir = os.path.join(OUT_DIR, "MISSED")
    os.makedirs(detected_dir, exist_ok=True)
    os.makedirs(missed_dir, exist_ok=True)

    num_classes = len(CLASS_NAMES)
    model = get_model(num_classes)
    model.to(device)

    # .pt (TorchScript) 모델 로드
    assert os.path.isfile(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}"
    # .pt 파일은 torch.load가 아닌 torch.jit.load로 로드해야 함
    try:
        model = torch.jit.load(CKPT_PATH, map_location=device)
        print(f"[LOADED TorchScript] {CKPT_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load TorchScript model: {e}")
        return
    
    model.to(device)
    model.eval()

    # 이미지 목록 수집
    img_paths = []
    for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG", "*.bmp", "*.BMP", "*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
        img_paths.extend(glob.glob(os.path.join(TEST_IMG_DIR, ext)))
    img_paths = sorted(img_paths)

    print(f"[INFO] found {len(img_paths)} images in {TEST_IMG_DIR}")
    print(f"[OUT] DETECTED -> {detected_dir}")
    print(f"[OUT] MISSED   -> {missed_dir}")
    print(f"[THRESH] SCORE_THRESH = {SCORE_THRESH}")

    # 각 이미지를 순회하며 추론
    for img_path in img_paths:
        run_inference_on_image(model, device, img_path, detected_dir, missed_dir)

    print("[DONE]")


if __name__ == "__main__":
    main()
