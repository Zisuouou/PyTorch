import os
import glob

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

from torch.utils.tensorboard import SummaryWriter



# 1. 설정 (학습 때랑 반드시 동일 해야함)

# 클래스 이름 (학습 때 쓴 것과 완전히 동일 해야함)
CLASS_NAMES = [
    "__background__",  # 0
    "MilPinContamin"
]
# 체크포인트 경로 (학습 스크립트에서 저장한 위치)
CKPT_PATH = r"C:\Users\SVT\Desktop\PyTorch\checkpoints\fasterrcnn_custom.pth"

# 예측 돌릴 이미지 폴더
# 학습에 썼던 폴더 사용 or 나중에 새로운 테스트 이미지 폴더 (기존 TF사용했던거처럼 경로 바꿔도 됨)
TEST_IMG_DIR = r"C:\Users\SVT\Desktop\PyTorch\annos\image" 

# 결과 저장 폴더
OUT_DIR = r"C:\Users\SVT\Desktop\PyTorch\results"
os.makedirs(OUT_DIR, exist_ok=True)

# 신뢰도 threshold (한계점) (이 값 이상인 것만 그림)
SCORE_THRESH = 0.5

# 2. 모델 구조 정의 (학습 때와 동일)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None    # 추론 시 : pretrain 불필요, weight 로드할거라 None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = \
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            num_classes
        )
    return model

# 3. 한 장의 이미지에 대한 추론 & 박스 그리기

def run_inference_on_image(model, device, img_path, out_path):
    # 1) 이미지 ㅗㄹ드
    img = Image.open(img_path).convert("RGB")

    # 2) Tensor로 변환 (학습 때 ToTensor()와 동일)
    import torchvision.transforms.functional as F
    img_tensor = F.to_tensor(img).to(device)    # [C,H,W], 0~1

    # 3) 모델 추론
    model.eval()
    with torch.no_grad():
        outputs = model([img_tensor]) # list[dict]
    output = outputs[0]

    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()
    scores = output["scores"].cpu()

    # 4) PIL 이미지에 그리기
    draw = ImageDraw.Draw(img)

    # 폰트 설정 (없어도 동작, 없으면 기본 폰트)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        
    for box, label, score in zip(boxes, labels, scores):
        if score < SCORE_THRESH:
            continue

        x1, y1, x2, y2 = box.tolist()
        cls_name = CLASS_NAMES[label]

        # 박스 그리기
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # 라벨 + 점수 텍스트
        text = f"{cls_name} {score:.2f}"

        bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # text_size = draw.textsize(text, font=font)
        # text_bg = [x1, y1 - text_size[1], x1 + text_size[0]. y1]
        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill="red")
        draw.text((x1, y1 - text_h), text, fill="white", font=font)

    # 5) 저장
    img.save(out_path)
    print(f"[SAVED] {out_path}")


# 4. 메인 : 폴더 돌면서 전부 추론
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    num_classes = len(CLASS_NAMES)
    model = get_model(num_classes)
    model.to(device)

    # weight 로드
    assert os.path.isfile(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}"
    state_dict = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print(f"[LOADED] {CKPT_PATH}")

    # 이미지 목록 수집
    img_paths = []
    for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
        img_paths.extend(glob.glob(os.path.join(TEST_IMG_DIR, ext)))
    img_paths = sorted(img_paths)

    print(f"[INFO] found {len(img_paths)} images in {TEST_IMG_DIR}")

    # 각 이미지를 순회하며 추론
    for img_path in img_paths:
        file_name = os.path.basename(img_path)
        out_path = os.path.join(OUT_DIR, file_name)
        run_inference_on_image(model, device, img_path, out_path)

if __name__ == "__main__":
    main()