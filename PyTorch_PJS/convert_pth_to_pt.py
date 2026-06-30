"""
.pth 모델을 .pt (TorchScript) 형식으로 변환하는 스크립트
"""

"""
주요 기능:
1. FasterRCNNWrapper: boxes, scores, labels만 반환하는 간단 추론 모델
2. JIT Tracing: 동적 모델을 정적 TorchScript로 변환
3. 에러 처리: 파일 존재 여부 및 변환 과정 검증
4. Warmup 기능: CUDA 최적화를 위한 선택적 warmup
5. 파일 크기 출력: 변환 후 모델 크기 확인
"""

# cmd 에서 python convert_pth_to_pt.py --pth checkpoints\fasterrcnn_best.pth --output checkpoints\fasterrcnn_best.pt --device cuda 입력 후 실행

import os
import torch
import argparse


class FasterRCNNWrapper(torch.nn.Module):
    """추론용 모델 래퍼 - boxes, scores, labels만 반환"""
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        with torch.no_grad():
            # x : [3, H, W]
            det = self.model([x])[0]
            boxes = det["boxes"].contiguous()
            scores = det["scores"].contiguous()
            labels = det["labels"].contiguous()
            return boxes, scores, labels


def infer_num_classes(pth_model_path, device="cpu"):
    """체크포인트의 box_predictor 클래스 수를 추론합니다."""
    state_dict = torch.load(pth_model_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    for key in [
        "roi_heads.box_predictor.cls_score.weight",
        "roi_heads.box_predictor.cls_score.bias",
        "roi_heads.box_predictor.bbox_pred.weight",
        "roi_heads.box_predictor.bbox_pred.bias",
    ]:
        if key in state_dict:
            shape = state_dict[key].shape
            if key.endswith("cls_score.weight"):
                return shape[0]
            if key.endswith("cls_score.bias"):
                return shape[0]
            if key.endswith("bbox_pred.weight"):
                return shape[0] // 4
            if key.endswith("bbox_pred.bias"):
                return shape[0] // 4

    raise RuntimeError(
        f"[ERROR] Unable to infer num_classes from checkpoint: missing expected predictor keys"
    )


def convert_pth_to_pt(
    pth_model_path,
    pt_save_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    input_height=2048,
    input_width=4096,
    num_warmup=0,
    num_classes=None
):
    """
    .pth 모델을 .pt (TorchScript) 형식으로 변환
    
    Args:
        pth_model_path (str): .pth 파일 경로
        pt_save_path (str): 저장할 .pt 파일 경로
        device (str): 사용할 디바이스 ('cuda' 또는 'cpu')
        input_height (int): 더미 입력의 높이
        input_width (int): 더미 입력의 너비
        num_warmup (int): CUDA warmup 반복 횟수
    
    Returns:
        str: 저장된 .pt 파일 경로
    """
    
    print(f"[INFO] Converting {pth_model_path} to .pt format...")
    print(f"[INFO] Device: {device}")
    
    # 1. 디렉토리 생성
    os.makedirs(os.path.dirname(pt_save_path) or ".", exist_ok=True)
    
    # 2. 파일 존재 여부 확인
    if not os.path.exists(pth_model_path):
        raise FileNotFoundError(f"[ERROR] Model file not found: {pth_model_path}")
    
    # 3. 모델 로드 (faster rcnn 가정)
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        if num_classes is None:
            num_classes = infer_num_classes(pth_model_path, device)
            print(f"[INFO] Inferred num_classes={num_classes} from checkpoint")

        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(pth_model_path, map_location=device))
        print(f"[SUCCESS] Model weights loaded from {pth_model_path}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load model: {e}")
    
    # 4. 모델을 device로 이동 및 eval 모드
    model.to(device)
    model.eval()
    
    # 5. Wrapper로 감싸기
    wrapper = FasterRCNNWrapper(model)
    wrapper.to(device)
    wrapper.eval()
    
    # 6. 더미 입력 생성
    dummy_input = torch.zeros(
        (3, input_height, input_width),
        dtype=torch.float32,
        device=device
    )
    
    # 7. Warmup (선택사항)
    if num_warmup > 0:
        print("[INFO] Performing warmup...")
        for i in range(num_warmup):
            with torch.no_grad():
                _ = wrapper(dummy_input)
    
    # 8. CUDA 동기화
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 9. JIT Tracing
    print("[INFO] Tracing model...")
    try:
        traced_model = torch.jit.trace(
            wrapper,
            dummy_input,
            strict=False,
            check_trace=False
        )
        print("[SUCCESS] Model traced successfully")
    except Exception as e:
        raise RuntimeError(f"[ERROR] JIT tracing failed: {e}")
    
    # 10. .pt 파일로 저장
    try:
        traced_model.save(pt_save_path)
        print(f"[SUCCESS] Model saved to {pt_save_path}")
        print(f"[INFO] File size: {os.path.getsize(pt_save_path) / (1024*1024):.2f} MB")
        return pt_save_path
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to save model: {e}")


def main():
    parser = argparse.ArgumentParser(description=".pth 모델을 .pt 형식으로 변환")
    parser.add_argument(
        "--pth",
        required=True,
        help=".pth 모델 파일 경로"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="저장할 .pt 파일 경로"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="사용할 디바이스 (cuda 또는 cpu)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=2048,
        help="입력 이미지 높이"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=4096,
        help="입력 이미지 너비"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup 반복 횟수"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="모델 클래스 수 (background 포함). 지정하지 않으면 체크포인트에서 추론 시도"
    )
    
    args = parser.parse_args()
    
    try:
        convert_pth_to_pt(
            pth_model_path=args.pth,
            pt_save_path=args.output,
            device=args.device,
            input_height=args.height,
            input_width=args.width,
            num_warmup=args.warmup,
            num_classes=args.num_classes
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
