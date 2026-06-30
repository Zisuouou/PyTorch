# PyTorch Faster R-CNN Object Detection Pipeline

PyTorch 기반 Faster R-CNN 객체 검출 모델의 학습, 모델 변환, 추론 과정을 정리한 프로젝트입니다.

본 프로젝트는 VOC XML 라벨 데이터를 기반으로 객체 검출 모델을 학습하고, 학습된 `.pth` 모델을 TorchScript `.pt` 형식으로 변환한 뒤, 검증 이미지에 대해 추론을 수행하는 구조입니다.

## 주요 기능

- VOC XML 형식의 라벨 데이터 로드
- train / valid 데이터 분리
- train 데이터 전용 이미지 증강
- Faster R-CNN ResNet50 FPN 모델 학습
- TensorBoard 로그 기록
- valid 데이터 기반 mAP 평가
- `.pth` checkpoint 저장
- `.pth` 모델을 TorchScript `.pt` 모델로 변환
- valid 이미지 추론 결과를 DETECTED / MISSED 폴더로 저장
- 
## 파일 구성

```text
PyTorch_PJS/
├── Torch_Train.py
├── convert_pth_to_pt.py
├── Torch_Inference.py
├── annos/
│   ├── image/
│   │   ├── train/
│   │   └── valid/
│   └── xml/
│       ├── train/
│       └── valid/
├── checkpoints/
└── results/


## 파일 설명
### Torch_Train.py
- Faster R-CNN 모델 학습용 파일입니다.
- 주요 기능은 다음과 같습니다.
  - 이미지/XML 데이터셋 로드
  - train / valid 폴더 자동 분리
  - train 데이터에만 증강 적용
  - Faster R-CNN ResNet50 FPN 학습
  - TensorBoard 로그 저장
  - valid 데이터로 mAP 평가
  - best / last .pth checkpoint 저장

### convert_pth_to_pt.py
- 학습된 .pth모델을 TorchScript .pt 형식으로 변환하는 파일입니다.
- .pth 모델을 불러온 뒤 Faster R-CNN 추론 결과 중, boxes, scores, labels만 반환하도록 wrapper를 적용하고, torch.jit.trace()를 이용해 .pt파일로 저장합니다.

### Torch_Inference.py
- 학습 또는 변환된 모델을 이용해 valid 이미지에 대해 추론을 수행하는 파일입니다.
- 추론 결과 score가 기준값 이상이면 DETECTED 폴더에 저장하고, 검출되지 않은 이미지는 MISSED 폴더에 저장

## 실행 순서
- 1. 모델 학습
- 2. .pth 모델을 .pt로 변환
- 3. valid 이미지 추론

## 데이터 구조
- 데이터는 다음 구조를 기준으로 사용합니다.

```text
annos/
├── image/
│   ├── train/
│   └── valid/
└── xml/
    ├── train/
    └── valid/
  
- 이미지 파일과 XML 파일은 같은 이름을 가져야 합니다.

## 주의 사항
- CLASS_NAMES는 학습, 변환, 추론 파일에서 동일하게 유지해야 합니다.
- .pth, .pt, runs/, checkpoints/, results/ 폴더는 용량이 클 수 있어 깃허브 업로드 시 제외하는 것을 권장합니다.
- TorchScript .pt 모델은 변환 방식에 따라 추론 출력 구조가 달라질 수 있으므로, 추론 코드와 변환 코드의 입출력 형식을 맞춰야 합니다.

## 개발 환경
- Python
- PyTorch
- Torchvision
- TensorBoard
- PIL
- OpenCV 선택 사항
