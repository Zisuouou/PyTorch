# PyTorch 기반 Faster R-CNN 객체 검출 학습 및 추론 파이프라인 추가
- 주요내용:
  - **Torch_Train.py** : VOC XML 라벨 기반 Faster R-CNN 학습, train/valid 분리, train 전용 증강, TensorBoard 로그 기록, mAP평가, .pth checkpoint 저장
  - **convert_pth_to_pt.py** : 학습된 .pth모델을 TorchScript .pt형식으로 변환
  - **Torch_Inference.py** : 학습/변환된 모델을 사용하여 valid 이미지 추론 수행 후 DETECTED / MISSED 결과 저장

- 이 구조는 학습 -> 모델 변환 -> 추론까지의 전체 흐름을 분리하여 관리하기 위한 목적 
