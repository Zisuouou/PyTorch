# PyTorch_PJS

PyTorch 학습,추론 및 TensorBoard Profiler 테스트 코드 정리 폴더

### PyTorch detection 공통 유틸/reference 파일 
1. coco_eval.py
- COCO기준 평가 담당
- mAP 같은 detection 성능 계산할 때 사용

2. coco_utils.py
- 데이터셋 COCO 평가 형식에 맞게 변환하는 보조 코드
- annotation 정리, COCO API 연결용 유틸 역할

3. engine.py
- 학습 / 평가 루프 핵심
- train_one_epoch , evaluate 같은 함수가 보통 여기에 해당

4. transforms.py
- 이미지와 bbox 같이 변환하는 전처리/증강 코드
- detection 은 박스도 같이 움직여야 해서 일반 classification transform 보다 중요함
  
5. utils.py
- 로그 출력, metric 기록, 분산처리 보조, 공통 함수 모음
- 말 그대로 전반적인 보조 유틸

**즉, 정리하면 이 5개 파일은 PyTorch detection 을 돌릴 때 자주 쓰이는 공통 지원 파일 5종 세트**
*다만, 이 5개는 주인공 파일이라기 보다 학습/평가를 받쳐주는 기반 파일*
*detection 운영용 공통 보조 파일이 맞고, PyTorch detection 핵심 유틸 묶음으로 보면 됨*

# Torch_v1.py
- PyTorch 학습/추론 전용 GUI

# PyTorch 기반 Faster R-CNN 객체 검출 학습 및 추론 파이프라인 추가
- 주요내용:
  - Torch_Train.py : VOC XML 라벨 기반 Faster R-CNN 학습, train/valid 분리, train 전용 증강, TensorBoard 로그 기록, mAP평가, .pth checkpoint 저장
  - convert_pth_to_pt.py : 학습된 .pth모델을 TorchScript .pt형식으로 변환
  - Torch_Inference.py : 학습/변환된 모델을 사용하여 valid 이미지 추론 수행 후 DETECTED / MISSED 결과 저장

- 이 구조는 학습 -> 모델 변환 -> 추론까지의 전체 흐름을 분리하여 관리하기 위한 목적 


