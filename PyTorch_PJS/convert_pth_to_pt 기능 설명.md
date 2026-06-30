**이 파일은 PyTorch Faster R-CNN 모델의 .pth가중치 파일을 ,pt TorchScript 파일로 변환하는 스크립트**

**즉, 학습 결과로 저장된 모델 가중치 .pth를 추론/배포하기 쉬운 .pt형태로 저장하는 용도**

*학습된 Faster R-CNN .pth 모델을 불러와서 -> 추론용 wrapper로 감싼 뒤 -> torch.jit.trace()로 TorchScript 변환 -> .pt 파일로 저장하는 코드*


# 주요 흐름
## 1. .pth 파일 존재 확인

## 2. Faster R-CNN 모델 구조 생성 

- Faster R-CNN ResNet50 FPN 구조를 새로 만들고, 학습한 클래스 수에 맞게 마지막 분류기 교체


## 3. 클래스 수 자동 추론

- --num-classes 를 따로 안 주면 .pth 내부의 predictor weight shapel를 보고 클래스 수를 자동 추론

- 예를 들어:
    - roi_heads.box_predictor.cls_score.weight

- 이 weight의 첫 번째 shape 값이 num_classes가 됨
- 주의할 점은 여기서 말하는 num_classes는 background 포함 클래스 수 

- 예:
    - 불량 클래스 1개 -> num_classes = 2
    - 불량 클래스 3개 -> num_classes = 4


## 4. 학습된 가중치 로드

- .pth에 저장된 가중치를 모델 구조에 넣음
    - 즉, .pth 안에 순수 state_dict가 들어있다고 가정


## 5. 추론 결과를 단순화하는 Wrapper 적용

- 원래 Faster R-CNN 출력은 dictionary 형태
- 근데 TorchScript 변환이나 추론 코드에서 다루기 쉽게 하기 위해 이 wrapper 가 결과를 아래 3개만 변환하게 만듦

- 즉, 최종 .pt 모델을 실행하면 결과가 
    - boxes, scores, labels = model(image) 이렇게 나옴


## 6. 더미 이미지로 JIT Tracing

- 기본 입력 크기는:
    - 3 x 2048 x 4096

- 이 더미 이미지를 모델에 한 번 흘려서 TorchScript 로 변환


## 7. .pt 파일 저장

- 최종적으로 .pt 파일을 저장하고 파일 크기도 출력


### 실행 예시
- python convert_pth_to_pt.py ^
-   --pth D:/AI_SVT_Training_mk/output/model_state_dict.pth ^
-   --output D:/AI_SVT_Training_mk/output/model.pt ^
-   --device cuda ^
-   --height 2048 ^
-   --width 4096 ^
-   --num-classes 5

    - --num-classes 를 생략하면 .pth 에서 자동 추론 시도


### 이 파일이 필요한 상황

- 이 코드는 이런 상황에서 씀
    - 학습 후 결과물이 .pth로 저장됐는데, 추론 프로그램에서는 .pt TorchScript 모델을 쓰고 싶을 때 필요

- 예를들어 내 흐름에선:
    - PyTorch 학습 완료
    - ↓
    - model_state_dict.pth 저장
    - ↓
    - convert_pth_to_pt.py 실행
    - ↓
    - model.pt 생성
    - ↓
    - inference.py 또는 GUI 추론에서 model.pt 사용


## 주의 점
- model.load_state_dict(torch.load(pth_model_path, map_location=device))
    - 여기가 가장 주의
    - .pth가 순수 state_dict일 때만 잘 맞음

- 그러나, .pth 가 이런 구조면

- {
    - "epoch": ...,
    - "state_dict": model.state_dict(),
    - "optimizer": ...
- }

- load_state_dict()에서 오류 날 수 있음 

- 또, torchvision 0.12.0 환경이면 weights = None 인자를 지원하지 않을 수 있음


## 결론
**이 파일의 용도는 PyTorch Faster R-CNN 학습 결과 .pth를 추론/배포용 .pt TorchScript 모델로 변환하는 변환기**


