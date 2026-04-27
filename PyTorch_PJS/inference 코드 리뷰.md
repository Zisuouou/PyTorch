- inference_.pt.py : 학습이 끝난 Faster R-CNN TorchScript 모델(.pt).를 불러와서 폴더 안 이미지 전체를 추론하고 결과를
    - 검출된 이미지 -> DETECTED 폴더
    - 검출 안 된 이미지 -> MISSED 폴더

- 즉, 이 파일은 학습 코드가 아니라 추론/검사 실행용 코드
- 학습 때 저장한 fasterrcnn_best.pt를 이용해서 실제 이미지들을 검사하는 역할

# 전체 흐름
1. 클래스명, 모델 경로, 이미지 폴더, 결과 폴더, threshold 설정
2. 학습 때와 같은 Faster R-CNN 구조 정의
3. .pt 모델 로드
4. 테스트 이미지 폴더 안 파일 목록 수집
5. 이미지 한 장씩 추론
6. score threshold 이상 박스가 있으면 DETECTED, 없으면 MISSED 저장

# 1. 상단 import / 설정 부분
## import 역할
- os, glob : 파일 경로 처리, 이미지 목록 수집
- torch, torchvision : 모델 로드 및 추론
- PIL.Image, ImageDraw, ImageFont : 이미지 열기, 박스 그리기, 텍스트 표시
- warnings.filterwarnings("ignore") : 경고 메시지 숨김

## 설정값들 
**CLASS_NAMES**

    - 이건 학습 때 사용한 클래스와 반드시 같아야 함
    - 왜냐면 모델이 숫자 label로 예측한 결과를 사람이 읽을 수 있는 이름으로 바꾸는 기준이기 때문

**CKPT_PATH**

    - 추론에 사용할 TorchScript 모델 파일 경로
    - .pth가 아닌 .pt 임
        - 즉, 이 코드는 torch.jit.load() 방식으로 불러오는 TorchScript 배포 모델용 코드

**TEST_IMG_DIR**

    - 테스트할 이미지가 들어있는 폴더

**OUT_DIR**

    - 결과 저장 루트 폴더

**SCORE_THRESH = 0.5**

    - 이 값 이상인 예측만 검출로 인정함
    - 즉, score가 0.49면 박스를 그리지 않음

# 2. get_model(num_classes)
- 이 함수는 학습 때와 같은 Faster R-CNN 구조를 만드는 함수

## 역할
- fasterrcnn_resnet50_fpn(weights=None)생성
- box predictor를 현재 클래스 수에 맞게 교체

## 필요 이유
- 이 파일은 .pt를 torch.jit.load()로 불러오니까, 엄밀히 말하면 TorchScript 로드 후엔 이미 완성된 모델 객체가 들어와서 get_model()의 의미가 약해짐

- 즉, 이 코드에서 get_model()은 
    - 초기 골격 준비용
    - 또는 .pth 방식과 코드 형태를 맞추기 위한 흔적

- 실제로는 아래에서 곧바로 model = torch.jit.load(...)로 덮어서 최종 추론에 쓰이는건 get_model()로 만든 모델이 아니라 로드한 TorchScript 모델임

# 3. run_inference_in_image(...)
- 이 함수가 파일의 핵심
- 이미지 한 장을 추론하고, 박스를 그린 뒤, DETECTED 또는 MISSED 폴더로 저장하는 함수

## 3-1) 이미지 로드
- img = Image.open(img_path).convert("RGB")
- 이미지를 열고 RGB로 변환
- 흑백이나 팔레트 이미지가 들어와도 모델 입력을 일정하게 맞추기 위해 RGB로 통일

## 3-2) Tensor 변환
- img_tensor = F.to_tensor(img).to(device)
- PIL 이미지를 PyTorch tensor로 변환함
- 학습코드에서 ToTensor() 했던 것과 같은 역할이며,
- shape은 [C,H,W], 값 범위는 0~1이 됨

## 3-3) 모델 추론
- with torch.no_grad():
    - results = model([img_tensor])
- gradient 계산 없이 추론만 수행
- 추론이니까 no_grad()를 써서 메모리와 연산 낭비를 줄임

### 중요 포인트
- 일반 모델은 detections 리스트 반환
- TorchScript 모델은 (losses, detections)튜플일 수 있음
- 즉, TorchScript 저장 방식 차이 때문에 결과 형식이 달라질 수 있어서 대응한 코드

## 3-4) 결과 
- boxes = output.get("boxes", ...)
- labels = output.get("labels", ...)
- scores = output.get("scores", ...)

- 모델 출력에서 필요한 예측 결과를 꺼내서 CPU 로 옮김
- CPU 인 이유 : 이후 PIL.ImageDraw로 박스를 그릴 건데 PIL은 GPU tensor를 직접 못 쓰기 때문

## 3-5) 이미지에 박스/문자 그리기
- draw = ImageDraw.Draw(img)
- 원본 이미지 위에 직접 표시를 그릴 준비를 함
- 폰트는 arial.ttf를 시도하고, 없으면 기본 폰트로 대체
- 즉, 폰트 파일이 없어도 되는 코드는 죽지 않게 생성

## 3-6) 검출 여부 판단
- detected = False

- 처음엔 검출 안 됐다고 놓고 시작함
- 반복문 안에서 threshold 이상 박스가 하나라도 나오면 True 로 바뀜

## 3-7) 각 예측 박스 처리
- for box, label, score in zip(boxes, labels, scores):

- 예측 결과를 하나씩 보면서 처리

**threshold 필터**

    - if float(score) < SCORE_THRESH:
        - continue
    - score가 기준보다 낮으면 무시
    - 즉, 최종 그림과 폴더 분리는 threshold 이상 결과만 기준임

**클래스명 변환**

    - lr = int(label)
    - cls_name = CLASS_NAEMS[li] if 0 <= li < len(CLASS_NAMES) else f"label_{li}"

    - 예측 label 숫자를 사람이 읽을 수 있는 이름으로 바꾸는 부분
    - 범위를 벗어나도 죽지 않게 예외 방어를 넣어 뒀음

**박스 그리기**

    - draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    
    - 검출 박스를 빨간색으로 그림

**텍스트 그리기**
    
    - text = f"{cls_name} {float(score):.2f}"

    - 그 다음 빨간 배경 박스를 만들고, 흰 글씨로 텍스트를 씀
    - 그래서 결과 이미지를 보면 클래스명 + confidence score가 같이 보임

## 3-8) DETECTED / MISSED 저장 분기
- if detected:
    - out_path = os.path.join(detected_dir, file_name)
- else:
    - out_path = os.path.join(missed_dir, file_name)

- threshold 이상 검출이 하나라도 있으면 DETECTED, 없으면 MISSED 로 저장

**의미**

    - 이 코드는 mAP 계산용이 아니라 실무 검사 자동 분류용 느낌
    - 즉, 
        - 불량이 하나라도 검출됨 -> DETECTED
        - 아무것도 못찾음 -> MISSED
    
    - 이렇게 바로 폴더 분류를 해줘서 결과 검토가 쉬워짐

# 4) main()
- 이 함수는 전체 추론 파이프라인을 실행하는 메인 함수

## 4-2) 결과 폴더 생성
- detected_dir = os.path.join(OUT_DIR, "DETECTED")
- missed_dir = os.path.join(OUT_DIR, "MISSED")
- os.makedirs(..., exist_ok=True)

- 결과 저장용 하위 폴더를 미리 만들어둠
- 없으면 생성, 있으면 그냥 사용

## 4-3) 모델 준비
- num_classes = len(CLASS_NAMES)
- model = get_model(num_classes)
- model.to(device)

- 초기 모델 구조를 만듦
- 그런데 아래에서 TorchScript 모델을 다시 로드해서 model에 덮어쓰기 때문에, 이 부분은 실제로 없어도 되는 중복 준비 코드 성격이 좀 있음

## 4-4) .pt TorchScript 모델 로드
- model = torch.jit.load(CKPT_PATH, map_location=device)

- 핵심은 .pt는 torch.load()가 아니라 torch.jit.load()로 읽어야 한다는 점이고

**에러처리**

    - 로드 실패 시 에러 메시지 출력 후 종료

## 4-5) 추론 모드 설정
- model.to(device)
- model.eval()

- TorchScript 모델을 지정 device로 올리고 evaluation 모드로 바꿈

## 4-6) 이미지 목록 수집
- 이 부분은 확장자 여러개 지원
- for ext in [*.jpg", "*.JPG", "*.jpeg", ...]:
    - img_paths.extend(...)

- 즉, 
    - jpg, jpeg, png, bmp, tif, tiff 등 다 수집해서 추론 대상으로 삼음

## 4-7) 정보 출력
- 몇 장 찾았는지, 결과 저장 위치가 어디인지, threshold가 얼마인지 콘솔에 출력

## 4-8) 전체 이미지 순회 추론
- for img_path in img_paths:
    - run_inference_on_image(...)

- 이미지 한 장씩 돌면서 추론 + 저장 수행

## 4-9) 종료 메시지
- 마지막에 [DONE] 출력

# 5) 마지막 실행부
- if __name__ == "__main__":
    - main()

- 이 파일을 직접 실행했을 때만 main() 실행

# 6) 이 코드의 핵심 목적
- 이 파일은 평가 지표 계산용이라기 보다 실무형 검사 결과 분리용 추론 스트립트임
- 즉 목적이:
    - 모델 불러오기
    - 이미지 폴더 전체 검사
    - 결과 이미지 저장
    - 검출 여부 기준으로 폴더 분리 

- 사용자 입장에서는 결과 폴더만 보면
    - 어떤 이미지가 불량 검출됐는지
    - 어떤 이미지는 아무것도 못 찾았는지
- 바로 구분 가능함

# 7) 이 코드의 장점

**1. TorchScript 대응**
    
    - .pt를 torch.jit.load()로 읽도록 되어 있어서 배포형 추론 코드에 맞음

**2. 결과 형식 차이 대응**

    - TorchScript 결과가 tuple 일 수 있는 점을 고려해서 분기 처리함

**3.여러 이미지 확장 지원**

    - 실사용 편함

**4. DETECTED / MISSED 자동 분리**

    - 현장 검수하기 좋음

**5. label index 범위 방어**

    - 잘못된 label 값이 와도 죽지 않게 처리


# 8) 아쉬운 ,  주의할 점

**1. get_model()은 현재 구조상 거의 의미가 약함**

    - 어차피 바로 아래에서 torch.jit.load()로 덮어씀
    - 즉, TorchScript 전용이면 이 부분은 생략 가능성이 큼

**2.SCORE_THRESH 하나로만 검출/미검출 판단**

    - 현재는 박스 하나라도 threshold 이상이면 DETECTED
    - 클래스별 threshold, NMS 추가 조정, top-k 제한 같은건 없음

**3. GT와 비교하는 평가는 아님**

    - 이 코드는 mAP 계산용이 아니라 그냥 추론 결과 저장용
    - 즉, 
        - 정답과 비교해서 맞았는지
        - false positive / false negative 가 몇 개인지 이런 평가는 하지 않음

**4. 폰트가 환경에 따라 다를 수 있음**

    - arial.ttf가 없으면 기본 폰트로 떨어짐

# 9) 한줄씩 요약
| CLASS_NAMES | CKPT_PATH | TEST_IMG_DIR | OUT_DIR | SCORE_THRESH | get_model() | run_inference_on_image() | main() |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 라벨 이름표 | 불러올 .pt 모델 위치 | 검사할 이미지 폴더 | 결과 저장 위치 | 몇 점 이상이면 검출로 볼지 기준 | Faster R-CNN 구조 정의 | 이미지 한 장 추론하고 박스 그린 뒤 DETECTED/MISSED로 저장 | 전체 폴더를 돌면서 모델 로드부터 추론까지 실행 | 

# 10) 결론
- inference_.pt.py는 학습 완료된 fasterrcnn_best.pt TorchScript 모델을 사용해서 이미지 폴더 전체를 검사하고, 결과 이미지를 저장하면서 검출 여부에 따라 폴더를 자동 분리하는 추론 스크립트



