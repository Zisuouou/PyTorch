# 1) 코드 전체 흐름
1. 설정값 정의
- 데이터 경로, 클래스 명, batch size, epoch 수, profiler 옵션 등
2. VOC XML 파싱
- XML에서 bounding box와 label 추출
3. Dataset 구성
- 이미지 + annotation을 모델 입력 형태로 반환
4. Transform 적용
- Tensor 변환, 좌우 반전 augmentation
5. 모델 생성
- fasterrcnn_resnet50_fpn 불러오고 클래스 수 맞게 head 교체
6. 학습 함수
- 한 step 학습
- 한 epoch 학습
- profiler 활성화 시 TensorBoard용 trace 저장
7. main()
- 데이터 split
- DataLoader 생성
- 학습 / 검증 / mAP 기록
- best model 저장
- 마지막에 .pt 저장

**즉, 실제 시행 시작점은 main()이고, 나머지는 main()이 호출하는 부품들**

# 2) 설정값 영역의 역할
### 주요 상수들
- DATA_ROOT, IMG_DIR, ANN_DIR
    - 데이터셋 위치
    - 이미지 폴더와 XML 폴더 경로를 정함

- CLASS_NAMES
    - 클래스 이름 목록
    - __background__ 는 Faster R-CNN에서 배경 클래스 역할

- NUM_EPOCHS, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, NUM_WORKERS
    - 학습 반복 횟수, 배치 크기, DataLoader worker 수 설정

- SCALAR_LOG_DIR
    - TensorBoard의 scalar 로그 저장 위치
    - ex: loss, mAP, epoch time

- PROFILER_LOG_DIR
    - TensorBoard profiler trace 저장 위치
    - Profiler 탭에서 읽는 데이터

- PROFILER_WAIT, PROFILER_WARMUP, PROFILER_ACTIVE, PROFILER_REPEAT
    - PyTorch 공식 profiler 스케줄용 설정값
    - 공식 문서 예제의 schedule(wait=1, warmup=1, active=3, repeat=1) 패턴과 같은 역할

- ENABLE_PROFILER
    - profiler를 켤지 말지 결정

- PROFILE_ONLY_FIRST_EPOCH
    - 첫 epoch만 profiler를 돌릴지 여부
    - profiler는 느려질 수 있기 때문에 전체 epoch에 다 켜지 않도록 한 안전장치

# 3) parse_voc_xml(xml_path, class_to_idx)
- 이 함수는 Pascal VOC 형식 XML 파일을 읽어서 객체 박스와 라벨을 추출
### 입력
- xml_path
    - XML 파일 경로

- class_to_idx
    - 클래스 이르을 숫자 index로 바꾸기 위한 dict
    - ex: {"__background__":0, "crack":1}

### 내부 동작
- 1. XML 파일을 파싱
- 2. < object > 태그를 하나씩 순회
- 3. 객체 이름(name)을 읽음
- 4. xmin, ymin, xmax, ymax 를 추출
- 5. 클래스명이 등록되지 않았으면 경고 출력 후 스킵
- 6. 잘못된 박스(xmax <= xmin 또는 ymax <= ymin)도 스킵
- 7. 최종적으로 boxes, labels를 tensor로 바꿔 변환

### 반환값
- boxes: shape [N, 4]
- labels: shape [N]

**Faster R-CNN은 학습 시 이미지뿐 아니라 각 이미지에 대응하는 객체 위치(box)와 클래시(label) 정보를 같이 받아야함. XML은 사람이 읽기 쉬운 annotation 형식이고, 모델은 tensor를 원해서 그 사이를 이어주는 변환 함수**

*요약 : XML annotation -> PyTorch detection target 정보로 바꿔주는 함수*

# 4) CustomVOCDataset(Dataset)
- 이 클래스는 PyTorch의 Dataset을 상속받은 커스텀 데이터셋 클래스
- 학습할 때 DataLoader가 이 클래스를 통해 데이터를 한 장씩 가져감

## 4-1) __init__(self, img_dir, ann_dir, transform=None)
### 역할
- 데이터셋 초기화 함수

### 하는 일
- 이미지 폴더 경로 저장
- annotation(XML)폴더 경로 저장
- transform 저장
- *.jpg 이미지 목록을 수집해서 self.img_paths 에 저장
- CLASS_NAMES 를 기반으로 class_to_idx 생성

**나중에 __getitem__()이 불릴 때 어떤 이미지 파일과 XML 파일을 읽어야 하는지 알아야해서 필요함**

## 4-2) __len__(self)
### 역할
- 데이터셋 크기 반환

### 반환값
- 총 이미지 개수

**DataLoader가 전체 데이터셋 길이를 알아야 배치를 만들 수 있음**

## 4-3) __getitem__(self, idx)
### 역할
- idx 번째 샘플 하나를 실제로 읽어서 (이미지, target) 형태로 반환

### 내부 동작
- 1. idx에 해당하는 이미지 경로를 가져옴
- 2. 파일 이름에서 확장자를 제거해서 대응되는 XML 파일 이름을 만듦
- 3. 이미지를 읽고 RGB로 변환
- 4. parse_voc_xml() 을 호출해 box, label 을 읽음
- 5. 박스가 없으면 빈 tensor로 처리
- 6. Faster R-CNN이 요구하는 target dict를 구성
    - boxes
    - labels
    - image_id
    - area
    - iscrowd
- 7. transform이 있으면 적용
- 8. 최종적으로 (img, target) 반환

### target 각 항목 의미
- boxes : 객체 위치
- labels : 객체 클래스
- image_id : 샘플 식별용 id
- area : 객체 크기
- iscrowd : crowd 여부 (COCO 스타일 평가 호환용)

** Faster R-CNN은 일반 분류 모델처럼 (image, label)이 아니라 (image, {boxes, labels, ...})형태를 요구함. 즉, 이 메서드는 모델이 바로 먹을 수 있는 형태의 샘플 생성기**

# 5. Transform 관련 클래스 / 함수
- 이 부분은 이미지와 bounding box를 함께 변환해주는 역할

## 5-1) ComposeTransforms
### 역할
- 여러 transform을 순서대로 적용하는 래퍼
- __init__(self, transforms)
    - transform 리스트를 저장

- __call__(self, image, target)
    - transform들을 순서대로 실행
    - 매번 (image, target)을 받아 다음 transform에 넘김

**object detection 에서는 이미지뿐 아니라 box 정보도 같이 바뀌어야 하므로, 단순 torchvision Compose 대신 (image, target) 쌍을 처리할 수 있어야해서 필요함**

## 5-2) ToTensor
### 역할
- PIL 이미지를 PyTorch tensor로 변환
- __call__(self, image, target)
    - F.to_tensor(image)실행
    - 이미지를 [C, H, W] 형태의 float tensor 로 바꿈
    - target 은 그대로 유지

**모델은 PIL 이미지가 아니라 tensor를 입력으로 받아서 필요**

## 5-3) RandomHorizontalFlip
### 역할
- 학습 데이터 augmentation 으로 좌우반전을 적용
- __init__(self, prob=0.5)
    - 반전 확률 저장

- __call__(self, image, target)
- 1. 랜덤값을 뽑아서 확률적으로 반전 여부 결정
- 2. 이미지를 좌우 반전
- 3. 박스 좌표도 같이 보정
    - 기존 xmin, xmax를 이미지 폭 기준으로 뒤집어서 새 좌표 계산

**object detection 에서는 이미지만 뒤집으면 안 되고, bounding box도 같이 뒤집어야 정답이 유지됨**

*데이터 다양성을 늘려서 일반화 성능을 높이는 augmentation 함수*

## 5-4) get_transform(train=True)
### 역할
- train/valid 상황에 맞는 transform 파이프라인을 구성

### 동작
- 항상 ToTensor() 포함
- train=True 면 RandomHorizontalFlip(0.5) 추가
- ComposeTranforms로 묶어서 반환

### 의미
- 학습 데이터 : Tensor 변환 + augmentation
- 검증 데이터 : Tensor 변환만

*이렇게 해야 validation 성능이 왜곡되지 않음*

# 6) get_model(num_classes)
### 역할
- Faster R-CNN 모델을 생성하고, 현재 클래스 개수에 맞게 classification head를 교체

### 내부 동작
- 1. torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")로 사전학습 모델 로드
- 2. 기존 box predictor의 입력 feature 크기(in_features)추출
- 3. FastRCNNPredictor로 새 predictor 생성
- 4. num_classes에 맞게 head 교체

**기본 pretrained Faster R-CNN은 COCO 클래스 수에 맞춰져 있음**

*사전학습 Faster R-CNN 을 내 데이터셋 클래스 수에 맞게 재구성하는 함수*

# 7) collate_fn(batch)
### 역할
- object detection용 DataLoader에서 배치를 묶을 때 사용하는 함수

- 일반 분류 문제는 이미지 크기와 label 형식이 일정해서 default collate가 잘 동작하지만, object detection은 샘플마다 객체 수가 다르므로 boxes 길이가 제각각임
- 그래서 기본 collate 대신: tuple(zip(*batch)) 형태로
    - images를 리스트로, targets를 리스트로 묶어야함

*가변 길이 target을 안전하게 batch로 묶기 위한 함수*

# 8) 학습 보조 함수들
## 8-1) move_to_device(images, targets, device)
### 역할
- 배치 데이터를 CPU에서 GPU(또는 지정 device)로 옮김

### 입력
- images: 이미지 tensor들의 리스트
- targets: dict들의 리스트
- device: cuda 또는 cpu

### 내부 동작
- 모든 이미지에 대해 img.to(device, non_blocking=True)
- targets 안의 값들 중 tensor인 것만 device로 이동
- tensor가 아닌 값은 그대로 둠

### 분리 이유
- 학습/검증 루프 안에 반복해서 들어감
- 함수로 분리하면
    - 코드 중복 제거
    - train/valid 모두 재사용 가능
    - 가독성 향상

### non_blocking=True 의미
- pin_memory=True와 함께 쓰면 CPU -> GPU 전송이 더 효율적임

## 8-2) run_train_step(model, optimizer, images, targets, device)
### 역할
- 배치 하나에 대한 학습 step 전체를 처리함

### 내부 동작
- 1. move_to_device()로 데이터를 device로 이동
- 2. model(images, targets) 호출
    - Faster R-CNN은 train 모드에서 loss dict를 반환
- 3. loss들을 합쳐 총 loss 계산
- 4. optimizer.zero_grad(set_to_none=True)
- 5. losses.backward()
- 6. optimizer.step()
- 7. loss_dict, losses 반환

**이 함수는 "한 step 학습"이라는 가장 작은 단위로, 이를 분리할 시 train_one_epoch() 안에서 읽기 쉬워짐**

### 반환값
- loss_dict: 예를 들면 classification loss, box regression loss 등 세부 손실
- losses: 전체 loss 합

## 8-3) train_one_epoch(...)
- 가장 중요한 핵심 함수

### 역할
- 한 epoch 동안 전체 train_loader를 돌면서 학습하고, 필요하면 profiler 를 켜서 TensorBoard용 trace도 저장

### 파라미터
- model
- optimizer
- data_loader
- device 
- epoch
- writer
- print_freq=1
- enable_profiler=False

### 내부 구조 핵심
- 이 함수 안에는 다시 run_loop(prof=None)라는 내부 함수가 있음

## 8-3-1 내부 함수 run_loop(prof=None)
- 실제 배치 반복 루프 

### 하는 일
- 1. DataLoader에서 (images, targets)를 받음
- 2. run_train_step() 호출
- 3. epoch loss 누적
- 4. global_step 계산
- 5. writer.add_scalar()로 세부 loss 기록
- 6. print 출력
- 7. profiler 가 있으면 prof.step() 호출

**profiler 를 켠 경우와 안 켠 경우 모두 같은 학습 루프를 써야 한느데, 코드를 두 번 쓰지 않기 위해 내부 함수로 묶음**

*즉, profiler 키면 run_loop(prof) 실행, 안 키면 run_loop() 실행*

## 8-3-2) profiler 활성화 분기
- if enable_profiler: 일 때만 profiler context를 엶 
### 내부 동작
- PROFILER_LOG_DIR 폴더 생성
- 활동 타입 설정
    - 기본 CPU
    - CUDA 사용 가능하면 CUDA 추가
- profiler(...) 실행

**activities**
- CPU, CUDA 어떤 연산을 측정할지 지정

**schedule(...)**
- 공식 문서 스타일 단계 제어
- wait: 기록 안 하는 대기 구간
- warmup: 워밍업 구간
- active: 실제 기록 구간
- repeat: 주기 반복 횟수

**on_trace_ready=tensorboard_trace_handler(PROFILER_LOG_DIR)**
- 핵심
- trace가 준비되면 TensorBoard 가 읽을 수 있는 형식으로 저장
- 그래서 Profile 탬에서 시각화할 수 있음

**record_shapes=True**
- 연산 입력 tensor shape 기록

**profile_memory=True**
- 메모리 사용량 기록

**with_stack=True**
- 어떤 코드 라인에서 연산이 발생했는지 추적
    - 그 후 run_loop(prof)를 실행

## 8-3-3) profiler 비활성화 분기
- profiler를 켜지 않으면 그냥 run_loop()만 실행
- 즉, 학습 동작 자체는 동일하고, profiler 정보 수집만 추가되는 구조

## 8-3-4) 반환값
- epoch_loss
    - 한 epoch 평균 train_loss

# 9) main()
- 이 함수는 전체 파이프라인을 실행하는 진입점(entry point)

## 9-1) device 설정
- device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### 역할
- GPU가 있으면 GPU 사용, 없으면 CPU 사용

## 9-2) SummaryWriter 생성
- writer = Summaryriter(log_dir=SCALAR_LOG_DIR)
### 역할
- TensorBoard Scalars 탭에서 볼 수 있는 값들을 기록
    - train loss
    - valid loss
    - mAP
    - mAP_50
    - epoch time
    - 세부 loss들

## 9-3) 데이터셋 분할
### 내부 동작
- 1. 전체 이미지 목록 수집
- 2. 랜덤 permutation 생성
- 3. 80%를 train, 20%를 valid로 분리

### 의미 
- 간단한 train/validation split 를 수행

## 9-4) train/valid 데이터셋 생성
- train_dataset = CustomVOCDataset(...)
- valid_dataset = CustomVOCDataset(...)

### 역할
- 학습용은 augmentation 포함
- 검증용은 augmentation 제외
- 그리고 img_paths 를 split 결과에 맞게 덮어써서 실제 train/valid 샘플만 쓰게 함

## 9-5) DataLoader 생성
**train_loader**
- shuffle=True
- batch_size = TRAIN_BATCH_SIZE

**valid_loader**
- shuffle=False
- batch_size = VALID_BATCH_SIZE

**pin_memory**
- CUDA가 있으면 켜짐
- GPU 전송 효율 향상 목적

## 9-6) 모델 생성 및 optimizer 설정
**get_model(num_classes)**
- Faster R-CNN 생성

**optimizer**
- SGD 사용
- lr, momentum, weight_decay 지정

## 9-7) 체크포인트 준비
- os.makedirs("checkpoints", exist_ok=True)
- best_map = 0.0

### 역할
- 최고 성능 모델 저장을 위한 준비

## 9-8) epoch 반복 학습
- for epoch in range(1, NUM_EPOCHS + 1):
- 여기서 매 epoch 마다 하는 일:

**(1)epoch 시간 측정 시작**
- GPU 연산이 끝난 뒤 정확한 시간 측정을 위해 torch.cuda.synchronize() 호출

**(2)profiler 사용 여부 결정**
- enable_profiler = ENABLE_PROFILER and (not PROFILE_ONLY_FIRST_EPOCH or epoch == 1)
- 의미:
    - profiler 전체 사용 여부가 켜져 있고
    - 첫 epoch만 프로파일링하도록 되어 있으면 epoch==1 일 때만 켬

**(3)train_one_epoch 실행**
- 학습 진행
- 필요 시 profiler trace 도 저장
- 평균 train loss 반환

**(4)train loss 기록**
- TensorBoard scalar로 기록

**(5)evaluate() 실행**
- 검증셋에서 COCO evaluator 기반 mAP 계산

**(6)mAP / mAP_50 기록**
- TensorBoard에 저장

**(7)validation loss 계산**
- Faster R-CNN은 loss 계산할 때 train() 모드여야 하므로 다시 model.train() 상태로 두고, 
- torch.no_grad() 안에서 valid set loss를 계산함

**(8)valid loss 기록**
- TensorBoard에 저장

**(9)epoch time 기록**
- 학습 시작~종료 시간 차이를 계산해서 TensorBoard에 기록

**(10)콘솔 출력**
- epoch별 성능 요약을 출력

**(11)최고 성능 모델 저장**
- 전체 epoch의 70% 이상 진행된 뒤
- 현재 mAP가 최고 기록보다 높으면
- fasterrcnn_best.pth로 저장

## 9-9) 마지막 모델 저장
- 학습 종료 후 현재 상태를 fasterrcnn_custom_last.pth로 저장
### 의미 
- 최고 성능 모델(best)과는 별개로
- 학습 마지막 상태(last)도 남겨둠

## 9-10) .pt 저장(TorchScript)
### 내부 동작
- 1. best_model_path 가 있으면 그 가중치를 로드
- 2. model.eval()로 전환
- 3. torch.jit.script(model) 수행
- 4. fasterrcnn_best.pt 로 저장

### 목적
- PyTorch 가중치 파일 .pth 외에도 배포/추론용으로 좀 더 독립적인 TorchScript .pt 파일을 만들기 위함

## 9-11) writer.close()
- TensorBoard writer를 정상 종료하여 로그를 안전하게 저장

# 10) 마지막 실행부
- if __name__ == "__main__":
    - main()
### 역할 
- 이 파일을 직접 실행했을 때만 main() 을 호출함

- 즉
    - 파일 단독 실행 -> 학습 시작
    - 다른 파일에서 import -> 자동 실행 안 됨

# 11) 함수들 관계를 한 번에 정리
- main()
    - CustomVOCDataset(...)
        - 내부에서 __getitem__()
            - parse_voc_xml()
            - get_transform() / transform 클래스들
    - get_model()
    - train_one_epoch()
        - 내부 run_loop()
            - run_train_step()
                - move_to_device()
    - evaluate()

**흐름: main-> train_one_epoch-> run_train_step-> move_to_device**

**데이터 쪽: Datasetgetitem -> parse_voc_xml + transforms**

# 12) 특히 profiler 관점에서 중요한 함수
**핵심1 : train_one_epoch(..., enable_profiler=True)**
- 이 함수가 profiler를 실제로 켜는 곳

**핵심2 : profile(...)**
- 여기서 CPU/CUDA/메모리/shape/stack 정보 수집 여부 설정

**핵심3 : tensorboard_trace_handler(PROFILER_LOG_DIR)**
- 이게 trace를 TensorBoard가 읽을 수 있는 로그 형식으로 저장
- 즉, Profile 탭에 보이게 만드는 핵심 포인트

# 13) 한 줄 요약
- parse_voc_xml() : XML annotation 읽기
- CustomVOCDataset.__getitem__() : 이미지와 target 만들기
- ComposeTransforms, ToTensor, RandomHorizontalFlip, get_transform() : 전처리와 augmentation
- get_model() : Faster R-CNN 생성 + head 교체
- collate_fn() : detection batch 묶기
- move_to_device() : GPU/CPU로 데이터 이동
- run_train_step() : 배치 1개 학습
- train_one_epoch() : epoch 전체 학습 + profiler 기록
- main() : 전체 학습/검증/저장 파이프라인 실행