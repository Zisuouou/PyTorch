| TensorBoard 의 Profile 탭에서 보이는 것들 -> on_trace_ready=tensorboard_trace_handler(PROFILER_LOG_DIR) |
| :-- |
| 위 코드는 파일을 profiler 결과를 TensorBoard가 읽을 수 있게 저장하는 코드 |
| runs/profiler 아래에 trace 가 저장되고, TensorBoard는 그것을 Profile 탭에서 보여줌 | 
| 반면 writer.add_scalar(...)로 저장한 loss, mAP 같은 값들은 Scalar 탭에서 보임 |

# 1. TensorBoard에서 크게 두 군데 확인
## scalar 탭 
- 보이는 것:
    - Loss/train
    - Loss/valid
    - Metrics/mAP
    - Netrics/mAP_50
    - Loss_Detail/...
    - Time/epoch_time_sec

- 즉, 학습 성능 그래프를 보는 곳

##  Profile 탭
- profiler 가 수집한 걸 보는 곳
- 보이는 것:
    - 어느 연산이 시간을 많이 먹는지
    - CPU에서 오래 걸리는지
    - CUDA(GPU)에서 오래 걸리는지
    - 데이터 로딩 / 전송 / forward / backward 중 어디가 병목인지
    - 어떤 op가 메모리를 많이 쓰는지
    - 한 step 안에서 시간 흐름이 어떻게 되는지

- 즉, 모델이 어디서 느린가 / 무거운가 를 보는 곳

# 2. Profile 탭에서 보게 되는 대표 화면들
- TensorBoard Profile 탭은 버전에 따라 이름이 조금씩 다름, 대체로 아래 항목들을 봄

## A. Overview / Summary 
- 전체 profiler 결과의 요약 화면
- 여기서 보는 것:
    - step time이 대략 얼마나 걸리는지
    - device가 얼마나 바쁘게 일하는지
    - input pipeline이 병목인지
    - CPU/GPU 중 어디가 더 놀고 있는지
    - 어느 구간 최적화가 필요한지에 대한 큰 그림

**Torch_2_TensorBoard_Profiler.py 에서는 한 train step이 대략 아래 순서로 돌아감**

1. DataLoader 가 배치 준비
2. move_to_device()로 GPU 전송
3. model(images, targets)로 forward + loss 계산
4. loss.backward()로 backward 
5. optimizer.step()으로 업데이트

- Overview는 이 전체 step을 크게 훑으면서, 이 학습 step이 어디서 시간을 많이 쓰고 있나를 처음 보는 화면이라고 생각하면 됨
- ex)
    - GPU utilization이 낮다
        - -> GPU가 놀고 있다
        - -> DataLoader / CPU preprocessing / I/O 병목 가능성 큼
    - input pipeline bottleneck 경고가 있다
        - -> 데이터를 준비하는 쪽이 느리다는 뜻
        - -> num_workers, 저장장치 속도, XML 파싱, 이미지 로딩 확인 필요
    - step time 이 너무 길다
        - -> forward/backward 자체가 무겁거나 batch가 큼

## B. Trace Viewer
- 시간축 기준으로 step 에서 어떤 일이 언제 일어났는지 줄 단위로 보여줌
- 보이는 줄 :
    - CPU thread
    - CUDA stream
    - DataLoader 관련 이벤트
    - PyTorch op들
    - kernel 실행 구간
    - memory copy(HtoD 등)

**즉, 학습 1 step을 타임라인으로 펼쳐놓은 화면**

- step의 내부적 흐름
    - move_to_device(images, targets, device)
    - model(images, targets)
    - optimizer.zero_grad(...)
    - losses.backward()
    - optimizer.step()

- Trace Viewer에서는 이런 흐름이 시간 순서대로 보임
- ex:
    - CPU 에서 XML 파싱은 이미 Dataset.__getitem__() 쪽에서 처리됨
    - DataLoader worker가 batch를 준비함
    - CPU에서 tensor 준비 후 GPU 전송
    - GPU에서 convolution / roi heads / box predictor 관련 kernel 실행
    - backward kernel 실행
    - optimizer step 실행

- 중요 포인트
1. 빈 구간이 많은지 : CPU 나 GPU 라인에 빈칸이 길면, 그 시간 동안 장치가 놀고 있는 것
- GPU가 비어 있으면
    - 데이터 공급이 느리거나
    - CPU 처리 대기 중이거나
    - 작은 batch 때문에 GPU 활용이 낮을 수 있음

2. HtoD memcpy가 길게 보이나 Host to Device 복사가 길면
- CPU -> GPU 데이터 전송이 병목일 수 있음
- pin_memory, non_blocking=True 를 잘 쓰고 있는지 확인 포인트

3. backward가 유난히 긴지 
- Faster R-CNN은 detection 모델이라 backward도 꽤 무거움
- 특히 RPN, ROI heads, classification/regression loss 쪽이 복잡해서 backward가 크게 나올 수 있음

## C. Operator / Kernel View
- 어떤 연산(op)이 시간을 많이 먹는지를 표 형태로 보여줌
- 여기서 보는 것
    - op 이름
    - 호출 횟수
    - CPU 시간
    - CUDA(GPU) 시간
    - self time / total time
    - 메모리 사용량
- 즉, 누가 시간을 제일 많이 잡아먹었나 순위표

### 자주 보이는 것들
- PyTorch detection 모델에서는 아래의 것들이 많이 보임
    - aten::conv2d
    - aten::convolution
    - aten::matmul
    - aten::add
    - aten::relu
    - NMS 관련 연산
    - ROI Align 관련 연산
    - box / proposal 관련 tensor ops

**해석 방법**
- self cpu time : 그 연산 자체가 CPU 에서 직접 쓴 시간
- cpu total time : 그 연산이 호출한 하위 연산 포함 전체 CPU 시간
- cuda time : GPU 쪽 실행 시간

- 예시 해석 
    - aten::conv2d가 크다
        - -> backbone convolution이 주요 시간 소모원
    - box / roi 관련 연산이 크다
        - -> detection head 쪽 비용이 큰 것
    - CPU op 시간이 유난히 크다
        - -> GPU 보다 CPU 쪽 로직, 후처리, proposal 생성, 박스 정리 쪽 병목 가능

## D. GPU Kernel View
- GPU에서 실제로 돌아간 kerbel 관점으로 보는 화면 
    - 어떤 CUDA kernel이 실행됐는지
    - 각 kernel 실행 시간이 얼마나 되는지
    - kernel 호출 횟수
    - 어떤 연산이 GPU에서 무거운지

- 즉, PyTorch op 보다 더 밑단에서, GPU가 실제 뭘 하고 있었는지 보는 화면

| 이럴 때 유용함 |
| :-- |
| GPU는 분명 바쁜데, 어떤 연산이 무거운지 더 자세히 보고 싶을 때 |
| convolution kernel이 오래 걸리는지 |
| 작은 kernel이 너무 많이 쪼개져서 비효율적인지 |
| mixed precision, batch size 변경 전후 비교할 때 |

## 5. Memory View / Memory Profile
- 메모리 관련 정보 볼 수 있음

- 여기서 보는 것
    - 어떤 op가 메모리를 많이 할당하는지
    - 어느 시점에 메모리가 급증하는지
    - CPU 메모리 / GPU 메모리 사용 흐름
    - 불필요한 tensor 생성이 많은지

- 난 Faster R-CNN은 detection 모델이라서
    - backbone feature map
    - RPN proposal
    - ROI feature
    - classification / bbox regression intermediate tensor 등이 많아 메모리 사용량이 큼

- 여기서 볼 포인트
    - 특정 step 에서 메모리가 급격히 튀는가
    - 어떤 연산이 큰 메모리를 차지하는가
    - batch size를 늘릴 여유가 있는가
    - OOM 직전 패턴이 보이는가 

## F. Input Pipeline Analyzer 비슷한 화면
- TensorBoard profiler 는 종종 입력 파이프라인 병목도 분석해줌

- 여기서 보는 것
    - DataLoader가 제때 배치를 공급하는지
    - CPU preprocessing이 느린지
    - 디스크 I/O가 병목인지
    - GPU가 데이터를 기다리는지

- 내 데이터셋은 매 샘플마다
    - jpg 로딩
    - XML 파싱
    - box 생성
    - transform 적용 함

**즉, classification 보다 데이터 준비 비용이 더 큼**

- 그래서 이런 경우가 보일 수 있음
    - worker 수가 부족하면 GPU 가 데이터 기다림
    - XML 파싱이 많으면 CPU 쪽 시간이 늘어남
    - 이미지가 많고 저장장치가 느리면 I/O 병목
    - num_workers=4 가 적절한지 확인 필요

*이런 화면에서 "input bound" 처럼 보이면 모델보다 데이터 준비쪽을 먼저 최적화 해야함*

# 3. 내 코드 기준 Profile 탭에 특히 나타날 만한 것들
- Torch_2_TensorBoard_Profiler.py는 일반 RenNet 분류와 달리 Faster R-CNN detection 이므로 좀 더 복잡함

## 특징 1. step 시간이 분류 모델보다 길고 들쭉날쭉할 수 있음
- 이유:
    - 이미지마다 객체 개수가 다름
    - propoal 수가 달라짐
    - ROI 처리량이 샘플마다 다름
- 그래서 같은 batch size 라도 step time이 일정하지 않을 수 있음

## 특징 2. CPU 비중이 의외로 클 수 있음
- Detection 모델은 pure conv만 있는게 아니라
    - 박스 처리
    - proposal filtering
    - NMS
    - ROI 관련 처리
        - 같은 CPU/텐서 조작이 섞일 수 있음

- 그래서 Profile 탭에서 GPU만 보는 것 보다 CPU time 도 같이 봐야함

## 특징 3. DataLoader 병목이 더 잘 보일 수 있음
- 난 XML 파싱이 들어가서, 단순 이미지 분류용 dataset 보다 입력 준비가 더 무거울 수 있음
- 즉
    - GPU utilization 낮음
    - step 사이사이 빈 시간 많음
    - input pipeline 경고

*이런 패턴이 보이면 모델 문제가 아니라 데이터 준비 문제일 수 있음*

## 특징 4. 내 코드에서 Profile 탭에 로그가 남는 정확한 시점
- 내 파일에선 train_one_epoch() 안에서 profiler를 이렇게 킴
- enabel_profiler=True 일 때만 profiler context 진입
- 그리고 batch 마다 prof.step() 호출

- 또 스케줄이:
    - wait = 1
    - warmup = 1
    - active = 3
    - repeat = 1

- 뜻 : 첫 몇 step은 다 저장되는 게 아님
    - 1 step: 대기
    - 1 step: 워밍업
    - 다음 3 step: 실제 기록

- 즉, 전체 epoch 일부 step 만 Profile 탭용 trace로 저장됨 
    - 학습 전체를 다 기록하면 너무 무겁기 때문임

# 5. Profile 탭에서 좋다/나쁘다 보는 방법
## 좋은 패턴
- GPU가 비교적 끊김 없이 계속 일함
- step 사이 빈 구간이 짧음
- DataLoader 대기 시간이 짧음
- CPU와 GPU가 적절히 겹쳐서 동작함
- 특정 op 하나가 비정상적으로 압도적이지 않음

## 나쁜 패턴
- GPU가 자주 놀고 있음
- step 시작 전에 CPU 준비 시간이 너무 김
- HtoD 복사가 길게 반복됨
- DataLoader 가 계속 병목
- 메모리 사용이 급격히 튀거나 불안정
- 특정 op 또는 kernel 하나가 비정상적으로 오래 걸럼

# 6. Profile 탭에서 문제를 보면 손을 댈 곳
## 경우 1. GPU utilization 낮음 / 빈 시간 많음
- 의심할 곳
    - DataLoader 느림
    - XML 파싱 부담
    - 저장장치 I/O
    - batch size 너무 작음

- 조정 포인트
    - NUM_WORKERS 조정
    - SSD 사용 여부 확인
    - annotation 파싱 캐싱 고려
    - batch size 조정

## 경우 2. HtoD 전송이 길다
- 의심할 곳
    - CPU -> GPU 복사 병목

- 조정 포인트   
    - pin_memory=True 유지
    - non_blocking=True 유지
    - 이미지 크기 / batch size 확인

## 경우 3. conv / backbone 쪽 CUDA 시간이 너무 큼
- 의심할 곳
    - 모델 자체가 무거움
    - 이미지 크기가 큼
    - batch size가 큼

- 조정 포인트
    - input resize
    - batch size
    - 더 가벼운 backbone 검토

## 경우 4. ROI / box / NMS 쪽이 큼
- 의심할 곳
    - detetion head 후처리 비용
    - proposal 수 과다

- 조정 포인트
    - RPN proposal 관련 설정
    - detection threshold / proposal 수 제한
    - anchor / image size 최적화

## 경우 5. memory peak 가 큼
- 의심할 곳
    - batch size 과다
    - 큰 입력 이미지
    - detection intermediate tensor 가 큼

- 조정 포인트
    - batch size 축소
    - mixed precision 검토
    - image resize

# 7. TensorBoard에서 보면 좋은 순서
## 1단계 : Scalars 탭
- loss 가 정상적으로 줄어드는지
- mAP 가 올라가는지 먼저 확인

- 이는 학습이 되고 있는지 보는 단계

## 2단계 : Profile 탭 Overview
- step time
- device utilization
- input bottleneck 여부 확인

- 어디가 느린지 보는 단계 

## 3단계 : Trace Viewer
- step 안에서 CPU/GPU가 어떻게 움직이는지 확인
- 빈 시간 / memcpy / forward / backward 흐름 확인

- 이는 시간축으로 실제 흐름 보기

## 4단계 : Operator / Kernel View
- 어떤 op가 시간을 가장 많이 쓰는지 확인

## 5단계 : Memory View
- 어떤 연산이 메모리를 많이 쓰는지 확인
    - OOM / batch size / 메모리 최적화 판단

# 8. Profile 탭을 볼 때 꼭 기억할 것
### 첫째
- Profiler 탭은 성능 분석용이라 loss나 mAP처럼 좋아졌다/나빠졌다를 보는곳이 아님

### 둘째
- Scalar 탭과 같이 봐야함
    - Scalar : 모델 성능
    - Profile : 학습 속도/병목

### 셋째
- 지금 코드 설정상 보통 epoch 일부 step만 기록 됨
- 따라서 Profile 탭에 아주 긴 전체 학습 기록이 아닌, 대표 step 몇 개가 보이는 게 정상

### 넷째
- Faster R-CNN은 classification보다 trace가 훨씬 복잡하게 보일 수 있음  
    - GPU가 놀고 있는지
    - DataLoader 가 병목인지
    - 어떤 op가 제일 큰지 확인을 하는것이 맞음

# 9. 제일 실전적인 한 줄 해석
- TensorBoard Profile 탭은 결국
    - Overview : 전체 결과
    - Trace Viewer : step 타임라인 CCTV
    - Operator View : 느린 연산 순위표
    - Kernel View : GPU 내부 작업 순위표
    - Memory View : 메모리 사용 추적기
    - Input Pipeline 관련 화면 : 데이터 로딩 병목 탐지기

# 10. 핵심 요약
- Torch_2_TensorBoard_Profiler.py를 실행하면:
    - run/custom_voc : Scalar 탭에서 loss / mAP / epoch time 확인
    - run/profiler : Profile 탭에서 CPU/GPU 시간, op별 병목, 메모리, step 타임라인 확인

- 즉, 학습성능(Scalar) + 학습병목(Profile)을 동시에 보게 만든 버전