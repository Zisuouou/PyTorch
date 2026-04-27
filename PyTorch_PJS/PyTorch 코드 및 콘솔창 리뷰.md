# train+best_mAP_.pt+profiler.py

## PyTorch를 사용하여 Faster R-CNN 객체 탐지 모델을 훈련하는 코드이며
## 훈련 흐름은 다음과 같다

1. 데이터 준비 : VOC 형식에 XML 어노테이션과 이미지를 로드하는 CustomVOCDataset 클래스를 정의하고, ToTensor 및 RandomHorizontalFlip 변환을 적용한 DataLoader 를 생성한다. 데이터셋을 훈련(80%)과 검증(20%)으로 분할한다.

2. 모델 설정 : torchvision 의 pre-trained Faster R-CNN ResNet50 FPN 모델을 로드하고, 클래스 수에 맞게 ROI 헤드를 교체한다. 모델을 GPU/CPU 디바이스로 이동한다.

3. 옵티마이저 및 로깅 : SGD 옵티마이저를 설정하고, TensorBoard Summary/Writer를 초기화한다.

4. 훈련 루프(에포크 반복):
    - 각 에포크에서 train_one_epoch 함수를 호출하여 배치별로:
        - 데이터를 디바이스로이동
        - 모델 forward pass로 손실 계산 (loss_dict: classification, bbox_regression 등)
        - 역전파 및 옵티마이저 스텝으로 가중치 업데이트
        - PyTorch Profiler로 성능 프로파일링(CPU/CUDA 시간, 메모리추적 )
    - 검증 단계에서 evaluate 함수로 mAP(IoU 0.50:0.95 및 0.50)를 계산하고, 검증 손실을 측정한다.
    - TensorBoard에 손실, mAP, 시간 등을 로깅한다.
    - 에포크 70% 이상 진행 시 최고 mAP 모델을 체크포인트로 저장한다.

5. 모델 저장 : 훈련 종료 후 마지막 모델과 최고 mAP 모델을 .pth 형식으로 저장하고, 최고 모델을 TorchScript(.pt) 형식으로 변환하여 저장한다.

전체적으로 10 에포크동안 반복하며, 배치크기 2(훈련)/1(검증), num_workers=4로 병렬처리함

## 훈련 완료 시 생성되는 폴더들
### 1. checkpoints 폴더
- fasterrcnn_best.pth : 훈련 중 최고 mAP를 기록한 에포크의 모델 가중치. 추론 시 최적 성능을 위해 사용
- fasterrcnn_custom_last.pth : 훈련 종료 시점의 최종 모델 가중치. 추가 훈련이나 비교용
- fasterrcnn_best.pt : 최고 성능 모델을 TorchScript 형식으로 변환한 파일. 배포 및 추론에 최적화된 형태로, Python 의 환경에서도 사용 가능

### 2. profiler_logs 폴더
- trace_step_*json : PyTorch Profiler가 생성한 성능 추적 파일. 각 스텝의 CPU/GPU 시간, 메모리 사용, 연산 그래프 등을 기록. Chrome Trace Viewer나 Perfetto로 분석하여 병목 지점(ex: 데이터 전송, 계산)을 식별하고 최적화에 활용

### 3. runs 폴더
- custom_voc/events.out.tfevents.*.0: TensorBoard 이벤트 파일. 훈련/검증 손실,mAP, 에포크 시간 등의 메트릭을 기록. TensorBoard로 시각화하여 학습 진행 상황을 모니터링하고, 하이퍼파라미터 튜닝에 사용

# Console 
## 1. 훈련 스텝별 손실 출력 ex:([Epoch 8] Step 1/20 loss: 0.2257 등)
- 의미 : 각 에포크의 배치(step)별 총 손실(loss)을 실시간으로 출력함. 손실은 모델의 예측 오차를 나타내며, 낮을수록 성능이 좋음

- 세부 설명:
    - Step 1/20 : 20개의 배치 중 첫 번째 배치의 손실
    - 값 예시 : 0.2257 → 이 배치의 총 손실 (classification + bbox_regression등 합산)
    - train_one_epoch 함수의 print_freq=1 에 따라 매 스텝마다 출력됨. 

## 2. PyTorch Profiler 테이블
- 의미 : PyTorch Profiler가 GPU/CPU 성능을 분석한 결과 테이블. 훈련 중 병목 지점(시간, 메모리)을 식별하여 최적화에 사용됨

- 테이블 구조: 
    - Name : 연산 이름 (예: forward, data_to_device, aten::copy_ 등)
    - Self CPU % / Self CPU : 해당 연산의 자체 CPU 시간 비율/절대 시간
    - CPU total % / CPU total : 연산과 하위 연산의 총 CPU 시간
    - Self CUDA / Self CUDA % : 자체 GPU 시간/비율
    - CUDA total : 연산과 하위 연산의 총 GPU 시간
    - CPU/CUDA Mem : 메모리 사용량 (CPU/GPU)
    - of Calls : 호출 횟수

- 주요 연산 분석:
    - forward : 모델 순전파. 총 CUDA 시간 232.85ms (47.77%), 메모리 0b
    - data_to_device : 데이터를 GPU로 전송. CUDA 시간 148.25ms(30.41%), 메모리 0b. 데이터 전송이 큰 비중을 차지
    - aten::copy_ / Memcpy HtoD : 호스트(CPU)에서 디바이스(GPU)로 데이터 복사. CUDA 시간 147.28ms (30.22%), 메모리 0b. GPU 메모리 할당/복사 병목
    - aten::convolution_backward : 역전파 컨볼루션. CUDA 시간 146.22ms (29.50%), 메모리 4.01Gb. 역전파가 메모리 집약적
    - aten::cudnn_convolution : cuDNN 컨볼루션. CUDA 시간 69.66ms (14.29%), 메모리 4.94Gb. GPU 메모리 사용량 높음
    - torchvision::nms : Non-Maximum Suppression(중복 박스 제거). CUDA 시간 13.61ms (2.71%), 메모리 152.50Kb
    - 총합 : Self CPU 687.32ms, Self CUDA 487.44ms. GPU가 주된 연산 장치로, 메모리 사용이 높아 최적화 필요(예: 배치 크기 조정)

- 인사이트 : 데이터 전송(data_to_device)이 30% 이상 차지하므로, num_workers나 pin_memory로 개선 가능. 메모리 피크가 4-5Gb로, GPU 메모리 한계에 근접

## 3. Trace 저장 ([PROFILER] Trace saved to trace_step_5.json)
- 의미 : 프로파일러가 성능 데이터를 JSON 파일로 저장. Chrome Trace Viewer 나 Perfetto 로 시각화하여 타임라인 분석 가능

- 사용법 : 파일을 열어 연산별 시간 분포, 메모리 할당 그래프를 확인. 병목(ex: 긴 GPU 대기)을 찾아 코드 최적화

## 4. 추가 훈련 스텝 (Step 6/20 ~ 20/20)
- 의미 : 프로파일러 사이클 후 이어지는 배치 손실. 손실이 안정적으로 낮아짐 (최저 0.0829)

- 패턴 : 스텝 18에서 0.0829로 최저, 이후 약간 상승. 모델이 수렴 중임을 시사

## 5. 검증 출력 (COCO 평가 결과)
- 의미 : 검증 데이터셋으로 모델 성능 평가. COCO 메트릭을 사용해 mAP(Mean Average Precision) 계산

- 세부 설명: 
    - Test 진행 : 10개 배치 평가. model_time(추론 시간) 평균 0.0524s, evaluator_time(평가 시간)0.0010s, 총 시간 0.7456s/ 배치
    - IoU 메트릭:
        - AP @[IoU=0.50:0.95] = 0.502: 평균 정밀도(IoU 0.5~0.95 범위). 0.502는 50.2% 정확도
        - AP @[IoU=0.50] = 1.000: IoU 0.5에서 100% 정확도 (완벽탐지)
        - AP @[IoU=0.75] = 0.458: IoU 0.75에서 45.8% (엄격한 기준)
        - AR (Average Recall): 재현율. maxDets=100에서 57.5% (탐지된 객체 비율)
        - area 분류: small/large 객체에서 -1.000 (데이터 부족으로 계산 불가), medium에서 0.502
    - 해석 : mAP 0.502는 중간 성능. .IoU 0.5에서는 완벽하지만, 엄격한 IoU에서는 낮아짐. 데이터 증강이나 모델 튜닝 필요

## 6. 에포크 요약 ([Epoch 8] train_loss: 0.1774, ...)
- 의미 : 에포크 전체 요약

- 세부 설명:
    - train_loss: 0.1774: 훈련 평균 손실 (낮음, 수렴중)
    - valid_loss: 0.1966: 검증 손실 (훈련보다 약간 높음, 과적합 징후 업음)
    - mAP: 0.5018: 검증 mAP (0.50:0.95 기준)
    - mAP_50: 1.0000: IoU 0.5 기준 mAP (완벽)
    - epoch_time: 34.15 sec: 에포크 소요 시간 (GPU 동기화 포함)

- 인사이트 : 모델이 안정적으로 학습 중. mAP가 0.5 근처로, 추가 에포크로 개선 가능. 최고 mAP 시 체크포인트 저장 조건 충족 시 저장됨


# inference_.pt.py

## 이 파일(inference_.pt.py)은 PyTorch Faster R-CNN 모델을 사용하여 객체 탐지 추론을 수행하는 코드임. 
## 훈련된 모델(특히 TorchScript 형식의 .pt파일)을 로드하여 이미지 폴더의 모든 이미지를 처리하고, 탐지된 객체를 시각화하여 "DETECTED" 또는 "MISSED" 폴더에 저장함.
## 훈련 코드와 달리, 모델 평가 및 배포에 초점을 맞춤

## 1. 임포트 및 설정
- 임포트 : PyTorch, torchvision, PIL(이미지 처리), glob(파일 검색), warnings(경고무시)

- 설정 변수:
    - CLASS_NAMES: 클래스 목록 (훈련 때와 동일해야함, background 포함)
    - CKPT_PATH: 모델 체크포인트 경로 (.pt파일, TorchScript 형식)
    - TEST_IMG_DIR: 추론할 이미지 폴더
    - OUT_DIR: 결과 저장 폴더 (하위에 DETECTED/MISSED 생성)
    - SCORE_THRESH: 신뢰도 임계값 (0.5). 이 값 이상의 박스만 "탐지"로 인정

## 2. 모델 구조 정의
- 함수 : get_model(num_classes):
    - Faster R-CNN ResNet50 FPN 모델을 생성 (weights=None으로 사전 학습 무시)
    - ROI 헤드를 교체하여 클래스 수에 맞춤 (훈련 때와 동일)
    - 반환 : 모델 객체

## 3. 추론 함수
- 함수 : run_inference_on_image(model, device, img_path, detected_dir, missed_dir):
    - 이미지 로드 및 변환 : PIL로 이미지 열고, ToTensor로 [C,H,W]텐서 변환 (0~1범위), GPU로 이동

    - 모델 추론:
        - model.eval() 로 평가 모드 설정
        - torch.no_grad() 로 그래디언트 계산 비활성화
        - TorchScript 모델은 튜플 반환 (losses, detections), 일반 모델은 리스트. detections[0]을 추출
        - 출력 : boxes(박스 좌표), labels(클래스), scores(신뢰도)
    
    - 시각화 및 저장:
        - PIL Draw로 이미지에 박스 그리기(빨간색, width = 2)
        - 신뢰도 >= SCORE_THRESH 인 박스만 처리
        - 텍스트 : 클래스 이름 + 신뢰도 (ex: crack 0.85)
        - 폰트 : arial.ttf시도, 실패 시 기본 폰트
        - 검출 여부 : 하나라도 SCORE_THRESH 이상 박스가 있으면 detected = True
        - 저장 : DETECTED 폴더 (탐지됨) 또는 MISSED 폴더 (미탐지)
        - 출력 : 저장 경로와 검출 상태

## 4. 메인 함수
- 디바이스 설정 : GPU/CPU 자동 선택

- 폴더 생성 : DETECTED, MISSED 폴더 생성

- 모델 로드
    - get_model 로 구조 생성
    - torch.jit.load로 TorchScript 모델 로드 (.pt 파일 전용)
    - 모델을 디바이스로 이동, eval 모드

- 이미지 수집 : TEST_IMG_DIR에서 jpg, png 등 확장자 파일 검색 (대소문자 무시)

- 추론 루프 : 각 이미지에 대해 run_inference_on_image 호출

- 출력 : 이미지 수, 폴더 경로, 임계값, 완료 메시지

## 주요 특징 및 주의사항
- TorchScript 호환 : .pt 파일을 로드하여 배포 환경(CPU/GPU)에서 실행 가능. 일반 .pth와 달리 JIT 컴파일됨
- 신뢰도 필터링 : SCORE_THRESH로 낮은 신뢰도 박스 제외, 정확도 향상
- 폴더 분리 : 탐지 결과에 따라 자동 분류, 평가 용이
- 예외 처리 : 폰트 로드 실패, 모델 로드 오류 등
- 성능 : 배치 처리 없음(한장씩), 메모리 효율적. 대량 이미지 시 시간 소요
- 사용법 : python inference_.pt.py 실행. 결과 이미지를 확인하여 모델 성능 평가

# events 파일
- events.out.tfevents.1776304474.박지수.13492.0 파일은 TensorBoard에서 읽을 수 있는 이벤트 로그 파일

- 설명
    - 형식: TensorFlow 이벤트 파일(.tfevents). PyTorch의 SummaryWriter 가 생성하며, 훈련 중 로깅된 메트릭(손실, mAP 등)을 기록함

    - 내용: 스칼라 값, 히스토그램, 이미지 등. 파일명은 타임스탬프(1776304474), 사용자(박지수), 프로세스 ID(13492), 실행 번호(0) 로 구성됨

    - TensorBoard로 보기:
        - 1. 터미널에서 tensorboard --logdir=runs/custom_voc 실행(runs 폴더 경로 지정) 
        - 2. 브라우저에서 http://localhost:6006 열기
        - 3. Scalars 탭에서 손실 그래프, Metrics 탭에서 mAP 등을 확인 가능
    
    - 위치: custom_voc 폴더에 저장됨 (코드의 SummaryWriter(log_dir="runs/custom_voc"))

- 이 파일로 훈련 진행 상황을 시각화 할 수 있음

# trace_step_5.json 파일
- trace_step_5.json은 PyTorch Profiler가 생성한 성능 추적(trace)파일로, 훈련 과정의 세부 성능 데이터를 JSON 형식으로 기록함

- 용도
    - 성능분석 : CPU/GPU 연산 시간, 메모리 사용량, 연산 그래프를 기록하여 병목 지점(ex: 데이터 전송, 계산 지연)을 식별
    - 최적화 도구 : 모델/코드 개선에 사용. ex: GPU 메모리 부족이나 느린 연산을 찾아 배치 크기 조정. num_workers 변경 등
    - 디버깅 : 훈련 중 비효율적인 부분을 시각화하여 하드웨어 활용도를 높임

- 사용 방법
    - 1. 시각화 도구로 열기:
        - Chrome Trace Viewer: 크롬 브라우저에서 chrome://tracing/ 열고, 파일 업로드. 타임라인으로 연산별 시간 분포 확인
        - Perfetto: 온라인 툴(https://ui.perfetto.dev/)에 파일 업로드. 더 상세한 분석 (메모리 할당 그래프 등)
    
    - 2. 프로그래밍적 분석 : Python 으로 JSON 로드하여 커스텀 분석(ex: 특정 연산 시간 합산)

    - 3. 저장 위치 : profiler_logs 폴더 (코드의 trace_path = os.path.join("profiler_logs", f"trace_step_{prof.step_num}.json")).

- 이 파일은 프로파일러의 schedule 에 따라 스텝5(활성 단계)에서 생성됨
- 훈련 최적화에 필수적이며, 대량 데이터 시 유용함