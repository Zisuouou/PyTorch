| 구분 | Torch_1.py | Torch_2.py | 차이 의미 |
| :--- | :---: | ---: | ---: |
| 목적 | 일반 학습용 | 일반 학습 + 성능 분석용 | Torch_2는 병목 확인용 |
| 추가 import | 기본 PyTorch/torchvision/tensorboard 위주 | 여기에 torch.profiler 관련 import 추가 | Torch_2만 프로파일링 가능 |
| Profiler 기능 | 없음 | 있음 | CPU/CUDA 시간, 메모리 trace 저장 가능 |
| trace_handler() | 없음 | 있음 | 프로파일 결과를 콘솔 출력 + json trace 저장 |
| train_one_epoch() 구조 | 일반 학습 루프 | with profile(...)로 감싼 학습 루프 | Torch_2가 더 무겁지만 분석 가능 |
| 구간별 기록 | 없음 | data_to_device, forward, backward, optimizer_step 기록 | 어떤 단계가 느린지 확인 가능 |
| prof.step() | 없음 | 있음 | step 단위 profiler 스케줄 진행 |
| 디바이스 전송 | to(device) | to(device, on_blocking=True) | Torch_2가 전송 최적화 요소 포함 |
| gradient 초기화 | optimizer.zero_grad() | optimizer.zero_grad(set_to_none=True) | Torch_2가 약간 더 효율적일 순 있음 |
| trace 파일 저장 | 없음 | profiler_logs/trace_step_*.json 저장 | Chrome Trace / Perfetto 분석 가능 |
| 학습 속도 | 상대적으로 가벼움 | 상대적으로 느려질 수 있음 | profiler 오버헤드 때문 |
| 실사용 추천 | 평소 학습/모델 생성 | 병목 분석, 속도 문제 추적 시 | 용도 분리해서 쓰는 게 좋음 |

## 핵심
- Torch_2.py에는 torch.profiler import가 추가되어 있고, trace_handler() 함수와 profiler 스케줄 기반 train_one_epoch()가 들어가 있음
- 또 non_blocking=True, set_to_none=True, prof.step() 같은 성능 분석용 요소 포함
- 반면에 Torch_1.py 는 그런 기능 없이 더 단순한 학습 코드

## 공통점
- 두 파일은 모두 같은 데이터셋 구조, VOC XML 파싱, Faster R-CNN 모델 생성, TensorBoard 기록, mAP/mAP50 계산, b est.pt(h) 저장, 마지막 .pth 저장 흐름은 동일함

### 실무적으로 보면
- Torch_1.py : 평소 학습용
- Torch_2.py : 느린 원인을 찾을 때만 실행
    - 계속 Torch_2로 학습하면 profiler 때문에  불필요하게 무거워질 수 있음