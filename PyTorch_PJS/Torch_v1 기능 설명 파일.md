# Torch.v1 기능
1. Torch_2_TensorBoard_Profiler.py 학습 실행
2. inference_.pt.py 추론 실행
3. CLASS_NAMES GUI 에서 수정
4. DATA_ROOT / IMG_DIR / ANN_DIR 선택
5. epoch, train, batch, valid batch, num workers 수정
6. TensorBoard scalar 로그 폴더 지정
7. PyTorch Profiler 로그 폴더 지정
8. Profiler wait / warmup / active / repeat 수정
9. .pt 모델 경로 선택
10. 테스트 이미지 폴더 선택
11. 결과 저장 폴더 선택
12. threshold 수정
13. 학습/추론 로그 GUI 콘솔에 출력
14. 진행률 ProgressBar 표시
15. TensorBoard 6007 포트로 실행
16. Scalar 로그 폴더 / Profiler 로그 폴더 / results / checkpoints 바로 열기

# 1. 전체 실행 흐름
- Torch_v1.py 실행 
- MainWindow 생성
- 공통 / 학습 / 추론 / 실행 탭 UI 생성
- 사용자가 학습 시작 또는 추론 시작 클릭
- 입력값 검증
- ProcessWorker 스레드 생성
- 임시 launcher.py 생성
- launcher.py 가 Torch_2_TensorBoard_Profiler.py 또는 inference_.pt.py 로드
- GUI 입력값으로 원본 파일의 상수값 덮어쓰기
- 원본 파일의 main() 실행
- 콘솔 로그를 GUI에 출력
- 로그 패턴을 읽어서 ProgressBar 갱신

# 2. 전역 상수
## APP_NAME
- GUI 프로그램 이름
- 창 제목, 설정 저장 이름 등에 사용

## ORG_NAME
- Qt 설정 저장용 조직 이름
- QSettings 가 이전에 입력했던 경로, epoch, batch size 같은 값을 저장할 때 사용

## DEFAULT_BASE
- 기본 프로젝트 경로
    - C:\Users\SVT\Desktop\PyTorch_PJS\Torch_2_TensorBoard_Profiler.py
    - C:\Users\SVT\Desktop\PyTorch_PJS\inference_.pt.py
    - C:\Users\SVT\Desktop\PyTorch_PJS\runs\custom_voc
    - C:\Users\SVT\Desktop\PyTorch_PJS\results

# 3. LAUNCHER_CODE
- 문자열로 들어있는 임시 실행기 코드
- GUI가 학습/추론을 직접 실행하는 게 아닌, 실행 시점에 임시 .py 파일 하나 만들고 그 안에 이 코드를 넣음
    - 역할
        - GUI 입력값 받기
        - 원본 학습/추론 파일 import
        - DATA_ROOT, IMG_DIR, CLASS_NAMES 같은 상수값 덮어 쓰기
        - 원본 파일의 main() 실행
    
- 즉, 원본 파일을 수정하지 않고도 GUI 값으로 실행할 수 있게 해주는 핵심 코드

## LAUNCHER_CODE 안의 load_module_from_path(script_path)
- 기능
    - 지정된 .py 파일을 파이썬 모듈처럼 불로오는 함수

- GUI에서 학습을 누르면 Torch_2_TensorBoard_Profiler.py 를 모듈로 불러오고,
- 추론을 누르면 inference_.pt.py를 모듈로 불러옴

### 중요한 역할
- script_dir = os.path.dirname(script_path)

- if script_dir not in sys.path:
    - sys.path.insert(0, script_dir)

- os.chdir(script_dir)

- 이 부분은 Torch_2_TensorBoard_Profiler.py가 같은 폴더 안의 파일을 import 할 수 있게 해줌 

- 아래 같은 파일들이 같은 폴더에 있음
    - engine.py
    - utils.py
    - coco_eval.py
    - coco_utils.py
    - transforms.py

- 그래서 GUI에서 다른 위치에서 실해앟면 import가 깨짐
- script_dir를 sys.path 에 추가하고 작업 폴더도 그쪽으로 바꿔줌

### LAUNCHER_CODE 안의 main()
- 기능
    - 실제 학습/추론 파일을 실행하는 함수


- 동작 순서
    - cfg = json.loads(os.environ.get("PYTORCH_GUI_CONFIG", "{}"))
- GUI 에서 넘겨준 설정값을 환경변수에서 읽음


- module = load_module_from_path(script_path)
    - 원본 학습/추론 파일을 모듈로 불러옴


- for key, value in overrides.items():
    - setattr(module, key, value)
- 이 부분이 핵심
- 원본 파일 안에 있는 상수 값을 GUI 값으로 덮어 씀

# 4. 일반 유틸 함수
## norm_path(text: str) -> str
- 기능
    - GUI 입력칸에 들어온 경로 문자열을 정리해주는 함수
    - 앞뒤 공백고 따옴표를 제거해서 정상 경로로 바꿔줌

## parse_classes(text: str) -> list[str]
- 기능
    - GUI의 클래스 입력칸에 적은 내용을 CLASS_NAMES 리스트로 변환해주는 함수

    - 사용자가 직접 입력할 수 있음, 콤마로 가로로 입력해도 됨
    - EX) 
        - __background__
        - BlackSpot
        - MilPinContamin
        - Contamin
        - Bubble
    - 혹은
        - __background__, BlackSpot, MilPinContamin, Contamin, Bubble

- 콤마를 줄바꿈으로 바꿔줌
- 중복 클래스명을 제거해줌

- PyTorch detection 모델은 0번 클래스가 __background__이므로 자동으로 맨 앞 추가 

# 5. ProcessWorker 클래스
- 학습/추론 프로세스를 GUI와 분리해서 실행하는 스레드 

## ProcessWorker.\__init__()
- 기능
    - 스레드 실행에 필요한 값을 저장

- 받는 값

| 인자 | 의미 |
| -- | -- |
| python_exe | 사용할 Python 실행 파일 |
| script_path | 실행할 원본 학습/추론 파일 |
| mode | "train" 또는 "infer" |
| overrides | GUI 에서 입력한 덮어쓰기 설정값 | 


## ProcessWorker.run()
- 기능 
    - 실제로 학습/추론을 실행하는 핵심 함수

### 1단계 : Python 경로 확인
- GUI에서 Python exe를 입력했으면 그걸 쓰고, 비어 있으면 현재 GUI를 실행한 Python 을 씀

### 2단계 : 학습/추론 스크립트 확인
- Torch_2_TensorBoard_Profiler.py 또는 inference_.pt.py 파일이 실제로 있는지 확인 
    - 없으면 실행하지 않음

### 3단계 : 임시 launcher 파일 생성

### 4단계 : GUI 설정값을 환경변수로 전달
- GUI 입력값을 JSON으로 만들어서 환경변수에 넣음
    - 환경변수로 넘기는 이유는 명령어 인자로 길게 넘기는 것보다 안전하기 때문

### 5단계 : subprocess 실행
- 로그를 모아서 한 번에 출력하지 않고 바로바로 GUI 에 전달하기 위한 옵션

- 학습/추론 프로세스를 실행
- 표준 출력과 에러 출력을 모두 GUI에서 읽을 수 있게 해줌
    - 그래서 오류도 GUI 콘솔창에 표시됨

### 6단계 : 로그 실시간 읽기
- 학습/추론 파일에서 출력되는 로그를 한 줄씩 읽어서 GUI에 보내줌

### 7단계 : 종료 코드 반환
- 프로세스가 끝날 때까지 기다린 뒤 종료 코드를 GUI에 보냄

### 8단계 : 임시 launcher 삭제
- 실행이 끝나면 임시 파일 삭제 

## ProcessWorker.stop()
- 기능 
    - 실행중인 학습/추론 프로세스를 중지함 
    - GUI 의 중지 버튼을 누르면 호출 됨

**학습 중 강제 종료하면 checkpoint 저장 중이었을 경우 파일이 불완전할 수 있음, 따라서 가능하면 epoch 종료 시점이나 저장 완료 후 중지하는 게 좋음** 

# 6. PathRow 클래스
- 경로 입력칸과 찾기 버튼을 한 줄로 묶어주는 위젯

## PathRow.\__init__()
- 기능
    - 경로 입력칸과 찾기 버튼을 생성함

- 인자 설명

| 인자 | 의미 |
| -- | -- |
| line_edit | 실제 경로가 들어갈 입력칸 |
| mode | "file" 이면 파일 선택, "dir" 이면 폴더 선택 |
| file_filter | 파일 선택창에서 보여줄 확장자 필터 |

# 7. MainWindow 클래스
- GUI 메인 창 전체를 담당하는 클래스 
    - 버튼, 입력탄, 탭, 로그창, TensorBoard 실행, 학습 실행, 추론 실행 모두 관리

# 8. UI 생성 함수들
- GUI 전체 화면 구조를 만든느 함수

| 만드는 것 |
| -- |
| 메인 제목 |
| 서브 제목 |
| 상단 탭 영역 |
| 하단 로그 영역 |
| ProgressBar |
| 로그 지우기 버튼 |

- 핵심 구조 
    - 위쪽 : 공통 설정 / 학습 설정 / 추론 설정 / 실행 탭
    - 아래쪽 : 실행 로그 콘솔 + ProgressBar

- 탭 위젯 생성
    - 1. 공통 설정
    - 2. 학습 설정
    - 3. 추론 설정
    - 4. 실행 / TensorBoard


#### TensorBoard 로그 폴더 

| GUI 항목 | 의미 |
| -- | -- |
| SCALAR_LOG_DIR | loss, mAP, IoU 같은 일반 scalar 로그 저장 폴더 |
| PROFILER_LOG_DIR | PyTorch Profiler 로그 저장 폴더 |

#### Profiler 설정
- 각 원본 학습 파일의 Profiler 관련 상수값에 들어감

| GUI 항목 | 의미 |
| -- | -- |
| ENABLE_PROFILER | Profiler 사용 여부 |
| PROFILER_ONLY_FIRST_EPOCH | 첫 epoch만 profiling 할지 |
| PROFILER_WAIT | profiler 대기 step |
| PROFILER_WARMUP | warmup step |
| PROFILER_ACTIVE | 실제 기록 step |
| PROFILER_REPEAT | 반복 횟수 |

- SCORE_THRESH
    - 검출 confidence threshold 
    - 예를 들어 0.5면 score 0.5 이상인 detection만 결과에 반영

#### 실행 버튼

| 버튼 | 기능 |
| -- | -- |
| 학습 시작 | Torch_2_TensorBoard_Profiler.py 실행 |
| 추론 시작 | inference_.pt.py 실행 |
| 중지 | 실행 중인 프로세스 종료 |

#### TensorBoard 버튼

| 버튼 | 기능 |
| -- | -- |
| Scalar TensorBoard 열기 | SCALAR_LOG_DIR 기준 TensorBoard 실행 |
| Profiler TensorBoard 열기 | PROFILER_LOG_DIR 기준 TensorBoard 실행 |

# 9. 이벤트 연결 함수 
- 버튼 클릭과 실제 함수를 연결
    - 학습 시작 버튼을 누르면 start_train() 실행
    - 추론 시작 버튼을 누르면 start_infer() 실행 
    - 중지 버튼을 누르면 stop_process() 실행
    - 로그 지우기 버튼을 누르면 콘솔창을 비움

- DATA_ROOT 값이 바뀌면 IMG_DIR, ANN_DIR 도 자동 보정할 수 있게 연결 

# 10. 설정 저장/불러오기 함수
### _load_settings()
- 기능 
    - 이전에 GUI 에서 입력했던 값을 다시 불러옴

### _save_settings()
- 기능
    - 현재 GUI 입력값을 저장함
    - 학습/추론 시작할 때도 호출되고, GUI 창을 닫을 때도 호출

#### 저장되는 것
1. Python exe
2. 학습 스크립트 경로
3. 추론 스크립트 경로
4. 데이터셋 경로
5. 로그 폴더
6. 모델 경로
7. 결과 폴더
8. CLASS_NAMES 
9. epoch
10. batch size
11. worker 수
12. profiler 설정
13. threshold 
14. TensorBoard port

# 11. 데이터셋 경로 자동 동기화
### _maybe_sync_dataset_paths()
- 기능 
    - DATA_ROOT를 바꿨을 때 IMG_DIR, ANN_DIR을 자동으로 맞춰주는 함수

- 예시 
    - DATA_ROOT 를 D:\AI_DATA\annos 로 바꾸면
- 자동으로 아래 경로를 예상함
    - IMG_DIR = D:\AI_DATA\annos\image
    - ANN_DIR = D:\AI_DATA\annos\xml

- 기본 이미지 폴더명은 image, XML 폴더명은 xml로 가정함 

**주의: 이 함수는 기존 값이 기본 경로일 때만 자동으로 변경되도록 되어 있음. 즉, 사용자가 수동으로 다른 경로를 넣어둔 경우에는 함부로 덮어쓰지 않게 하려는 구조**


# 12. 공통 검증 함수
### validate_common()
- 기능 
    - 학습/추론 실행 전 공통으로 필요한 항목을 검사합니다.

### 검사 1. Python exe 존재 여부
- Python 실행 파일이 실제로 있는지 확인함
- 없으면 메세지창 띄움

### 검사 2. CLASS_NAMES 확인
- 클래스 첫 번째가 \__background__ 인지 확인

# 13. 학습 실행 함수
### start_train()
- 기능
    - 학습 시작 버튼을 눌렀을 때 실행되는 함수

### 1단계 : 이미 실행중인지 확인
- 이미 학습 또는 추론이 실행 중이면 새로 실행하지 않음
- 동시에 여러 개 실행 시 GPU 메모리 충돌이나 파일 충돌이 날 수 있기 때문

### 2단계 : 공통 검증
- Python exe, 클래스명 등 확인

### 3단계 : 학습 스크립트 확인
- Torch_2_TensorBoard_Profiler.py 파일이 있는지 확인

### 4단계 : 데이터셋 폴더 혹인
- 이미지 폴더와 XML 폴더가 실제로 있는지 확인

### 5단계 : overrides 생성
- 여기서 원본 학습 파일에 덮어씌울 값들을 만듦
- 즉, GUI 값이 원본 파일의 이런 변수들을 대체함
    - DATA_ROOT
    - IMG_DIR
    - ANN_DIR
    - CLASS_NAMES
    - NUM_EPOCHS
    - TRAIN_BATCH_SIZE
    - VALID_BATCH_SIZE
    - NUM_WORKERS
    - SCALAR_LOG_DIR
    - PROFILER_LOG_DIR
    - PROFILER_WAIT
    - PROFILER_WARMUP
    - PROFILER_ACTIVE
    - PROFILER_REPEAT
    - ENABLE_PROFILER
    - PROFILER_ONLY_FIRST_EPOCH

### 6단계 : 로그 폴더 생성
- TensorBoard 로그 폴더가 없으면 자동 생성

### 7단계 : 진행률 초기화
- ProgressBar 계산을 위해 전체 epoch 수와 현재 모드를 저장함

# 14. 추론 실행 함수
### start_infer()
- 기능
    - 추론 시작 버튼을 눌렀을 때 실행되는 함수

### 1단계 : 이미 실행 중인지 확인
- 학습이나 추론이 이미 실행 중이면 새로 실행하지 않음

### 2단계 : 공통 검증
- Python exe 와 클래스명 등 확인

### 3단계 : 추론 스크립트 확인
- inference_.pt.py 파일이 실제로 있는지 확인

### 4단계 : 모델 파일 확인
- .pt 모델 파일이 실제로 있는지 확인

### 5단계 : 테스트 이미지 폴더 확인
- 추론할 이미지 폴더가 있는지 확인

### 6단계 : overrides생성
- 추론 파일에 덮어씌울 값들
- 즉, GUI 값이 원본 추론 파일의 아래 상수들을 대체함
    - CLASS_NAMES
    - CKPT_PATH
    - TEST_IMG_DIR
    - OUT_DIR
    - SCORE_THRESH

### 7단계 : 결과 폴더 생성
- 결과 저장 폴더가 없으면 생성함

### 8단계 : 진행률 초기화
- 추론 이미지 개수 기반으로 ProgressBar 를 계산하기 위한 초기값

### 9단계 : 설정 저장 후 worker 실행
- 추론 프로세스를 시작

# 15. worker 실행 관련 함수
### _start_worker()
- 기능
    - ProcessWorker를 만들고 실제로 실행하는 함수
    - 학습과 추론 모두 이 함수를 통해 실행됨

- 동작
    - 실행 중 상태로 바꿈
    - 즉, 
        - 학습 시작 버튼 비활성화
        - 추론 시작 버튼 비활성화
        - 중지 버튼 활성화

    - 실제 실행 스레드 생성
    - worker 에서 발생한 로그를 GUI 콘솔 출력 함수에 연결함
    - 프로세스 ID를 GUI 로그에 출력
    - 프로세스가 끝나면 on_finished() 가 실행되도록 연결
    - 스레드 실행 시작

### stop_process()
- 기능
    - GUI의 중지 버튼을 눌렀을 때 실행 

    - 현재 실행중인 worker 가 있으면 그 안의 프로세스를 종료함

### on_finished(code: int)
- 기능
    - 학습/추론 프로세스가 종료되었을 때 실행

- 동작
    - 종료 코드를 로그에 출력

    - 정상 종료면 ProgressBar를 100%로 만듦 

    - 실행 상태를 초기화
    - 즉,
        - 학습 시작 버튼 다시 활성화
        - 추론 시작 버튼 다시 활성화
        - 중지 버튼 비활성화

### set_running_state(running: bool)
- 기능
    - 학습/추론 실행 중인지 여부에 따라 버튼 상태를 바꿈

    - 실행 중일 때
        - 학습 시작 비활성화
        - 추론 시작 비활성화
        - 중지 활성화

    - 실행 안 할 때
        - 학습 시작 활성화
        - 추론 시작 활성화
        - 중지 비활성화

# 16. 로그 출력 및 진행률 함수
### on_log(text: str)

- 기능
    - worker에서 전달받은 로그를 GUI 콘솔창에 출력

- 동작
    - 콘솔 맨 아래에 새 로그를 추가하고, 자동으로 스크롤을 아래로 내림

    - 로그 내용을 분석해서 ProgressBar를 갱신함

### _update_progress_from_log(text: str)
- 기능
    - 학습/추론 로그 문장을 보고 ProgressBar를 업데이트함

- 학습 모드일 때
    - 학습 로그에서 epoch와 step 정보를 찾음

#### 패턴1
- [Epoch 3] Step 5/100 loss... 이런 로그를 찾음
- 여기서 추출 값은

| 값 | 의미 |
| -- | -- |
| 3 | 현재 epoch |
| 5 | 현재 step |
| 100 | 해당 epoch의 전체 step |

#### 진행률 계산
- 예를 들어: 
    - 전체 epoch = 10
    - 현재 epoch = 3
    - 현재 step = 50/100

- 2.5 epoch 진행 / 10 epoch = 25% 정도로 계산

#### 패턴2
- [Epoch 3] train_loss=... 이런 로그를 보면
    - step 로그가 없을 때 epoch 단위로 진행률을 올리기 위한 보조 패턴

### 추론 모드일 때
- 추론 로그에서 전체 이미지 수와 저장 완료 수를 찾음

#### 전체 이미지 수 찾기
- [INFO] found 120 images 이런 로그를 보면
    - 전체 이미지 수를 120으로 설정

#### 저장 완료 카운트
- [SAVED] 추론 파일에서 이미지 저장할 때마다 이런 로그가 나온다고 가정하면
    - 그 로그가 나올 때마다 완료 이미지 수를 1씩 증가시킴

#### 추론 완료
- 추론 완료 로그가 나오면 ProgressBar를 100%로 만듦

### 중요 주의점
**ProgressBar는 원본 학습/추론 파일의 로그 형식에 의존함**

**만약 원본 파일 로그 형식이 다르다면 학습/추론은 정상 실행 되어도 ProgressBar가 안 움직일 수 있음, 그럴 때는 _update_progress_from_log()의 정규식만 바꾸면 됨**

# 17. TensorBoard 실행 함수
### start_tensorboard(logdir_text: str)
- 기능
    - TensorBoard를 실행하고 브라우저로 엶

- 받는 값 
    - logdir_text
        - TensorBoard가 읽을 로그 폴더

    - Scalar 버튼을 누르면:
        - self.edit_scalar_log.text()
    - Profiler 버튼을 누르면:
        - self.edit_profiler_log.text() 가 들어감

- 로그 폴더가 없으면 자동 생성

#### URL 생성
- 기본값은 6007이며
    - http://localhost:6007/ 이 열림

- 이미 TensorBoard 가 실행중이면 새로 실행하지 않고 브라우저만 엶

#### 실행 후보 명령어
- TensorBoard 실행 방식이 환경마다 다를 수 있어서 3가지 방식으로 시도함

- 1순위:
    - python -m tensorboard.main

- 2순위:
    - python -m tensorboard

- 3순위:
    - tensorboard

#### 브라우저 열기
- TensorBoard 서버가 켜질 시간을 조금 준 뒤 브라우저를 엶


# 18. 폴더 열기 함수 
### open_path(path_text: str) 
- 기능
    - 입력된 경로를 OS 탐색기로 엶
    - Windows 에서는 파일 탐색기가 열림

- 경로가 없으면 폴더 자동 생성

### open_checkpoints_dir()
- 기능
    - 학습 스크립트가 있는 폴더 기준으로 checkpoints() 폴더를 엶

- 예시로 학습 스크립트가:
    - C:\Users\SVT\Desktop\PyTorch_PJS\Torch_2_TensorBoard_Profiler.py 이면 checkpoints 폴더는
        - C:\Users\SVT\Desktop\PyTorch_PJS\checkpoints

# 19. 창 종료 처리 함수
### closeEvent(self, event)
- 기능 
    - GUI 창을 닫을 때 실행되는 함수

### 1단계 : 현재 설정 저장
- 창을 닫기 전에 입력값을 저장함

### 2단계 : 학습/추론 실행 중인지 확인
- 실행 중인 프로세스가 있으면 바로 닫지 않고 물어봄

### 3단계 : 사용자가 Yes 선택
- 프로세스를 종료하고 창을 닫음

### 4단계 : 사용자가 No 선택
- 창 닫기를 취호

# 20. 프로그램 시작 함수
### main() 
- 기능
    - GUI 프로그램을 실제로 시작하는 함수

- 동작
    - PyQt6 애플리케이션 객체 생성

    - 설정 저장용 이름을 지정

    - 메인 창을 만들고 화면에 표시

    - Qt 이벤트 루프를 실행
        - sys.exit(app.exec())
        - 이 코드가 있어야 버튼 클릭, 창 이동, 로그 출력 같은 GUI 동작이 계속 유지됨


# 21. Torch_v1 GUI 에서 가장 중요한 함수 TOP5
- 실제 수정하거나 문제가 생겼을 때 가장 많이 볼 함수 

## 1. start_train()
- 학습 실행 설정값을 만드는 곳
- 학습 파일에 새 상수를 추가했다면 여기 overrides에 추가하면 됨
    - 예: "LEARNING_RATE": float(self.spin_lr.value())

## 2. start_infer()
- 추론 실행 설정값을 만드는 곳
- 추론 파일에 새 옵션을 추가했다면 여기 overrides에 추가하면 됨
    - 예: "IOU_THRESH": float(self.spin_iou.value())

## 3. _update_progress_from_log()
- ProgressBar가 안 움직이면 대부분 여기 문제
- 원본 파일 로그 형식과 GUI가 찾는 정규식이 안 맞으면 수정해야 함

## 4. ProcessWorker.run()
- 실제 subprocess 실행 담당
- Python 환경, launcher 생성, 로그 수집 문제가 생기면 여기 보면 됨

## 5. start_tensorboard()
- TensorBoard 실행이 안 되면 여기 보면 됨

    - python -m tensorboard 가 안 먹는 경우가 있어 여러 후보 명령어 넣어둔거임


# 22. 한 줄 요약
- 이 GUI는 단순 버튼 UI 가 아닌

**GUI 입력값 수집. -> PyTorch 학습/추론 파일의 상수값 덮어쓰기 -> subprocess로 실행 -> 로그 실시간 표시 -> 로그 패턴 기반 ProgressBar 갱신 -> TensorBoard/Profiler 실행**

**까지 담당하는 PyTorch 학습/추론 전용 실행기 구조**