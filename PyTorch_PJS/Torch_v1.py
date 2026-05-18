"""
PyTorch 전용 GUI_v1.0
- Torch_2_TensorBoard_Profiler.py 학습 실행
- inference_.pt.py TorchScript(.pt) 추론 실행
- TensorBoard Scalar, Profiler 실행

특징
1. 원본 학습/추론 파일을 직접 수정하지 않음
2. GUI에서 입력한 값을 임시 launcher에서 모듈 상수로 덮어씌운 뒤 main() 실행
3. 콘솔 로그를 GUI에 그대로 표시
4. 학습/추론 프로세스는 별도 subprocess로 실행되어 GUI 멈추지 않음
"""

import json, os, re, subprocess, sys, tempfile, traceback, webbrowser
from PyQt6.QtCore import Qt, QSettings, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget)

APP_NAME = "PyTorch GUI v1.0"
ORG_NAME = "SVT"
DEFAULT_BASE = r"C:\Users\SVT\Desktop\PyTorch_PJS"

LAUNCHER_CODE = """
# -*- coding: utf-8 -*-
import importlib.util
import json
import os
import sys
import traceback

def load_module_from_path(script_path):
    script_path = os.path.abspath(script_path)
    script_dir = os.path.dirname(script_path)

    # engine.py, utils.py, coco_eval.py 등 같은 폴더 import 대응
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    os. chdir(script_dir)

    spec = importlib.util.spec_from_file_location("svt_target_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load target script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    cfg = json.loads(os.environ.get("PYTORCH_GUI_CONFIG", "{}"))
    script_path = cfg["script_path"]
    overrides = cfg.get("overrides", {})

    print("=" * 80)
    print("[GUI LAUNCHER] target:", script_path)
    print("[GUI LAUNCHER] mode:", cfg.get("mode", "unknown"))
    print("[GUI LAUNCHER] overrides:")
    for k, v in overrides.items():
        print(f" - {k} = {v}")
    print("=" * 80)

    module = load_module_from_path(script_path)

    # 원본 파일 상수를 GUI 값으로 덮어쓰기
    for key, value in overrides.items():
        setattr(module, key, value)

    if not hasattr(module, "main"):
        raise RuntimeError("target script에 main() 함수가 없습니다.")

    module.main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[STOPPED] KeyboardInterrupt")
    except Exception:
        print("[GUI LAUNCHER ERROR]")
        traceback.print_exc()
        sys.exit(1)
"""

def norm_path(text: str) -> str:
    return os.path.normpath(text.strip().strip('"').strip("'"))

def parse_classes(text: str) -> list[str]:
    """
    입력 예시 허용:
    __background__
    crack
    
    또는:
    __background__, crack
    """
    raw = text.replace(",", "\n").splitlines()
    classes = []
    for item in raw:
        item = item.strip().strip('"').strip("'")
        if not item:
            continue
        if item not in classes:
            classes.append(item)

    if not classes:
        classes = ["__background__", "crack"]

    if classes[0] != "__background__":
        classes = ["__background__"] + [c for c in classes if c != "__background__"]

    return classes

class ProcessWorker(QThread):
    log = pyqtSignal(str)
    finished_code = pyqtSignal(int)
    started_pid = pyqtSignal(int)

    def __init__(self, python_exe: str, script_path: str, mode: str, overrides: dict, parent=None):
        super().__init__(parent)
        self.python_exe = python_exe
        self.script_path = script_path
        self.mode = mode
        self.overrides = overrides
        self.proc: subprocess.Popen | None = None
        self.launcher_path: str | None = None

    def run(self):
        try:
            py = self.python_exe or sys.executable
            script_path = os.path.abspath(self.script_path)
            script_dir = os.path.dirname(script_path)

            if not os.path.isfile(py):
                self.log.emit(f"[ERROR] Python 실행 파일을 찾을 수 없습니다: {py}\n")
                self.finished_code.emit(-1)
                return

            if not os.path.isfile(script_path):
                self.log.emit(f"[ERROR] 스크립트 파일을 찾을 수 없습니다: {script_path}\n")
                self.finished_code.emit(-1)
                return
            
            fd, self.launcher_path = tempfile.mkstemp(prefix="svt_pytorch_gui_launcher_", suffix=".py")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(LAUNCHER_CODE)

            env = os.environ.copy()
            env["PYTORCH_GUI_CONFIG"] = json.dumps(
                {
                    "script_path": script_path,
                    "mode": self.mode,
                    "overrides": self.overrides,
                },
                ensure_ascii=False,
            )

            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            cmd = [py, "-u", self.launcher_path]
            self.log.emit(f"[CMD] {' '.join(cmd)}\n")
            self.log.emit(f"[CWD] {script_dir}\n\n")

            self.proc = subprocess.Popen(
                cmd,
                cwd=script_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )

            self.started_pid.emit(self.proc.pid)

            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.log.emit(line)

            code = self.proc.wait()
            self.finished_code.emit(code)

        except Exception :
            self.log.emit("[GUI PROCESS ERROR]\n")
            self.log.emit(traceback.format_exc())
            self.finished_code.emit(-1)
        finally:
            if self.launcher_path and os.path.exists(self.launcher_path):
                try:
                    os.remove(self.launcher_path)
                except Exception:
                    pass

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.log.emit("\n[STOP] 프로세스 종료 요청...\n")
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.log.emit("[STOP] terminate 실패 -> kill 시도\n")
                    self.proc.kill()
            except Exception as e:
                self.log.emit(f"[STOP ERROR] {e}\n")

class PathRow(QWidget):
    def __init__(self, line_edit: QLineEdit, mode: str = "file", file_filter: str = "All Files (*.*)", parent=None):
        super().__init__(parent)
        self.line_edit = line_edit
        self.mode = mode
        self.file_filter = file_filter

        btn = QPushButton("찾기")
        btn.setFixedWidth(58)
        btn.clicked.connect(self.browse)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(btn)

    def browse(self):
        current = self.line_edit.text().strip()
        start_dir = current if os.path.isdir(current) else os.path.dirname(current)
        if not start_dir:
            start_dir = DEFAULT_BASE
        
        if self.mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "폴더 선택", start_dir)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "파일 선택", start_dir, self.file_filter)
        
        if path:
            self.line_edit.setText(os.path.normpath(path))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1180, 820)

        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.worker: ProcessWorker | None = None
        self.tensorboard_proc: subprocess.Popen | None = None
        self.current_mode: str | None = None
        self.total_infer_images = 0
        self.done_infer_images = 0
        self.total_epochs = 0

        self._build_ui()
        self._load_settings()
        self._connect_signals()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QVBoxLayout(root)

        title = QLabel("PyTorch Faster R-CNN 전용 GUI v1.0")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        main_layout.addWidget(title)

        subtitle = QLabel("Torch_2_TensorBoard_Profiler.py 학습 + inference_.pt.py 추론 + TensorBoard/Profiler 실행")
        subtitle.setStyleSheet("color: #555;")
        main_layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter, 1)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        splitter.addWidget(top)

        tabs = QTabWidget()
        top_layout.addWidget(tabs)

        self.common_tab = QWidget()
        self.train_tab = QWidget()
        self.infer_tab = QWidget()
        self.run_tab = QWidget()

        tabs.addTab(self.common_tab, "공통 설정")
        tabs.addTab(self.train_tab, "학습 설정")
        tabs.addTab(self.infer_tab, "추론 설정")
        tabs.addTab(self.run_tab, "실행 / TensorBoard")

        self._build_common_tab()
        self._build_train_tab()
        self._build_infer_tab()
        self._build_run_tab()

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        splitter.addWidget(bottom)

        log_header = QHBoxLayout()
        log_label = QLabel("실행 로그")
        log_label.setStyleSheet("font-weight: 700;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.btn_clear_log = QPushButton("로그 지우기")
        log_header.addWidget(log_label)
        log_header.addWidget(self.progress, 1)
        log_header.addWidget(self.btn_clear_log)
        bottom_layout.addLayout(log_header)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(20000)
        self.console.setStyleSheet(
            """
            QPlainTextEdit {
                background: #111;
                color: #e8e8e8;
                font-family: Consolas, 'D2Coding', monospace;
                font-size: 10.5pt;
            }
            """
        )
        bottom_layout.addWidget(self.console, 1)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

    def _build_common_tab(self):
        layout = QVBoxLayout(self.common_tab)

        box = QGroupBox("스크립트 / Python 경로")
        form = QFormLayout(box)

        self.edit_python = QLineEdit(sys.executable)
        self.edit_train_script = QLineEdit(os.path.join(DEFAULT_BASE, "Torch_2_TensorBoard_Profiler.py"))
        self.edit_infer_script = QLineEdit(os.path.join(DEFAULT_BASE, "inference_.pt.py"))

        form.addRow("Python exe", PathRow(self.edit_python, "file", "Python (*.exe);;All Files(*.*)"))
        form.addRow("학습 스크립트", PathRow(self.edit_train_script, "file", "Python (*.py);;All Files(*.*)"))
        form.addRow("추론 스크립트", PathRow(self.edit_infer_script, "file", "Python (*.py);;All Files(*.*)"))

        layout.addWidget(box)

        box2 = QGroupBox("클래스 이름")
        v = QVBoxLayout(box2)
        self.edit_classes = QTextEdit()
        self.edit_classes.setPlaceholderText("__background__\ncrack")
        self.edit_classes.setFixedHeight(120)
        self.edit_classes.setPlainText("__background__\ncrack")
        v.addWidget(QLabel("학습/추론 파일의 CLASS_NAMES에 들어갈 값입니다. 0번은 반드시 __background__ 입니다."))
        v.addWidget(self.edit_classes)
        layout.addWidget(box2)

        note = QLabel("※ 이 GUI는 원본 .py 파일을 직접 고치지 않고, 실행할 때만 임시 launcher에서 상수값을 덮어씁니다.")
        note.setStyleSheet("color: #666;")
        layout.addWidget(note)
        layout.addStretch(1)

    def _build_train_tab(self):
        layout = QVBoxLayout(self.train_tab)

        dataset_box = QGroupBox("데이터셋 경로")
        form = QFormLayout(dataset_box)

        self.edit_data_root = QLineEdit(os.path.join(DEFAULT_BASE, "annos"))
        self.edit_img_dir = QLineEdit(os.path.join(DEFAULT_BASE, "annos", "image"))
        self.edit_ann_dir = QLineEdit(os.path.join(DEFAULT_BASE, "annos", "xml"))

        form.addRow("DATA_ROOT", PathRow(self.edit_data_root, "dir"))
        form.addRow("IMG_DIR", PathRow(self.edit_img_dir, "dir"))
        form.addRow("ANN_DIR", PathRow(self.edit_ann_dir, "dir"))
        layout.addWidget(dataset_box)

        train_box = QGroupBox("학습 파라미터")
        grid = QGridLayout(train_box)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 100000)
        self.spin_epochs.setValue(10)

        self.spin_train_batch = QSpinBox()
        self.spin_train_batch.setRange(1, 1024)
        self.spin_train_batch.setValue(2)

        self.spin_valid_batch = QSpinBox()
        self.spin_valid_batch.setRange(1, 1024)
        self.spin_valid_batch.setValue(1)

        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 64)
        self.spin_workers.setValue(4)

        self.edit_scalar_log = QLineEdit(os.path.join(DEFAULT_BASE, "runs", "custom_voc"))
        self.edit_profiler_log = QLineEdit(os.path.join(DEFAULT_BASE, "runs", "profiler"))

        grid.addWidget(QLabel("NUM_EPOCHS"), 0, 0)
        grid.addWidget(self.spin_epochs, 0, 1)
        grid.addWidget(QLabel("TRAIN_BATCH_SIZE"), 0, 2)
        grid.addWidget(self.spin_train_batch, 0, 3)

        grid.addWidget(QLabel("VALID_BATCH_SIZE"), 1, 0)
        grid.addWidget(self.spin_valid_batch, 1, 1)
        grid.addWidget(QLabel("NUM_WORKERS"), 1, 2)
        grid.addWidget(self.spin_workers, 1, 3)

        grid.addWidget(QLabel("SCALAR_LOG_DIR"), 2, 0)
        grid.addWidget(PathRow(self.edit_scalar_log, "dir"), 2, 1, 1, 3)

        grid.addWidget(QLabel("PROFILER_LOG_DIR"), 3, 0)
        grid.addWidget(PathRow(self.edit_profiler_log, "dir"), 3, 1, 1, 3)

        layout.addWidget(train_box)

        prof_box = QGroupBox("PyTorch Profiler 설정")
        grid2 = QGridLayout(prof_box)

        self.chk_enable_profiler = QCheckBox("ENABLE_PROFILER")
        self.chk_enable_profiler.setChecked(True)
        self.chk_profile_first = QCheckBox("PROFILE_ONLY_FIRST_EPOCH")
        self.chk_profile_first.setChecked(True)

        self.spin_prof_wait = QSpinBox()
        self.spin_prof_wait.setRange(0, 1000)
        self.spin_prof_wait.setValue(1)

        self.spin_prof_warmup = QSpinBox()
        self.spin_prof_warmup.setRange(0, 1000)
        self.spin_prof_warmup.setValue(1)

        self.spin_prof_active = QSpinBox()
        self.spin_prof_active.setRange(1, 1000)
        self.spin_prof_active.setValue(3)

        self.spin_prof_repeat = QSpinBox()
        self.spin_prof_repeat.setRange(1, 1000)
        self.spin_prof_repeat.setValue(1)

        grid2.addWidget(self.chk_enable_profiler, 0, 0, 1, 2)
        grid2.addWidget(self.chk_profile_first, 0, 2, 1, 2)
        grid2.addWidget(QLabel("WAIT"), 1, 0)
        grid2.addWidget(self.spin_prof_wait, 1, 1)
        grid2.addWidget(QLabel("WARMUP"), 1, 2)
        grid2.addWidget(self.spin_prof_warmup, 1, 3)
        grid2.addWidget(QLabel("ACTIVE"), 2, 0)
        grid2.addWidget(self.spin_prof_active, 2, 1)
        grid2.addWidget(QLabel("REPEAT"), 2, 2)
        grid2.addWidget(self.spin_prof_repeat, 2, 3)

        layout.addWidget(prof_box)
        layout.addStretch(1)

    def _build_infer_tab(self):
        layout = QVBoxLayout(self.infer_tab)

        box = QGroupBox("TorchScript 추론 설정")
        form = QFormLayout(box)

        self.edit_ckpt = QLineEdit(os.path.join(DEFAULT_BASE, "checkpoints", "fasterrcnn_best.pt"))
        self.edit_test_img_dir = QLineEdit(os.path.join(DEFAULT_BASE, "annos", "image"))
        self.edit_out_dir = QLineEdit(os.path.join(DEFAULT_BASE, "results"))
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.0, 1.0)
        self.spin_thresh.setDecimals(3)
        self.spin_thresh.setSingleStep(0.05)
        self.spin_thresh.setValue(0.5)

        form.addRow("CKPT_PATH (.pt)", PathRow(self.edit_ckpt, "file", "TorchScript (*.pt);;All Files (*.*)"))
        form.addRow("TEST_IMG_DIR", PathRow(self.edit_test_img_dir, "dir"))
        form.addRow("OUT_DIR", PathRow(self.edit_out_dir, "dir"))
        form.addRow("SCORE_THRESH", self.spin_thresh)

        layout.addWidget(box)

        info = QLabel("추론 결과는 OUT_DIR 아래 DETECTED / MISSED 폴더로 자동 분리 저장됩니다.")
        info.setStyleSheet("color: #666;")
        layout.addWidget(info)
        layout.addStretch(1)

    def _build_run_tab(self):
        layout = QVBoxLayout(self.run_tab)

        btn_box = QGroupBox("실행")
        grid = QGridLayout(btn_box)

        self.btn_train = QPushButton("학습 실행")
        self.btn_train.setMinimumHeight(42)
        self.btn_infer = QPushButton("추론 실행")
        self.btn_infer.setMinimumHeight(42)
        self.btn_stop = QPushButton("실행 중지")
        self.btn_stop.setMinimumHeight(42)
        self.btn_stop.setEnabled(False)

        grid.addWidget(self.btn_train, 0, 0)
        grid.addWidget(self.btn_infer, 0, 1)
        grid.addWidget(self.btn_stop, 0, 2)

        layout.addWidget(btn_box)

        tb_box = QGroupBox("TensorBoard")
        grid2 = QGridLayout(tb_box)

        self.spin_tb_port = QSpinBox()
        self.spin_tb_port.setRange(1, 65535)
        self.spin_tb_port.setValue(6007)

        self.btn_tb_scalar = QPushButton("Scalar TensorBoard 실행")
        self.btn_tb_profiler = QPushButton("Profiler TensorBoard 실행")
        self.btn_open_scalar_dir = QPushButton("Scalar 로그 폴더")
        self.btn_open_profiler_dir = QPushButton("Profiler 로그 폴더")
        self.btn_open_results = QPushButton("추론 결과 폴더")
        self.btn_open_checkpoints = QPushButton("체크포인트 폴더")

        grid2.addWidget(QLabel("TensorBoard Port"), 0, 0)
        grid2.addWidget(self.spin_tb_port, 0, 1)
        grid2.addWidget(self.btn_tb_scalar, 1, 0, 1, 2)
        grid2.addWidget(self.btn_tb_profiler, 1, 2, 1, 2)
        grid2.addWidget(self.btn_open_scalar_dir, 2, 0)
        grid2.addWidget(self.btn_open_profiler_dir, 2, 1)
        grid2.addWidget(self.btn_open_results, 2, 2)
        grid2.addWidget(self.btn_open_checkpoints, 2, 3)

        layout.addWidget(tb_box)

        guide = QLabel(
            "Profiler 탭이 TensorBoard에 안 보이면 현재 Python 환경에 torch-tb-profiler가 설치되어 있는지 확인하세요.\n"
            "설치 예: python -m pip install torch-tb-profiler"
        )
        guide.setStyleSheet("color: #666;")
        layout.addWidget(guide)
        layout.addStretch(1)

    def _connect_signals(self):
        self.btn_train.clicked.connect(self.start_train)
        self.btn_infer.clicked.connect(self.start_infer)
        self.btn_stop.clicked.connect(self.stop_process)
        self.btn_clear_log.clicked.connect(self.console.clear)

        self.btn_tb_scalar.clicked.connect(lambda: self.start_tensorboard(self.edit_scalar_log.text()))
        self.btn_tb_profiler.clicked.connect(lambda: self.start_tensorboard(self.edit_profiler_log.text()))
        self.btn_open_scalar_dir.clicked.connect(lambda: self.open_path(self.edit_scalar_log.text()))
        self.btn_open_profiler_dir.clicked.connect(lambda: self.open_path(self.edit_profiler_log.text()))
        self.btn_open_results.clicked.connect(lambda: self.open_path(self.edit_out_dir.text()))
        self.btn_open_checkpoints.clicked.connect(self.open_checkpoints_dir)

        self.edit_data_root.textChanged.connect(self._maybe_sync_dataset_paths)

    def _load_settings(self):
        widgets = [
            ("python", self.edit_python),
            ("train_script", self.edit_train_script),
            ("infer_script", self.edit_infer_script),
            ("data_root", self.edit_data_root),
            ("img_dir", self.edit_img_dir),
            ("ann_dir", self.edit_ann_dir),
            ("scalar_log", self.edit_scalar_log),
            ("profiler_log", self.edit_profiler_log),
            ("ckpt", self.edit_ckpt),
            ("test_img_dir", self.edit_test_img_dir),
            ("out_dir", self.edit_out_dir),
        ]
        for key, edit in widgets:
            val = self.settings.value(key, "")
            if val:
                edit.setText(str(val))

        classes = self.settings.value("classes", "")
        if classes:
            self.edit_classes.setPlainText(str(classes))

        spin_pairs = [
            ("epochs", self.spin_epochs),
            ("train_batch", self.spin_train_batch),
            ("valid_batch", self.spin_valid_batch),
            ("workers", self.spin_workers),
            ("prof_wait", self.spin_prof_wait),
            ("prof_warmup", self.spin_prof_warmup),
            ("prof_active", self.spin_prof_active),
            ("prof_repeat", self.spin_prof_repeat),
            ("tb_port", self.spin_tb_port),
        ]
        for key, spin in spin_pairs:
            val = self.settings.value(key, None)
            if val is not None:
                spin.setValue(int(val))

        val = self.settings.value("score_thresh", None)
        if val is not None:
            self.spin_thresh.setValue(float(val))

        for key, chk in [
            ("enable_profiler", self.chk_enable_profiler),
            ("profile_first", self.chk_profile_first),
        ]:
            val = self.settings.value(key, None)
            if val is not None:
                chk.setChecked(str(val).lower() in ("1", "true", "yes"))

    def _save_settings(self):
        widgets = [
            ("python", self.edit_python),
            ("train_script", self.edit_train_script),
            ("infer_script", self.edit_infer_script),
            ("data_root", self.edit_data_root),
            ("img_dir", self.edit_img_dir),
            ("ann_dir", self.edit_ann_dir),
            ("scalar_log", self.edit_scalar_log),
            ("profiler_log", self.edit_profiler_log),
            ("ckpt", self.edit_ckpt),
            ("test_img_dir", self.edit_test_img_dir),
            ("out_dir", self.edit_out_dir),
        ]
        for key, edit in widgets:
            self.settings.setValue(key, edit.text())

        self.settings.setValue("classes", self.edit_classes.toPlainText())

        spin_pairs = [
            ("epochs", self.spin_epochs),
            ("train_batch", self.spin_train_batch),
            ("valid_batch", self.spin_valid_batch),
            ("workers", self.spin_workers),
            ("prof_wait", self.spin_prof_wait),
            ("prof_warmup", self.spin_prof_warmup),
            ("prof_active", self.spin_prof_active),
            ("prof_repeat", self.spin_prof_repeat),
            ("tb_port", self.spin_tb_port),
        ]
        for key, spin in spin_pairs:
            self.settings.setValue(key, spin.value())

        self.settings.setValue("score_thresh", self.spin_thresh.value())
        self.settings.setValue("enable_profiler", self.chk_enable_profiler.isChecked())
        self.settings.setValue("profile_first", self.chk_profile_first.isChecked())

    
    def _maybe_sync_dataset_paths(self):
        root = self.edit_data_root.text().strip()
        if not root:
            return
        
        img_default = os.path.join(root, "image")
        xml_default = os.path.join(root, "xml")

        if not self.edit_img_dir.text().strip() or self.edit_img_dir.text().endswith(os.path.join("annos", "image")):
            self.edit_img_dir.setText(img_default)
        if not self.edit_ann_dir.text().strip() or self.edit_ann_dir.text().endswith(os.path.join("annos", "xml")):
            self.edit_ann_dir.setText(xml_default)

    def validate_common(self) -> bool:
        py = norm_path(self.edit_python.text())
        if not os.path.isfile(py):
            QMessageBox.warning(self, "확인 필요", f"Python 실행 파일이 없습니다.\n\n{py}")
            return False
        
        classes = parse_classes(self.edit_classes.toPlainText())
        if classes[0] != "__background__":
            QMessageBox.warning(self, "확인 필요", "클래스 이름 목록의 첫 번째 항목은 반드시 __background__ 여야 합니다.")
            return False
        
        return True
    
    def start_train(self):
        if self.worker is not None:
            QMessageBox.information(self, "실행 중", "이미 실행 중인 프로세스가 있습니다.")
            return
        
        if not  self.validate_common():
            return
        
        script = norm_path(self.edit_train_script.text())
        if not os.path.isfile(script):
            QMessageBox.warning(self, "확인 필요", f"학습 스크립트 파일이 없습니다.\n\n{script}")
            return

        data_root = norm_path(self.edit_data_root.text())
        img_dir = norm_path(self.edit_img_dir.text())
        ann_dir = norm_path(self.edit_ann_dir.text())

        if not os.path.isdir(img_dir):
            QMessageBox.warning(self, "확인 필요", f"이미지 폴더가 없습니다.\n\n{img_dir}")
            return
        if not os.path.isdir(ann_dir):
            QMessageBox.warning(self, "확인 필요", f"XML 폴더가 없습니다.\n\n{ann_dir}")
            return
        
        overrides = {
            "DATA_ROOT": data_root,
            "IMG_DIR": img_dir,
            "ANN_DIR": ann_dir,
            "CLASS_NAMES": parse_classes(self.edit_classes.toPlainText()),
            "NUM_EPOCHS": int(self.spin_epochs.value()),
            "TRAIN_BATCH_SIZE": int(self.spin_train_batch.value()),
            "VALID_BATCH_SIZE": int(self.spin_valid_batch.value()),
            "NUM_WORKERS": int(self.spin_workers.value()),
            "SCALAR_LOG_DIR": norm_path(self.edit_scalar_log.text()),
            "PROFILER_LOG_DIR": norm_path(self.edit_profiler_log.text()),
            "PROFILER_WAIT": int(self.spin_prof_wait.value()),
            "PROFILER_WARMUP": int(self.spin_prof_warmup.value()),
            "PROFILER_ACTIVE": int(self.spin_prof_active.value()),
            "PROFILER_REPEAT": int(self.spin_prof_repeat.value()),
            "ENABLE_PROFILER": bool(self.chk_enable_profiler.isChecked()),
            "PROFILE_ONLY_FIRST_EPOCH": bool(self.chk_profile_first.isChecked()),
        }

        os.makedirs(overrides["SCALAR_LOG_DIR"], exist_ok=True)
        os.makedirs(overrides["PROFILER_LOG_DIR"], exist_ok=True)

        self.total_epochs = int(self.spin_epochs.value())
        self.progress.setValue(0)
        self.current_mode = "train"

        self.console.appendPlainText("\n\n" + "#" * 80)
        self.console.appendPlainText("[TRAIN START]")
        self.console.appendPlainText("#" * 80 + "\n")

        self._save_settings()
        self._start_worker(script, "train", overrides)


    def start_infer(self):
        if self.worker is not None:
            QMessageBox.information(self, "실행 중", "이미 실행 중인 프로세스가 있습니다.")
            return
        
        if not self.validate_common():
            return
        
        script = norm_path(self.edit_infer_script.text())
        if not os.path.isfile(script):
            QMessageBox.warning(self, "확인 필요", f"추론 스크립트 파일이 없습니다.\n\n{script}")
            return
        
        ckpt = norm_path(self.edit_ckpt.text())
        test_img_dir = norm_path(self.edit_test_img_dir.text())
        out_dir = norm_path(self.edit_out_dir.text())

        if not os.path.isfile(ckpt):
            QMessageBox.warning(self, "확인 필요", f".pt 모델 파일이 없습니다.\n\n{ckpt}")
            return
        if not os.path.isdir(test_img_dir):
            QMessageBox.warning(self, "확인 필요", f"테스트 이미지 폴더가 없습니다.\n\n{test_img_dir}")
            return

        overrides = {
            "CLASS_NAMES": parse_classes(self.edit_classes.toPlainText()),
            "CKPT_PATH": ckpt,
            "TEST_IMG_DIR": test_img_dir,
            "OUT_DIR": out_dir,
            "SCORE_THRESH": float(self.spin_thresh.value()),
        }

        os.makedirs(out_dir, exist_ok=True)

        self.total_infer_images = 0
        self.done_infer_images = 0
        self.progress.setValue(0)
        self.current_mode = "infer"

        self.console.appendPlainText("\n\n" + "#" * 80)
        self.console.appendPlainText("[INFERENCE START]")
        self.console.appendPlainText("#" * 80 + "\n")

        self._save_settings()
        self._start_worker(script, "infer", overrides)

    def _start_worker(self, script: str, mode: str, overrides: dict):
        self.set_running_state(True)

        self.worker = ProcessWorker(
            python_exe=norm_path(self.edit_python.text()),
            script_path=script,
            mode=mode,
            overrides=overrides,
        )
        self.worker.log.connect(self.on_log)
        self.worker.started_pid.connect(lambda pid: self.on_log(f"[PID] {pid}\n"))
        self.worker.finished_code.connect(self.on_finished)
        self.worker.start()

    def stop_process(self):
        if self.worker:
            self.worker.stop()

    def on_finished(self, code: int):
        self.on_log(f"\n[PROCESS FINISHED] exit code = {code}\n")
        if code == 0:
            self.progress.setValue(100)

        self.worker = None
        self.current_mode = None
        self.set_running_state(False)

    def set_running_state(self, running: bool):
        self.btn_train.setEnabled(not running)
        self.btn_infer.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def on_log(self, text: str):
        self.console.moveCursor(QTextCursor.MoveOperation.End)
        self.console.insertPlainText(text)
        self.console.moveCursor(QTextCursor.MoveOperation.End)
        self._update_progress_from_log(text)

    def _update_progress_from_log(self, text: str):
        if self.current_mode == "train":
            # 에폭 진행 상황 로그에서 현재 에폭과 총 에폭을 파싱하여 진행률 업데이트
            m = re.search(r"\[Epoch\s+(\d+)\]\s+Step\s+(\d+)/(\d+)", text)
            if m:
                epoch = int(m.group(1))
                step = int(m.group(2))
                total_step = max(1, int(m.group(3)))
                total_epochs = max(1, self.total_epochs)
                pct = ((epoch -1) + (step / total_step)) / total_epochs * 100
                self.progress.setValue(max(0, min(100, int(pct))))
                return
            
            # ex: [Epoch 3] train_loss...
            m = re.search(r"\[Epoch\s+(\d+)\].*train_loss", text)
            if m:
                epoch = int(m.group(1))
                total_epochs = max(1, self.total_epochs)
                pct = epoch / total_epochs * 100
                self.progress.setValue(max(0, min(100, int(pct))))
                return
            
        elif self.current_mode == "infer":
            # ex: [INFO] found 120 images
            m = re.search(r"\[INFO\]\s+found\s+(\d+)\s+images", text)
            if m:
                self.total_infer_images = int(m.group(1))
                self.done_infer_images = 0
                self.progress.setValue(0)
                return
            
            if "[SAVED]" in text:
                self.done_infer_images += 1
                if self.total_infer_images > 0:
                    pct = self.done_infer_images / self.total_infer_images * 100
                    self.progress.setValue(max(0, min(100, int(pct))))
                return
            
            if "[DONE]" in text:
                self.progress.setValue(100)

    def start_tensorboard(self, logdir_text: str):
        logdir = norm_path(logdir_text)
        if not logdir:
            QMessageBox.warning(self, "확인 필요", "TensorBoard logdir가 비어 있습니다.")
            return
        
        os.makedirs(logdir, exist_ok=True)

        py= norm_path(self.edit_python.text())
        port = int(self.spin_tb_port.value())
        url = f"http://localhost:{port}/"

        if self.tensorboard_proc and self.tensorboard_proc.poll() is None:
            self.on_log(f"[TensorBoard] already running -> {url}\n")
            webbrowser.open(url)
            return
        
        cmd_candidates = [
            [py, "-m", "tensorboard.main", "--logdir", logdir, "--port", str(port), "--reload_interval", "5"],
            [py, "-m", "tensorboard", "--logdir", logdir, "--port", str(port), "--reload_interval", "5"],
            ["tensorboard", "--logdir", logdir, "--port", str(port), "--reload_interval", "5"],
        ]

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        last_error = None
        for cmd in cmd_candidates:
            try:
                self.on_log(f"[TensorBoard CMD] {' '.join(cmd)}\n")
                self.tensorboard_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                QTimer.singleShot(1800, lambda: webbrowser.open(url))
                self.on_log(f"[TensorBoard] opening {url}\n")
                return
            except Exception as e:
                last_error = e

        QMessageBox.warning(self, "TensorBoard 실행 실패", f"TensorBoard 실행에 실패했습니다.\n\n{last_error}")

    def open_path(self, path_text: str):
        path = norm_path(path_text)
        if not path:
            return
        
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
            except Exception:
                QMessageBox.warning(self, "폴더 열기 실패", f"경로가 없습니다:\n\n{path}")
                return
            
        try:
            if os.name == "nt":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            QMessageBox.warning(self, "열기 실패", str(e))

    def open_checkpoints_dir(self):
        train_script = norm_path(self.edit_train_script.text())
        base_dir = os.path.dirname(train_script) if train_script else DEFAULT_BASE
        self.open_path(os.path.join(base_dir, "checkpoints"))

    def closeEvent(self, event):
        self._save_settings()
        if self.worker is not None:
            reply = QMessageBox.question(
                self,
                "종료 확인",
                "실행 중인 학습/추론 프로세스가 있습니다. 종료할까요?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                event.accept()
            else:
                event.ignore()
                return

        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationName(APP_NAME)

    win = MainWindow()
    win.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()