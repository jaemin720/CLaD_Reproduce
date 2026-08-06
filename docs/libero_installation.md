# CLaD용 LIBERO 설치 가이드

이 문서는 CLaD의 기존 학습 환경을 유지하면서 공식 LIBERO runtime을
설치하는 기준 절차를 설명한다. LIBERO 자체의 원본 환경을 재현하려는 경우가
아니라, 이 저장소의 Stage 3/4 rollout 평가를 실행하는 경우를 대상으로 한다.

## 환경 파일의 역할

재현성 파일은 용도가 서로 다르다.

| 파일 | 역할 | 사용 시점 |
| --- | --- | --- |
| `environment.yml` | 지원 범위를 표현하는 portable conda/pip 명세 | 일반 개발, 다른 Linux host로 이식 |
| `pyproject.toml` | CLaD base/dev/train/eval Python 직접 의존성 | editable 개발 및 기존 환경 확장 |
| `locks/conda-linux-64-explicit.txt` | 검증 host의 conda binary artifact 208개 정확한 URL | Linux-64 exact 재현 |
| `locks/pip-linux-py310.constraints.txt` | 설치된 Python distribution 116개의 정확한 버전 | Python 3.10 exact resolver 제한 |
| `.gitmodules` | DecisionNCE와 LIBERO source revision 고정 | source/라이선스 재현 |

일반 사용자는 `environment.yml`을 사용한다. 논문 결과를 같은 software
resolution으로 재현하려는 연구자는 두 lock 파일을 함께 사용한다. conda
explicit lock은 Linux x86-64 전용이며 macOS/Windows에는 사용할 수 없다.
pip lock은 constraints이므로 단독 `pip install -r` 대상으로 사용하지 않는다.

## 핵심 원칙

다음 명령은 CLaD 환경에서 실행하지 않는다.

```bash
pip install -r third_party/LIBERO/requirements.txt
```

고정한 LIBERO revision의 공식 설치 지침은 Python 3.8.13, PyTorch 1.11,
CUDA 11.3을 기준으로 작성되었다. `requirements.txt` 역시 NumPy 1.22.4,
Hydra 1.2.0, WandB 0.13.1, OpenCV 4.6.0.66, matplotlib 3.5.3 등을 정확히
고정한다. 이를 CLaD 환경에 그대로 설치하면 현재 PyTorch 2.2/CUDA 12.1
학습 stack의 패키지를 downgrade하거나 resolver 충돌을 일으킬 수 있다.

이 프로젝트는 LIBERO의 lifelong-learning trainer 전체를 사용하지 않는다.
평가에 필요한 benchmark registry, BDDL, robosuite/MuJoCo environment와
영상 writer만 설치한다. `transformers`, `robomimic`, `thop` 등 upstream
trainer 전용 패키지는 추가하지 않는다.

## 검증된 호환 기준

현재 조합은 Python 3.10 및 다음 핵심 버전에서 실제 rollout까지 검증했다.

### 검증 host와 binary stack

| 항목 | 실제 검증 값 |
| --- | --- |
| OS / architecture | Ubuntu 20.04.6 LTS / Linux x86-64 |
| Kernel / glibc | Linux 5.15 series / glibc 2.31 |
| Python | 3.10.20, conda-forge build |
| conda / pip | 25.11.0 / 26.2 |
| PyTorch / TorchVision | 2.2.2 / 0.17.2 |
| PyTorch CUDA runtime | 12.1 |
| cuDNN | 8.9.2 (`torch.backends.cudnn.version() == 8902`) |
| HDF5 / NumPy / PyYAML | h5py 3.11.0 / NumPy 1.26.4 / PyYAML 6.0.3 |

GPU model과 NVIDIA driver는 repository artifact에 고정하지 않았다. 검증
host의 별도 GPU에서 EGL 600-step rollout을 실제로 완료했으며, 각 연구자는
자신의 실행 host에서 다음 결과를 run artifact로 보존해야 한다.

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("available:", torch.cuda.is_available())
print("devices:", torch.cuda.device_count())
PY
```

NVIDIA driver는 CUDA 12.1과 EGL device rendering을 지원해야 한다. CPU-only
rollout, OSMesa, macOS 및 Windows는 현재 검증 범위가 아니다.

### 직접 의존성과 실제 선택 버전

| 구성요소 | CLaD 기준 | 선택 이유 |
| --- | --- | --- |
| PyTorch / TorchVision | 2.2.2 / 0.17.2 | 기존 Stage 1/2 CUDA 12.1 학습 환경 유지 |
| NumPy | 1.26.4 | 기존 cache/trainer 환경 유지, NumPy 2 제외 |
| setuptools | 80.9.0 | DecisionNCE의 `openai-clip`이 `pkg_resources` 사용 |
| robosuite | 1.4.0 | LIBERO가 1.4의 `SingleArmEnv`/`SingleArm` API 사용 |
| MuJoCo | 2.3.7 | robosuite 1.4 API와 검증된 DeepMind binding |
| BDDL | 1.0.1 | LIBERO BDDL parser의 upstream 고정 버전 |
| Gym | 0.25.2 | LIBERO vector environment와 동일한 API |
| matplotlib | `>=3.5,<4` | LIBERO environment의 직접 runtime import |
| termcolor | `>=2,<4` | robosuite 1.4 logger의 암묵적 runtime import |

CLaD 학습 및 DecisionNCE에 실제 선택된 주요 버전도 다음과 같다.

| 영역 | package==version |
| --- | --- |
| 학습 | `einops==0.8.2`, `hydra-core==1.3.4`, `omegaconf==2.3.1`, `tqdm==4.70.0`, `wandb==0.28.1` |
| 개발/검증 | `pytest==8.4.2`, `ruff==0.16.1` |
| DecisionNCE | `timm==0.9.12`, `mmengine==0.10.7`, `tensorboardX==2.6.5`, `gdown==6.1.0`, `openai-clip==1.0.1`, `chardet==7.4.3` |
| LIBERO 직접 runtime | `bddl==1.0.1`, `cloudpickle==3.1.2`, `easydict==1.9`, `future==0.18.2`, `gym==0.25.2`, `imageio==2.37.4`, `imageio-ffmpeg==0.6.0`, `matplotlib==3.10.9`, `mujoco==2.3.7`, `robosuite==1.4.0`, `termcolor==3.3.0` |
| MuJoCo 전이 | `absl-py==2.5.0`, `glfw==2.10.2`, `PyOpenGL==3.1.10` |
| robosuite 전이 | `numba==0.66.0`, `llvmlite==0.48.0`, `scipy==1.15.3`, `Pillow==12.3.0`, `opencv-python==4.11.0.86` |
| BDDL 전이 | `jupytext==1.19.5`, `nbformat==5.10.4`, `jsonschema==4.26.0`, `networkx==3.4.2` |

### Source와 model identity

| 항목 | 고정 identity |
| --- | --- |
| DecisionNCE source | `ebdc585c5e6833ec3a2ba77f801b15c74d7a28f8` |
| LIBERO source | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| DecisionNCE variant | `DecisionNCE-T` |
| 검증된 DecisionNCE-T checkpoint SHA-256 | `75242de7e60c4007b186ddcdde220f6ca981e814775df564673af7782afcd03f` |

DecisionNCE checkpoint는 저장소가 배포하지 않으며 upstream cache에서 별도로
준비한다. CLaD parent source는 각 실험에서 `git rev-parse HEAD`로 기록한다.
dirty worktree에서 수행한 결과는 commit hash만으로 재현할 수 없으므로 정식
실험 전에 `git status --short`가 비어 있는지도 함께 확인한다.

표에 생략된 HTTP, JSON schema, plotting 및 notebook 하위 의존성까지 포함한
116개 전체 exact version은
[`locks/pip-linux-py310.constraints.txt`](../locks/pip-linux-py310.constraints.txt)에
기계 판독 가능한 형태로 기록한다. conda의 CUDA/codec/X11/system library까지
포함한 전체 binary resolution은
[`locks/conda-linux-64-explicit.txt`](../locks/conda-linux-64-explicit.txt)에
기록한다.

정확한 선언은 루트의 `environment.yml`과 `pyproject.toml`의 `eval` extra가
단일 기준이다. LIBERO revision 또는 robosuite major/minor를 변경할 때는
이 표를 그대로 신뢰하지 말고 import와 실제 environment smoke test를 다시
수행해야 한다.

## 1. Submodule 준비

새 clone은 submodule을 함께 받는다.

```bash
git clone --recurse-submodules https://github.com/jaemin720/CLaD_Reproduce.git
cd CLaD_Reproduce
```

이미 clone한 저장소라면 다음 명령으로 초기화한다.

```bash
git submodule update --init --recursive
git submodule status
```

정상적인 LIBERO gitlink는 다음 revision을 표시한다.

```text
8f1084e3132a39270c3a13ebe37270a43ece2a01 third_party/LIBERO
```

## 2A. 새 conda 환경 생성

`environment.yml`은 CLaD 학습 stack, DecisionNCE, 최소 LIBERO runtime과
두 submodule의 editable 설치를 포함한다. 저장소 루트에서 실행한다.

```bash
conda env create -f environment.yml
conda activate clad
```

LIBERO의 upstream `setup.py`는 implicit namespace layout을 사용한다. 최신
setuptools의 기본 PEP 660 editable finder는 이 package를 빈 mapping으로
설치할 수 있으므로 `environment.yml`은 LIBERO에만
`editable_mode=compat`를 지정한다.

환경 생성 직후 다음 항목을 확인한다.

```bash
test "$CONDA_DEFAULT_ENV" = clad
python -VV
python -m pip --version
python -m pip check
git submodule status
```

## 2B. 기존 `clad` 환경을 보존하며 추가

먼저 핵심 버전을 기록하고 실제 설치 없이 resolver 결과를 확인한다.

```bash
conda activate clad
python -m pip show torch torchvision numpy hydra-core wandb opencv-python setuptools
python -m pip install --dry-run -e ".[eval]"
```

출력에 기존 핵심 패키지의 `Would uninstall`, downgrade 또는 예상하지 않은
교체가 보이면 설치하지 않고 원인을 먼저 확인한다. 정상이라면 평가 extra와
LIBERO source를 별도 단계로 설치한다.

```bash
python -m pip install --upgrade-strategy only-if-needed -e ".[eval]"
python -m pip install --upgrade-strategy only-if-needed \
  --config-settings editable_mode=compat \
  -e ./third_party/LIBERO
```

두 번째 명령에 `editable_mode=compat`가 없으면 `pip show libero`는 성공해도
`import libero`가 `ModuleNotFoundError`를 낼 수 있다. upstream source를
수정하는 대신 setuptools가 제공하는 호환 editable mode를 사용한다.

이 저장소에서 실제 설치 전후를 비교했을 때 패키지 28개가 추가되었고,
제거된 패키지와 기존 버전 변경은 모두 0개였다. BDDL의 metadata 때문에
`jupytext`와 `nbformat` 계열이 함께 설치되는 것은 정상이다.

## 2C. 검증 환경과 동일한 exact Linux-64 설치

portable 범위가 아니라 검증된 artifact/version resolution을 그대로 재현하려면
별도 환경 이름을 사용한다. 이 방법은 Linux x86-64 전용이다.

```bash
conda create -n clad-lock --file locks/conda-linux-64-explicit.txt
conda activate clad-lock
```

그 다음 exact constraints 아래에서 CLaD와 DecisionNCE를 설치한다.

```bash
python -m pip install --upgrade-strategy only-if-needed \
  -c locks/pip-linux-py310.constraints.txt \
  -e ".[dev,train,eval]" \
  -e ./third_party/DecisionNCE

python -m pip install --upgrade-strategy only-if-needed \
  -c locks/pip-linux-py310.constraints.txt \
  --config-settings editable_mode=compat \
  -e ./third_party/LIBERO
```

constraints는 package version만 고정하며 PyPI wheel hash까지 고정하지는 않는다.
완전한 artifact-level 재현이 필요한 배포는 별도의 hashed wheelhouse 또는
container image를 보존해야 한다. 두 submodule의 source 내용은 Git gitlink가
고정한다.

## 3. LIBERO 경로와 robosuite 로그 설정

upstream LIBERO는 `config.yaml`이 없으면 최초 import 중 터미널 입력을
요구하고 기본적으로 `~/.libero`를 사용한다. CLaD는 repository-local이며
gitignore된 `.cache/libero/config.yaml`을 비대화형으로 생성한다.

`--dataset-root`에는 `libero_10` 폴더 자체가 아니라 이를 포함하는 부모
디렉터리를 전달한다.

```bash
python scripts/configure_libero.py \
  --dataset-root /path/to/libero_datasets
```

이 명령은 다음 두 작업을 수행한다.

- benchmark/assets/BDDL/init-state/dataset 절대 경로를
  `.cache/libero/config.yaml`에 기록한다.
- robosuite의 공식 `macros_private.py` hook으로 hard-coded
  `/tmp/robosuite.log` 파일 기록만 끈다.

기존 `macros_private.py`가 있으면 덮어쓰지 않고 중단한다. 기존 사용자 설정을
직접 관리하려면 `--no-configure-robosuite-logging`을 사용한다. 설정 경로를
교체할 때만 내용을 확인한 뒤 `--force`를 사용한다.

## 4. 설치 검증

먼저 GPU/EGL context를 만들지 않는 import 검사를 수행한다. robosuite 1.4는
`MUJOCO_GL=glfw`를 다시 EGL로 강제하므로 import-only 검사에는 `glx`를 쓴다.

```bash
MUJOCO_GL=glx LIBERO_CONFIG_PATH=.cache/libero python - <<'PY'
import importlib.metadata as metadata

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

print(sorted(benchmark.get_benchmark_dict()))
print(OffScreenRenderEnv)
for name in ("libero", "robosuite", "mujoco", "bddl", "numpy", "torch"):
    print(name, metadata.version(name))
PY
```

dependency metadata와 CLaD 회귀 테스트도 확인한다.

```bash
python -m pip check
pytest -q
```

exact constraints와 실제 환경 사이의 차이는 다음처럼 확인할 수 있다. 출력이
없으면 package name/version 집합이 동일하다. editable source 경로는 별도로
`pip show`와 Git revision을 확인한다.

```bash
diff -u \
  <(grep -v '^#' locks/pip-linux-py310.constraints.txt | sort -f) \
  <(python -m pip list --format=freeze | sort -f)

python -m pip show clad-reproduce DecisionNCE libero
git rev-parse HEAD
git status --short
git submodule status
```

마지막으로 별도 GPU에서 실제 EGL rollout을 짧게 실행한다. 아래 명령에서
physical GPU 1은 프로세스 내부의 `cuda:0`으로 보인다.

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \
python scripts/evaluate_clad_libero.py \
  --checkpoint outputs/clad_stage2_official/stage2_latest.pt \
  --foresight-checkpoint outputs/clad_stage1_official/stage1_foresight.pt \
  --cache-dir .cache/decisionnce/libero_long \
  --output-dir outputs/clad_evaluation_smoke \
  --device cuda:0 \
  --task-ids 0 \
  --rollouts-per-task 1 \
  --max-steps 30
```

## 5. 실험과 함께 환경 증적 보존

각 training/evaluation run에는 config/checkpoint hash뿐 아니라 software와
driver 정보를 함께 보존한다. 다음 파일은 연구 artifact용이며 repository에
자동 commit하지 않는다.

```bash
mkdir -p outputs/environment_snapshot
conda env export --no-builds > outputs/environment_snapshot/conda-environment.yml
conda list --explicit > outputs/environment_snapshot/conda-explicit.txt
python -m pip list --format=freeze > outputs/environment_snapshot/pip-freeze.txt
git rev-parse HEAD > outputs/environment_snapshot/git-head.txt
git submodule status > outputs/environment_snapshot/git-submodules.txt
nvidia-smi -q > outputs/environment_snapshot/nvidia-smi.txt
```

추가로 아래 항목을 실험 기록에 남긴다.

- dataset suite 이름, dataset root 및 HDF5 file identity;
- DecisionNCE variant, checkpoint SHA-256, feature-cache manifest SHA-256;
- Stage 1/2 checkpoint SHA-256와 global/attempt step;
- `CUDA_VISIBLE_DEVICES`, `MUJOCO_GL`, camera mapping 및 rollout config;
- 성공률 summary와 episode JSONL.

경로에는 사용자명이나 storage mount가 포함될 수 있으므로 외부 공개 전
민감한 절대 경로가 없는지 확인한다.

## 자주 발생하는 문제

### `ModuleNotFoundError: No module named 'libero'`

LIBERO가 일반 editable mode로 설치된 경우다. 다음 명령으로 LIBERO의
editable link만 다시 만든다.

```bash
python -m pip install --config-settings editable_mode=compat \
  -e ./third_party/LIBERO
```

### import 도중 dataset 경로를 묻는 입력 prompt가 표시됨

`LIBERO_CONFIG_PATH`가 준비되지 않은 것이다. `configure_libero.py`를 먼저
실행한다. 평가 CLI는 기본적으로 `.cache/libero`를 사용하며 설정이 없으면
LIBERO를 import하기 전에 오류를 낸다.

### `/tmp/robosuite.log` PermissionError

다른 사용자나 container가 만든 전역 로그 파일과 충돌한 것이다. 시스템
파일을 삭제하거나 권한을 바꾸지 말고 `configure_libero.py`로 private macro를
설정한다.

### `Cannot initialize a EGL device display`

GPU가 없는 import 검사라면 `MUJOCO_GL=glx`를 사용한다. 실제 rollout에서는
EGL을 지원하는 GPU/driver가 필요하며 `CUDA_VISIBLE_DEVICES`와
`MUJOCO_GL=egl`을 지정한다.

### Gym 또는 `pkg_resources` deprecation 경고

현재 고정된 upstream 코드에서 예상되는 경고다. Gym을 임의로 Gymnasium으로
교체하거나 setuptools를 81 이상으로 올리면 각각 LIBERO API와 DecisionNCE
`openai-clip` import가 깨질 수 있다. 검증된 dependency를 유지한다.

### `SingleArmEnv` 또는 `robosuite.robots.single_arm` import 오류

robosuite 1.5 계열을 설치했을 가능성이 높다. 현재 LIBERO revision은 1.4 API를
직접 import하므로 `robosuite==1.4.0`을 사용한다.

### exact lock이 다른 OS 또는 architecture에서 설치되지 않음

정상적인 제한이다. conda explicit lock에는 Linux-64 artifact URL이 들어 있다.
다른 platform에서는 `environment.yml`로 새 portable resolution을 만들고 실제
import/rollout 검증 후 해당 platform 전용 lock을 별도 이름으로 생성한다.

## 라이선스와 저장소 관리

LIBERO source는 `third_party/LIBERO` submodule 내부에서 MIT 라이선스를
유지한다. 부모 CLaD source의 Apache-2.0 라이선스로 재허가하지 않는다.
고정 revision과 저작권은 `THIRD_PARTY_NOTICES.md` 및
`LICENSES/LIBERO-MIT.txt`에 별도로 보존한다. LIBERO dataset과 model
checkpoint는 이 저장소에 commit하거나 재배포하지 않는다.
