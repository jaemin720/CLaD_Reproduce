# Stage 3: EMA policy loading and LIBERO evaluation

이 단계는 학습이 끝난 Stage 2 checkpoint에서 EMA policy를 복원하고,
고정된 LIBERO initial state로 rollout하여 task별 성공률을 계산한다.

## 구현 범위

- Stage 2 full trainer checkpoint를 memory-map하여 optimizer state를 GPU에
  올리지 않고 EMA 또는 raw policy weight만 복원한다.
- checkpoint에 기록된 Stage 1 foresight SHA-256과 실제
  `stage1_foresight.pt`를 대조한다.
- 학습 cache manifest와 로컬 DecisionNCE checkpoint SHA-256을 대조하고,
  task text feature는 cache에 저장된 값을 그대로 사용한다.
- 환경의 live RGB frame만 DecisionNCE로 encode한다.
- 학습 데이터의 9D `robot_states`와 동일하게
  `gripper qpos(2) + EEF position(3) + EEF quaternion(4)`를 구성한다.
- 초기 history는 최초 관측 반복과 zero action으로 왼쪽 padding한다.
- 6-action DDPM chunk를 생성하고 configurable한 개수만큼 실행한 뒤
  다시 계획한다.
- episode 결과를 즉시 JSONL에 append하여 중단된 평가를 재개한다.

기본 설정은 [`configs/eval/libero_long.yaml`](../configs/eval/libero_long.yaml)에
있다. 논문의 single-checkpoint 표기(`‡`)에서 직접 확인되는 값은 task당
50 rollout이다. 최대 600 policy step과 128x128 agent view는 논문에 적혀 있지
않은 이 재현의 기본값이다. fixed initial state와 초기 5회의 zero action은
공식 LIBERO benchmark API 및 평가 관행을 따른다. initial-state 순서와 seed도
논문이 공개한 설정은 아니다.

논문은 6개 action을 생성한다고 설명하지만 몇 개를 실행한 뒤 다시
계획하는지는 밝히지 않는다. 이 구현은 기본적으로 6개 전체를 실행한다.
매 control step마다 다시 계획하는 실험은 `--execution-steps 1`로 수행할 수
있다.

## LIBERO runtime 준비

충돌을 피하는 설치 원칙, 새 환경/기존 환경별 명령, 검증된 버전 및 문제
해결은 [`docs/libero_installation.md`](libero_installation.md)를 기준으로 한다.
아래는 이미 `clad` 환경이 있는 경우의 요약이다.

LIBERO는 공식 저장소를 `third_party/LIBERO` Git submodule로 고정한다.
부모 프로젝트의 Apache-2.0 소스와 LIBERO의 MIT 소스는 서로 다른 경로와
라이선스를 유지한다. submodule이 없는 clone은 먼저 초기화한다.

```bash
git submodule update --init --recursive
```

공식 LIBERO의 `requirements.txt`는 오래된 NumPy, Hydra, WandB 버전을
정확히 pin하고 있으므로 현재 학습 환경에 그대로 설치하면 기존 패키지를
downgrade할 수 있다. 따라서 그 파일은 직접 설치하지 않는다. 현재
`environment.yml`과 `.[eval]`은 필요한 robosuite 1.4 API, MuJoCo 2.3 및
환경 runtime만 추가하고 기존 NumPy/Hydra/OpenCV/WandB 범위를 유지한다.

기존 `clad` 환경에는 평가 extra와 submodule만 editable 설치한다. upstream
LIBERO의 implicit namespace layout은 최신 setuptools의 기본 PEP 660 finder가
발견하지 못하므로 LIBERO 설치에만 `editable_mode=compat`를 사용한다.

```bash
pip install -e ".[eval]"
pip install -C editable_mode=compat -e ./third_party/LIBERO
```

LIBERO의 upstream import는 설정 파일이 없으면 터미널 입력을 요구한다.
평가 프로세스가 예기치 않게 멈추거나 `~/.libero`를 바꾸지 않도록, 이
프로젝트는 gitignore된 `.cache/libero/config.yaml`을 비대화형으로 만든다.
`--dataset-root`는 `libero_10` 자체가 아니라 이를 포함하는 부모 경로다.

```bash
python scripts/configure_libero.py \
  --dataset-root /path/to/libero_datasets
```

같은 설정 스크립트는 robosuite의 공식 `macros_private.py` hook을 이용해
hard-coded `/tmp/robosuite.log` 파일 기록만 끈다. 기존 사용자 private macro가
있으면 덮어쓰지 않고 오류를 내며, console/MuJoCo/GPU 설정은 변경하지 않는다.
이 동작이 필요하지 않다면 `--no-configure-robosuite-logging`을 사용한다.

그 다음 설정 경로를 명시하여 runtime import를 확인한다.

```bash
MUJOCO_GL=glx LIBERO_CONFIG_PATH=.cache/libero python - <<'PY'
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

print(sorted(benchmark.get_benchmark_dict()))
print(OffScreenRenderEnv)
PY
```

평가 CLI는 기본적으로 `.cache/libero`를 사용하며 설정 파일이 없으면 LIBERO를
import하기 전에 명확한 오류를 낸다. 다른 위치는 `--libero-config-dir` 또는
`LIBERO_CONFIG_PATH`로 지정할 수 있다.

## checkpoint 검증

학습과 같은 GPU에 동시에 대형 평가 모델을 올리면 OOM이 발생할 수 있다.
따라서 아래 명령은 Stage 2 학습이 끝난 뒤 실행한다.

```bash
python scripts/evaluate_clad_libero.py \
  --checkpoint outputs/clad_stage2/stage2_latest.pt \
  --foresight-checkpoint outputs/clad_stage1/stage1_foresight.pt \
  --device cuda \
  --checkpoint-only
```

기본값은 EMA weight이다. 비교 목적으로 online training weight를 확인할
때만 `--weights raw`를 사용한다.

## 평가 shell launcher

반복 실행에는 `scripts/evaluate_libero.sh`를 사용한다. 첫 번째 인자는 host에서
보이는 물리 GPU 번호이며 필수다. 예를 들어 GPU 1에서 기본 논문 프로토콜
(10 tasks × 50 rollouts)을 실행하려면 다음과 같이 입력한다.

```bash
./scripts/evaluate_libero.sh 1
```

launcher는 `CUDA_VISIBLE_DEVICES=1`로 해당 GPU만 노출하고 Python 평가
프로세스에는 `--device cuda:0`을 전달한다. 따라서 첫 번째 인자가 1이어도
PyTorch 내부 장치 번호가 `cuda:0`으로 출력되는 것이 정상이다. 콘솔 출력은
`outputs/clad_evaluation/eval_console.log`에도 누적된다.

다른 checkpoint나 출력 경로를 사용할 때는 다음 환경변수를 명령 앞에
지정한다.

```text
CLAD_STAGE2_CHECKPOINT
CLAD_FORESIGHT_CHECKPOINT
CLAD_DECISIONNCE_CACHE_DIR
CLAD_LIBERO_CONFIG_DIR
CLAD_EVAL_OUTPUT_DIR
CLAD_PYTHON
```

서로 다른 checkpoint는 반드시 서로 다른 `CLAD_EVAL_OUTPUT_DIR`를 사용해야
한다. evaluator가 결과 디렉터리의 checkpoint hash를 검증하므로 다른 모델의
episode 결과가 섞이지 않는다.

## rollout smoke test

먼저 task 0의 한 initial state에 대해 짧게 실행한다.

```bash
CLAD_EVAL_OUTPUT_DIR=outputs/clad_evaluation_smoke \
./scripts/evaluate_libero.sh 1 \
  --task-ids 0 \
  --rollouts-per-task 1 \
  --max-steps 30 \
  --save-videos
```

확인할 항목은 다음과 같다.

- EMA/global step과 두 checkpoint SHA-256이 출력되는가;
- DecisionNCE checkpoint가 학습 cache manifest와 일치하는가;
- observation key 및 9D proprioception 오류가 없는가;
- sampled action에 NaN/Inf가 없는가;
- 30 step 이내 성공 여부와 관계없이 episode record가 저장되는가.

## 50-rollout 본 평가

기본 config가 10개 task 각각 50 rollout을 지정하므로 override 없이
실행한다.

```bash
./scripts/evaluate_libero.sh 1
```

같은 명령을 다시 실행하면 이미 완료된 `(task_id, rollout_id)`는 건너뛴다.
평가 정체성과 다른 checkpoint/config를 같은 output directory에 섞으려 하면
오류를 낸다.

출력은 다음과 같다.

```text
outputs/clad_evaluation/
├── run_identity.json       # checkpoint/cache hashes and resolved protocol
├── episode_results.jsonl   # crash-safe per-episode records
├── summary.json            # task SR and macro average SR
└── videos/                 # --save-videos 사용 시
```

논문의 다른 표기(`†`)는 top-3 checkpoint 각각의 20-rollout 결과를 평균한다.
그 프로토콜을 사용하려면 서로 다른 checkpoint snapshot을 각각 별도 output
directory에서 `--rollouts-per-task 20`으로 평가해야 한다. 현재 trainer의
`stage2_latest.pt`는 원자적으로 덮어쓰는 resume checkpoint이므로, top-3
선정을 계획한다면 원하는 학습 step의 파일을 별도 이름으로 보존해야 한다.

## Camera extension

기본 cache가 `agentview_rgb` 하나만 포함하므로 live key
`agentview_image` 하나만 사용한다. 이후 두 카메라 cache로 다시 학습한
경우 다음처럼 명시적으로 mapping할 수 있다.

```bash
python scripts/evaluate_clad_libero.py \
  ... \
  --camera agentview_rgb=agentview_image \
  --camera eye_in_hand_rgb=robot0_eye_in_hand_image
```

mapping의 view 집합이 학습 cache의 view 집합과 정확히 같지 않으면 평가를
거부하여 train/eval camera mismatch를 방지한다.
