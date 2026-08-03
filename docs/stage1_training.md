# Stage 1 학습 가이드

이 문서는 CLaD의 Stage 1인 Cross-Modal Latent Dynamics를 LIBERO-LONG
데이터로 학습하는 절차를 설명한다. 모든 명령은 프로젝트 루트인
`/home/jack/practice/CLaD`에서 실행하는 것을 기준으로 한다.

## 1. 학습 파이프라인

Stage 1은 학습 중 DecisionNCE를 직접 실행하지 않는다. 먼저 고정된
DecisionNCE로 LIBERO 이미지와 task instruction을 인코딩하고, 저장된 특징을
학습 데이터와 결합한다.

```text
LIBERO HDF5 원본 데이터
        ↓
DecisionNCE 이미지·텍스트 특징 캐시
        ↓
CLaD Stage 1 학습 (25K optimizer steps)
        ↓
Stage 1 체크포인트
        ↓
Stage 2 Diffusion Policy 학습
```

따라서 전체 10개 task로 본 학습을 시작하기 전에 10개 task의 DecisionNCE
특징 캐시를 모두 만들어야 한다. 현재 `libero_long_smoke` 캐시는 한 task만
포함하므로 학습 파이프라인 점검에만 사용한다.

## 2. 환경 확인

Conda 환경을 활성화하고 CUDA가 PyTorch에서 인식되는지 확인한다.

```bash
cd /home/jack/practice/CLaD
conda activate clad

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()`이 `True`여야 한다. DecisionNCE submodule과 프로젝트를
아직 설치하지 않았다면 다음 명령도 실행한다.

```bash
git submodule update --init --recursive
pip install -e third_party/DecisionNCE
pip install -e ".[dev,train]"
```

## 3. 전체 DecisionNCE 특징 캐시 생성

`--max-tasks`를 지정하지 않아야 10개 task 전체를 처리한다.

```bash
python scripts/cache_decisionnce_features.py \
  --dataset-dir /data/jack/libero_datasets/libero_10 \
  --cache-dir .cache/decisionnce/libero_long \
  --model-name DecisionNCE-T \
  --device cuda \
  --batch-size 128
```

GPU 메모리가 부족하면 이 명령의 `--batch-size`만 64, 32 등으로 낮춘다. 이는
특징 추출 batch 크기이며 Stage 1 학습 batch 크기와 무관하다.

캐시 생성은 task 단위로 원자적으로 처리된다. 중간에 중단되었을 때 같은
명령을 다시 실행하면 fingerprint가 일치하는 완성된 task는 건너뛴다. 기존
캐시가 다른 데이터, 카메라, DecisionNCE revision 또는 체크포인트로 생성된
경우에는 자동으로 섞지 않고 오류를 발생시킨다.

완료 후 manifest에 10개 task가 있는지 확인한다.

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path(".cache/decisionnce/libero_long/manifest.json").read_text()
)
print("cached tasks:", len(manifest["tasks"]))
for task in manifest["tasks"]:
    print(task["task_id"])
PY
```

출력의 `cached tasks`가 `10`이어야 전체 Stage 1 학습을 시작할 수 있다.

## 4. Stage 1 설정

설정은 역할별로 세 파일에 나뉜다.

- `configs/data/libero_long.yaml`: 데이터 경로, 카메라, horizon
- `configs/model/clad_stage1.yaml`: hidden dimension, token 수, attention 구조,
  EMA와 reconstruction objective
- `configs/train/stage1.yaml`: optimizer, batch, scheduler, AMP, logging,
  checkpoint

본 학습의 기본 설정은 다음과 같다.

```yaml
max_steps: 25000
batch_size: 128
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
weight_decay: 0.01
beta1: 0.9
beta2: 0.95
warmup_steps: 500
min_lr_ratio: 0.01
max_grad_norm: 1.0
amp_enabled: true
amp_dtype: float16
amp_init_scale: 2048.0
max_consecutive_optimizer_skips: 16
checkpoint_interval: 1000
```

논문이 명시한 값은 25K step, batch 128, EMA momentum 0.995이다. AdamW,
learning rate, warmup/cosine scheduler, gradient clipping과 fp16 AMP는 논문에
보고되지 않아 이 재현에서 사용한 명시적 가정이다. 기본 `amp_init_scale`은
전체 8-layer GPU smoke test에서 PyTorch 기본값 65536이 다섯 번 overflow한 뒤
안정화된 2048을 사용한다. 이후에도 GradScaler가 scale을 동적으로 조정한다.

설정을 별도로 보존하려면 기존 파일을 복사하고 `--train-config`로 선택한다.

```bash
cp configs/train/stage1.yaml configs/train/stage1_local.yaml
```

GPU 메모리가 부족할 경우 micro-batch를 낮추고 gradient accumulation을 늘려
유효 batch 크기를 유지할 수 있다.

```yaml
batch_size: 16
gradient_accumulation_steps: 8
```

```text
effective batch size = batch_size × gradient_accumulation_steps
                     = 16 × 8
                     = 128
```

한 optimizer step은 accumulation에 사용된 모든 micro-batch가 처리된 뒤
증가한다. EMA와 scheduler도 optimizer step이 실제로 수행된 후 한 번씩만
갱신된다.

## 5. GPU smoke test

본 학습 전에 기존 1-task 캐시로 기본 8-layer 모델이 GPU에서 한 optimizer
step을 실행할 수 있는지 확인한다.

```bash
python scripts/train_clad_stage1.py \
  --dataset-dir /data/jack/libero_datasets/libero_10 \
  --cache-dir .cache/decisionnce/libero_long_smoke \
  --file-pattern KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5 \
  --output-dir outputs/clad_stage1_smoke \
  --device cuda \
  --max-steps 1 \
  --batch-size 1 \
  --warmup-steps 0 \
  --num-workers 0 \
  --log-interval 1 \
  --checkpoint-interval 0 \
  --no-save-final-checkpoint
```

다음 조건을 확인한다.

- `completed_step`이 `1`이다.
- loss 값이 `nan`이나 `inf`가 아니다.
- `optimizer_step_skipped`가 `0.0`이다.
- CUDA out-of-memory 오류가 발생하지 않는다.

`max_steps`는 시도 횟수가 아니라 성공한 optimizer update 수다. AMP overflow가
발생하면 해당 시도는 `attempt_step`과 `skipped_optimizer_steps`에만 기록되며,
Trainer는 성공한 update가 `max_steps`에 도달할 때까지 계속 실행한다. 연속
skip이 `max_consecutive_optimizer_skips`에 도달하면 무한 반복하는 대신 원인과
현재 AMP scale을 포함한 오류를 발생시킨다.

데이터부터 optimizer까지의 연결만 빠르게 확인하고 싶다면
`--attention-layers 1`을 추가할 수 있다. 이 경우 모델이 축소되므로 기본
8-layer 모델의 메모리 검증을 대신하지는 않는다.

## 6. Stage 1 본 학습

10-task 캐시와 기본 설정으로 학습한다. `clad` Conda 환경을 활성화한 뒤 다음
shell script를 실행하는 것이 가장 간단하다.

```bash
./scripts/train_stage1.sh
```

이 script는 아래 Python 명령을 실행하면서 콘솔 출력을 자동으로 저장한다.

```bash
python scripts/train_clad_stage1.py \
  --data-config configs/data/libero_long.yaml \
  --model-config configs/model/clad_stage1.yaml \
  --train-config configs/train/stage1.yaml \
  --dataset-dir /data/jack/libero_datasets/libero_10 \
  --cache-dir .cache/decisionnce/libero_long \
  --output-dir outputs/clad_stage1 \
  --device cuda
```

추가한 인자는 Python 학습 명령으로 그대로 전달된다. 예를 들어 다른 학습
설정 파일을 사용하려면 다음처럼 실행한다.

```bash
./scripts/train_stage1.sh --train-config configs/train/stage1_local.yaml
```

별도 설정을 만들었다면 `--train-config configs/train/stage1_local.yaml`로
바꾼다. 명령행에서 지정한 `--max-steps`, `--batch-size`,
`--gradient-accumulation-steps` 등의 값은 YAML 설정보다 우선한다. 본 실험은
재현성을 위해 YAML 파일에 값을 고정하고 일시적인 smoke test에만 명령행
override를 사용하는 것을 권장한다.

콘솔에는 `log_interval`마다 다음과 같은 한 줄만 출력된다.

```text
[Stage1]    10/25000 (  0.0%) | ETA 02:17:24 | 0.330s/step | loss 12.2339 | grad 91.9 | lr 5.050e-05 | amp 2048 | skips 0 | ok
```

ETA는 성공한 optimizer step의 소요 시간을 지수 이동 평균으로 평활화해
계산한다. AMP skip과 checkpoint 저장에 걸린 시간도 이후 관측 구간에
포함된다. 학습 초반 몇 번의 출력은 표본이 적어 ETA 변동이 클 수 있으며,
진행할수록 안정된다. 실제 시도 횟수가 성공 step과 다를 때만 `try` 항목을
추가로 표시한다.

전체 수치는 다음 파일에 보존된다.

- `outputs/clad_stage1/train_console.log`: 오류를 포함한 전체 콘솔 출력
- `outputs/clad_stage1/train_metrics.jsonl`: 손실과 모든 진단 metric
- `outputs/clad_stage1/run_config_<run-id>.json`: 실행별 resolved 설정

`train_console.log`와 `train_metrics.jsonl`은 이어 쓰며, 각 JSONL record에는
실행을 구분하는 `run_id`와 UTC 시간이 포함된다. JSONL에는 다음 항목이
저장된다. `estimated_seconds_per_step`과 `estimated_eta_seconds`도 함께 저장되어
학습 후 처리 속도를 분석할 수 있다.

- `loss`, `loss_latent`, `loss_reconstruction`
- modality별 latent/reconstruction loss
- `action_mask_ratio`
- clipping 전 `gradient_norm`
- 성공한 update 수 `step`과 전체 시도 수 `attempt_step`
- `skipped_optimizer_steps`, `consecutive_optimizer_skips`, `amp_scale`
- 다음 step에 적용될 `learning_rate`
- AMP overflow에 따른 `optimizer_step_skipped`

## 7. 체크포인트와 학습 재개

기본 설정은 1,000 step마다 다음 파일을 원자적으로 교체한다.

```text
outputs/clad_stage1/stage1_latest.pt
```

체크포인트에는 다음 상태가 포함된다.

- online CLaD와 EMA target encoder 가중치
- AdamW optimizer와 learning-rate scheduler
- AMP gradient scaler
- global step과 최근 metrics
- 전체 optimizer 시도 횟수와 AMP skip 통계
- Python, NumPy, CPU/CUDA PyTorch RNG
- shuffled DataLoader의 정확한 batch 위치

다음 명령으로 재개한다.

```bash
./scripts/train_stage1.sh \
  --resume outputs/clad_stage1/stage1_latest.pt
```

동일한 동작을 Python 명령으로 직접 실행할 수도 있다.

```bash
python scripts/train_clad_stage1.py \
  --data-config configs/data/libero_long.yaml \
  --model-config configs/model/clad_stage1.yaml \
  --train-config configs/train/stage1.yaml \
  --dataset-dir /data/jack/libero_datasets/libero_10 \
  --cache-dir .cache/decisionnce/libero_long \
  --output-dir outputs/clad_stage1 \
  --device cuda \
  --resume outputs/clad_stage1/stage1_latest.pt
```

재개할 때 dataset, file pattern, batch size와 seed를 변경하면 안 된다. 이 값이
달라지면 저장된 shuffled batch 위치를 정확히 복원할 수 없으므로 Trainer가
오류를 발생시킨다. 모델 구조도 체크포인트를 만들 때와 같아야 한다.

## 8. 자주 발생하는 문제

### Feature cache does not cover all dataset tasks

한 task짜리 `libero_long_smoke` 캐시를 전체 데이터 학습에 사용했거나 전체
캐시 생성이 끝나지 않은 상태다. `libero_long` manifest가 10개 task를
포함하는지 확인한다.

### CUDA out of memory

먼저 `configs/train/stage1.yaml`의 `batch_size`를 낮추고
`gradient_accumulation_steps`를 같은 비율로 높인다. batch 1에서도 기본
8-layer 모델이 실행되지 않으면 `--attention-layers 1`로 파이프라인만 점검할
수 있지만, 이는 논문 규모 모델의 본 학습이 아니다.

### Dataloader has no complete batches

Trainer는 일관된 batch 크기를 위해 마지막 불완전 batch를 버린다. 선택한
task subset의 window 수보다 `batch_size`가 크면 이 오류가 발생한다.

### optimizer_step_skipped가 1.0

fp16 gradient에 `inf` 또는 `nan`이 감지되어 AMP scaler가 optimizer update를
건너뛴 경우다. 이 시도는 25K optimizer-step budget에 포함되지 않는다.
간헐적인 skip은 scaler가 자동 조정하지만, 설정된 연속 skip 한도까지 반복되면
학습을 중단한다. 이 경우 loss와 입력 특징을 확인하고 필요하면
`amp_init_scale`을 낮춘다.

## 9. Stage 1 완료 조건

다음 조건이 충족되면 Stage 2 구현과 학습에 사용할 수 있다.

- 전체 10-task cache fingerprint 검증 통과
- 25,000 optimizer steps 완료
- 최종 loss가 유한하고 장기간 AMP skip이 반복되지 않음
- `outputs/clad_stage1/stage1_latest.pt` 저장 확인
- 동일 설정으로 checkpoint load 및 inference forward 통과

다음 단계에서는 이 체크포인트의 CLaD를 고정하고, 예측된 latent foresight를
조건으로 사용하는 Diffusion Policy를 학습한다.
