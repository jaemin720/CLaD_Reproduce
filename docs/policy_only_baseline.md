# Policy-only baseline

## 목적과 전체 계획상의 위치

이 baseline은 CLaD 논문의 modality ablation 중 `Policy-only` 결과를 진단용으로
재현한다. 전체 구현 이후 성능 차이를 분석하는 첫 번째 ablation 단계이며, 다음
질문을 분리해서 확인한다.

> 현재 LIBERO data contract, DecisionNCE feature, diffusion denoiser와 rollout만으로
> 충분한 행동 정책을 학습할 수 있는가?

논문은 이 baseline의 LIBERO-LONG 평균 success rate를 84.8%로 보고한다. 현재
구현의 결과가 그 수준에 미치지 못하면 CLaD foresight 이전의 공통 경로부터
점검해야 한다.

논문은 Policy-only의 내부 observation encoder를 공개하지 않는다. 따라서 이
구현은 저자 architecture의 복제가 아니라, 기존 CLaD Stage 2와 비교 변수를
최소화한 명시적 baseline이다.

## 모델 입력과 CLaD 대비 차이

Policy-only가 사용하는 입력은 현재 시점의 다음 세 가지뿐이다.

```text
proprio_now                 [B, 9]
DecisionNCE image_now       [B, 1024]
DecisionNCE instruction     [B, 1024]
```

visual feature와 instruction은 trainable FiLM으로 융합한다. proprioception과
semantic state는 각각 two-layer MLP로 1024차원 condition이 되며, 두 condition을
기존과 동일한 conditional 1D U-Net에 제공한다.

신규 학습의 9D proprioception은 공식 LIBERO policy observation 순서인
`joint_states(7) + gripper_states(2)`다. online 평가도 같은 순서의
`robot0_joint_pos + robot0_gripper_qpos`를 사용한다.

다음 정보는 사용하지 않는다.

- Stage 1 checkpoint와 latent foresight;
- `t-6` observation과 과거 action history;
- cross-modal transition과 asymmetric attention;
- future observation target.

U-Net, 100-step DDPM, action normalization, optimizer, policy EMA, 200K steps,
batch size 128은 CLaD Stage 2와 동일하다. full configuration의 trainable parameter
수는 약 232.7M으로, CLaD Stage 2의 약 231.6M과 유사하다. 따라서 큰 parameter
budget 차이보다 foresight 유무를 비교하는 데 목적이 있다.

## GPU smoke test

기존 ten-task DecisionNCE cache만 필요하며 Stage 1 artifact는 필요 없다.

```bash
conda activate clad
cd /path/to/CLaD_Reproduce
export LIBERO_DATASET_DIR=/path/to/libero_datasets/libero_10

./scripts/train_policy_only.sh \
  --max-steps 1 \
  --batch-size 1 \
  --warmup-steps 0 \
  --num-workers 0 \
  --log-interval 1 \
  --checkpoint-interval 0 \
  --no-save-final-checkpoint
```

성공 조건은 finite loss와 gradient, `step=1`, optimizer skip 0이다. 이 명령은
full-width 0.23B U-Net을 그대로 사용하므로 실제 본 학습의 VRAM contract도
확인한다.

## 200K 본 학습

```bash
export LIBERO_DATASET_DIR=/path/to/libero_datasets/libero_10
./scripts/train_policy_only.sh
```

출력은 CLaD checkpoint와 섞이지 않도록 별도 directory에 저장된다.

```text
outputs/policy_only_official/
├── stage2_latest.pt
├── train_metrics.jsonl
├── train_console.log
└── run_config_*.json
```

재개 명령은 다음과 같다.

```bash
./scripts/train_policy_only.sh \
  --resume outputs/policy_only_official/stage2_latest.pt
```

checkpoint의 `policy_variant`는 `policy_only`이고 `foresight_checkpoint`는 `null`이다.
loader는 CLaD와 Policy-only를 이 값으로 자동 구분하며 variant가 다른 checkpoint로
resume하는 것을 거부한다.

`outputs/policy_only/`에 이미 생성된 이전 200K checkpoint는 EEF 기반
`robot_states` 실험으로 보존한다. 두 입력은 모두 9D지만 의미와 순서가 다르므로
새 launcher의 기본 출력은 `outputs/policy_only_official/`로 분리했다. 이전
checkpoint를 새 공식 dataset으로 resume할 수 없으며 새 결과에는 재학습이
필요하다. DecisionNCE cache는 image/text-only이므로 single-view cache는 그대로
재사용할 수 있다.

## 구현 감사 결과

현재 구현은 논문이 명시한 `diffusion policy without foresight conditioning`을 다음
관점에서 충족한다.

- conditioner는 `proprio_now`, `image_now`, instruction만 읽고 Stage 1, 과거
  observation/action, future target을 사용하지 않는다.
- 6-step action을 data 범위에서 `[-1, 1]`로 정규화한 뒤 cosine DDPM forward
  process의 noise-prediction MSE를 학습한다.
- conditional 1D U-Net, 200K update, batch 128, trainable-parameter EMA는 CLaD
  Stage 2와 동일하다.
- checkpoint는 `policy_variant=policy_only`, conditioner 설정, EMA와 action
  normalization 통계를 함께 저장한다. 평가는 기본적으로 EMA를 복원한다.
- 새 single-view checkpoint는 `camera_views=[agentview_rgb]` 계약을 기록한다.
  이 필드 추가 전에 생성된 Policy-only checkpoint는 역호환을 위해 같은 단일 뷰
  기본값으로 해석한다. 다른 view cache로 학습 재개 또는 rollout하려 하면 즉시
  실패한다.
- 새 checkpoint는 `proprioception=libero_joint_gripper`를 기록한다. 이 필드가
  없는 checkpoint는 `robot_states`로만 해석하며 평가 시 해당 live observation
  layout을 자동 선택한다.

다만 논문은 Policy-only observation encoder, U-Net channel, diffusion timestep,
action normalization, 카메라 수를 공개하지 않는다. 따라서 현재 구현은 84.8% 숫자를
보장하는 저자 코드의 완전 복제가 아니라, 논문의 명시사항을 만족하면서 CLaD와
변수를 통제한 재현 baseline이다. 특히 DecisionNCE-T 선택, two-layer observation
MLP, 100-step DDPM은 문서화된 재현 판단이다.

## 2-view Policy-only 실험

이 실험은 외부 카메라와 wrist 카메라를 모두 사용하되 다른 설정은 single-view
Policy-only와 동일하게 고정한다.

```text
agentview RGB ─► frozen DecisionNCE ─┐
                                     ├─ mean ─► language FiLM ─► semantic MLP
eye-in-hand RGB ► frozen DecisionNCE ┘
proprio_now ────────────────────────────────────────────────► proprio MLP
```

두 feature는 같은 가중치로 평균된다. 따라서 두 view가 모두 gradient 이전의 semantic
입력에 참여하지만 view identity를 위한 추가 trainable parameter는 없다. 이것은
single-view 대비 카메라 정보만 바꾸는 첫 controlled experiment다. learned fusion은
별도 후속 ablation으로 다루는 편이 결과 해석에 안전하다.

단일 뷰 cache를 덮어쓰지 않고 별도 cache를 먼저 만든다.

```bash
conda activate clad
cd /path/to/CLaD_Reproduce
export LIBERO_DATASET_DIR=/path/to/libero_datasets/libero_10

./scripts/cache_decisionnce_two_view.sh
```

다음 네 계약은 서로 정확히 일치해야 하며 코드가 이를 검증한다.

| 위치 | camera 순서 |
|---|---|
| raw HDF5 | `obs/agentview_rgb`, `obs/eye_in_hand_rgb` |
| two-view cache manifest | 위 두 HDF5 key |
| model checkpoint | `agentview_rgb`, `eye_in_hand_rgb` |
| live LIBERO observation | `agentview_image`, `robot0_eye_in_hand_image` |

full-width GPU smoke test는 다음과 같다.

```bash
./scripts/train_policy_only_two_view.sh \
  --max-steps 1 \
  --batch-size 1 \
  --warmup-steps 0 \
  --num-workers 0 \
  --log-interval 1 \
  --checkpoint-interval 0 \
  --no-save-final-checkpoint
```

본 학습은 override 없이 실행한다.

```bash
./scripts/train_policy_only_two_view.sh
```

출력은 `outputs/policy_only_two_view`, cache는
`.cache/decisionnce/libero_long_two_view`에 저장되어 기존 실험과 섞이지 않는다.
평가는 GPU 번호를 첫 인자로 지정한다.

```bash
./scripts/evaluate_policy_only_two_view.sh 1
```

짧은 평가 옵션도 그대로 전달할 수 있다.

```bash
./scripts/evaluate_policy_only_two_view.sh 1 \
  --task-ids 0 --rollouts-per-task 1 --max-steps 30
```

## 평가

기존 evaluator가 checkpoint variant를 자동 감지한다. GPU 1에서 짧게 확인하려면:

```bash
CLAD_STAGE2_CHECKPOINT=outputs/policy_only_official/stage2_latest.pt \
CLAD_EVAL_OUTPUT_DIR=outputs/policy_only_official_evaluation_smoke \
./scripts/evaluate_libero.sh 1 \
  --task-ids 0 \
  --rollouts-per-task 1 \
  --max-steps 30
```

전체 50-rollout 평가는 override를 제거한다.

```bash
CLAD_STAGE2_CHECKPOINT=outputs/policy_only_official/stage2_latest.pt \
CLAD_EVAL_OUTPUT_DIR=outputs/policy_only_official_evaluation \
./scripts/evaluate_libero.sh 1
```

이미 본 학습이 실행 중이라면 별도의 shell에서 다음 명령을 미리 실행할 수 있다.
학습 로그의 정확한 200K 완료 표식을 기다리고, final checkpoint step을 다시
검증한 뒤 즉시 GPU 1에서 전체 평가를 시작한다.

```bash
./scripts/evaluate_policy_only_when_ready.sh 1
```

짧은 평가를 예약하려면 평가 override를 그대로 뒤에 붙인다.

```bash
./scripts/evaluate_policy_only_when_ready.sh 1 \
  --task-ids 0 \
  --rollouts-per-task 10
```

학습 process가 완료 표식 없이 종료되거나 checkpoint가 200K가 아니면 평가를
시작하지 않는다. polling 주기는 `CLAD_WAIT_POLL_SECONDS`, 다른 의도적인 학습
길이는 `CLAD_POLICY_ONLY_EXPECTED_STEP`으로 바꿀 수 있다.

평가 결과에는 `policy_variant=policy_only`가 기록된다. Stage 1 checkpoint 경로가
launcher 환경에 존재하더라도 Policy-only loader는 이를 읽거나 검증하지 않는다.

## 해석 시 주의점

- 현재 raw LIBERO 128x128 image, 공식 LIBERO 순서의 9D joint+gripper
  proprioception, action alignment와 rollout protocol을 사용한다. 따라서 낮은
  성능은 이 공통 contract에 문제가 있다는 강한 신호지만 어느 한 요소를 단독
  원인으로 확정하지는 않는다.
- Policy-only가 높고 CLaD가 낮다면 Stage 1 representation 또는 Stage 2 FiLM
  bridge를 우선 점검한다.
- 둘 다 낮다면 data preprocessing, action alignment, diffusion inference와
  rollout history를 먼저 점검한다.
- 공식 proprioception으로 single-view 결과를 먼저 고정한 뒤 위 2-view 실험에서는
  카메라만 바꾼다. 두 신규 config 모두 같은 `libero_joint_gripper` 계약을 쓰므로
  action alignment와 rollout 설정도 함께 고정된다.
