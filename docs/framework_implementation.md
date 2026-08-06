# CLaD 전체 프레임워크 구현 구조

이 문서는 LIBERO demonstration에서 시작해 CLaD Stage 1과 Diffusion Policy
Stage 2를 학습하고, EMA policy로 online LIBERO rollout을 수행하기까지 실제
코드가 어떻게 연결되는지 설명한다. 논문에 없는 선택과 그 위험도는
[`reproduction_assumptions.md`](reproduction_assumptions.md)를 함께 참고한다.

논문상 학습 stage는 두 개다. 이 저장소에서는 준비와 평가까지 포함해 다음
다섯 실행 phase로 나눈다.

```text
LIBERO HDF5 demonstrations
          │
          ├── raw proprioception/actions ───────────────────────┐
          │                                                     │
          └── RGB + instruction                                 │
                    │                                           │
                    ▼                                           │
          frozen DecisionNCE feature cache                      │
                    │                                           │
                    ├───────────────────────────────────────────┤
                    ▼                                           │
     Stage 1: Cross-Modal Latent Dynamics                       │
       state/action tokens → transition attention               │
       → asymmetric dynamics → latent foresight                 │
       → EMA targets + reconstruction loss                      │
                    │                                           │
                    ▼                                           │
          compact frozen-foresight checkpoint                   │
                    │                                           │
                    ├───────────────────────────────────────────┘
                    ▼
     Stage 2: foresight-conditioned Diffusion Policy
       frozen CLaD → observation FiLM → conditional 1D U-Net
       → DDPM noise loss → trainable-policy EMA
                    │
                    ▼
            Stage 2 policy checkpoint
                    │
                    ▼
     LIBERO online rollout
       live DecisionNCE + history buffer + reverse DDPM
       → action chunk → environment → success metrics/video
```

## 1. 주요 디렉터리와 책임

| 경로 | 책임 |
| --- | --- |
| [`configs/data`](../configs/data) | dataset schema, camera, feature-cache 기본값 |
| [`configs/model`](../configs/model) | Stage 1/2 architecture와 DecisionNCE 선택 |
| [`configs/train`](../configs/train) | optimizer, scheduler, AMP, EMA, checkpoint |
| [`configs/eval`](../configs/eval) | LIBERO rollout protocol |
| [`src/clad/data`](../src/clad/data) | HDF5 discovery, temporal windows, cache, action statistics |
| [`src/clad/models`](../src/clad/models) | DecisionNCE adapter, CLaD, conditioning, diffusion policy |
| [`src/clad/training`](../src/clad/training) | resumable trainers와 metric logging |
| [`src/clad/evaluation`](../src/clad/evaluation) | checkpoint 복원, online history, LIBERO rollout |
| [`scripts`](../scripts) | 각 phase의 command-line entry point |
| [`third_party/DecisionNCE`](../third_party/DecisionNCE) | pinned official DecisionNCE source |
| [`third_party/LIBERO`](../third_party/LIBERO) | pinned official LIBERO source |

원본 dataset, 다운로드한 model checkpoint, feature cache, training output은
repository에 commit하지 않는다. source revision과 license만 submodule 및 notice로
보존한다.

## 2. 공통 표기와 tensor contract

현재 기본 설정의 dimension은 다음과 같다.

| 기호 | 값 | 의미 |
| --- | ---: | --- |
| `B` | runtime batch | batch size |
| `V` | 1 | camera view 수, 확장 가능 |
| `Dv` | 1024 | DecisionNCE visual feature |
| `Dl` | 1024 | DecisionNCE text feature |
| `Dp` | 9 | proprioception |
| `Da` | 7 | environment action |
| `H` | 1024 | CLaD hidden dimension |
| `Np`, `Ns` | 4, 4 | proprio/semantic state tokens |
| `τ` | 6 | history, foresight, action horizon |

한 collated training sample의 핵심 field는 다음과 같다.

| field | shape | Stage 1 | Stage 2 |
| --- | --- | --- | --- |
| `vision_features[view].prev` | `[B,Dv]` | 사용 | 사용 |
| `vision_features[view].now` | `[B,Dv]` | 사용 | 사용 |
| `vision_features[view].future` | `[B,Dv]` | target | cache load 생략 |
| `text_feature` | `[B,Dl]` | 사용 | 사용 |
| `proprio_prev` | `[B,Dp]` | 사용 | 사용 |
| `proprio_now` | `[B,Dp]` | 사용 | 사용 |
| `proprio_future` | `[B,Dp]` | target | 사용 안 함 |
| `past_actions` | `[B,τ,Da]` | 사용 | 사용 |
| `target_actions` | `[B,τ,Da]` | 사용 안 함 | DDPM target |

`prev`, `now`, `future`는 각각 `t-τ`, `t`, `t+τ`다. `past_actions`는
`[t-τ,t)`, `target_actions`는 `[t,t+τ)`다.

## 3. Phase 0: LIBERO data indexing

### 3.1 Task discovery

[`task_registry.py`](../src/clad/data/task_registry.py)는 dataset directory의
`*_demo.hdf5`를 정렬해 task를 발견한다. 각 파일에서 다음을 검증한다.

- `data` group 존재;
- `demo_N` group 존재 및 숫자 순서 정렬;
- `problem_info.language_instruction` 또는 fallback instruction;
- 선언된 `num_demos`와 실제 group 수 일치.

파일 stem에서 `_demo`를 제거한 문자열이 training/cache task ID가 된다.

### 3.2 Episode-safe window index

[`LiberoWindowDataset`](../src/clad/data/libero_dataset.py)은 시작할 때 모든
episode length를 읽고 valid anchor를 메모리에 색인한다. raw HDF5 handle은
DataLoader worker마다 lazy open하므로 parent process에서 dataset을 만든 뒤에도
multi-worker loading이 가능하다.

한 episode 길이가 `T`라면 valid anchor는 다음 범위다.

```text
t = τ, τ+1, ..., T-τ-1
```

`t+τ` observation이 실제 episode 내부에 있어야 하므로 마지막 index는 제외된다.
경계 padding이나 episode 간 연결은 없다.

### 3.3 Raw sample

dataset은 공식 LIBERO policy observation 순서대로 `obs/joint_states` 7D와
`obs/gripper_states` 2D를 연결하고 `actions`와 함께 float32 tensor로 읽는다.
raw image mode는 `uint8 [H,W,3]`를 반환하지만 실제 학습 entry point는
`include_images=False`로 설정하고 다음 phase의 cached feature를 결합한다.
이전 `robot_states` 9D 경로는 기존 checkpoint 재현용 named legacy contract로만
남아 있다.

## 4. Phase 1: frozen DecisionNCE feature cache

entry point는
[`cache_decisionnce_features.py`](../scripts/cache_decisionnce_features.py)다.

### 4.1 Adapter

[`DecisionNCEAdapter`](../src/clad/models/decisionnce_adapter.py)는 official
`DecisionNCE.load()`를 lazy import하고 다음만 담당한다.

1. `auto/cpu/cuda` device 결정;
2. `[B,H,W,3]`를 `[B,3,H,W]`로 변환;
3. upstream image/text encoder 호출;
4. output이 finite `[B,D]`인지 검증;
5. parameter freeze, eval mode, inference mode 강제.

CLIP resize/crop/normalize와 text tokenization은 upstream source가 소유한다.

### 4.2 Cache build

[`DecisionNCEFeatureCacheBuilder`](../src/clad/data/feature_cache.py)는 task마다
다음을 저장한다.

```text
<task-id>.hdf5
├── text_feature                         [1024]
└── data/<demo-id>/images/<view-name>    [T,1024]
```

image는 configurable batch로 encode하고 기본 float16으로 CPU HDF5에 쓴다.
task instruction은 task당 한 번만 encode한다. 임시 파일을 완성한 뒤 rename해
중간에 중단된 cache를 완성본으로 오인하지 않게 한다.

`manifest.json`에는 다음 실험 identity가 들어간다.

- DecisionNCE model name, source revision, checkpoint SHA-256;
- camera keys와 feature dtype;
- source HDF5 path, size, modification time, instruction, demo 수;
- task별 fingerprint와 cache filename.

학습과 평가 시 manifest가 requested task/camera를 완전히 포함하는지 검사한다.

### 4.3 Cached dataset join

[`CachedLiberoWindowDataset`](../src/clad/data/cached_dataset.py)은 raw-state
sample의 task/episode/anchor를 이용해 동일 시점의 cached feature를 읽는다.
Stage 1은 `prev/now/future`, Stage 2는 `prev/now`만 읽는다. VLM은 두 학습
stage에서 optimizer graph에 들어가지 않는다.

## 5. Phase 2: Stage 1 CLaD pre-training

entry point는 [`train_clad_stage1.py`](../scripts/train_clad_stage1.py), 전체 실행
launcher는 [`train_stage1.sh`](../scripts/train_stage1.sh)다.

### 5.1 입력 encoding

구현은 [`clad_inputs.py`](../src/clad/models/clad_inputs.py)에 있다.

```text
visual feature(s) [B,V,1024]
        │ mean across V
        ▼
language FiLM with text [B,1024]
        │
        ▼
2-layer MLP tokenizer ───────────────► semantic tokens [B,4,1024]

proprio [B,9]
        └── 2-layer MLP tokenizer ───► proprio tokens  [B,4,1024]

past actions [B,6,7]
        └── token-wise MLP + mask + position ─► action tokens [B,6,1024]
```

semantic/proprio encoder는 `prev`와 `now`에 같은 online parameter를 공유한다.
action mask는 model이 training mode일 때만 sampling한다.

### 5.2 Modality transition

[`clad_transition.py`](../src/clad/models/clad_transition.py)는 두 독립 branch를
구성한다.

```text
z_p = CrossAttention(
    query=proprio_now [B,4,H],
    context=[proprio_prev; action] [B,10,H],
)

z_s = CrossAttention(
    query=semantic_now [B,4,H],
    context=[semantic_prev; action] [B,10,H],
)
```

각 stack은 8개 pre-norm cross-attention/FFN layer다. 요청할 경우 layer/head별
attention map도 반환하지만 기본 학습에서는 저장하지 않는다.

### 5.3 Asymmetric cross-modal dynamics

[`clad_dynamics.py`](../src/clad/models/clad_dynamics.py)는 proprio transition을
query, semantic transition을 context로 사용한다.

```text
z_(p→s) = CrossAttention(z_p, z_s)     [B,4,H]
z_dyn   = CrossAttention(q_out, z_(p→s))[:,0]  [B,H]
```

마지막 단계의 `q_out`은 batch마다 복제되는 하나의 learned parameter다.

### 5.4 Grounded latent foresight와 loss

[`clad_foresight.py`](../src/clad/models/clad_foresight.py)는 `z_dyn`에서 두
future latent를 예측한다.

```text
z_hat_p = MLP_p(z_dyn)   [B,H]
z_hat_s = MLP_s(z_dyn)   [B,H]
```

online semantic/proprio encoder의 EMA copy가 future observation을 token화하고
token 평균으로 target을 만든다. target graph는 stop-gradient다.

```text
z_bar_p = mean(EMA_f_p(proprio_future), tokens)              [B,H]
z_bar_s = mean(EMA_f_s(FiLM_target(vision_future,text)), tokens) [B,H]
```

두 reconstruction head는 predicted latent에서 다음을 복원한다.

```text
p_recon = h_p(z_hat_p)                 [B,9]
v_recon = h_s(z_hat_s)                 [B,1024]
```

현재 objective는 다음과 같다.

```text
L_latent = ||z_hat_p - normalize(stopgrad(z_bar_p))||²
         + ||z_hat_s - normalize(stopgrad(z_bar_s))||²

L_recon  = ||p_recon - proprio_future||₁
         + ||v_recon - fused_visual_future||₁

L_stage1 = L_latent + 0.1 * L_recon
```

norm은 feature dimension 합, 최종 reduction은 batch 평균이다.

### 5.5 Composed model

[`CLaDStage1Model`](../src/clad/models/clad_stage1.py)이 위 module을 한 forward로
연결한다. 반환값에는 total loss뿐 아니라 modality loss, transition token,
attention metadata, `z_dyn`, foresight, EMA target, reconstruction, 실제 action
mask가 모두 들어 있어 ablation과 debugging에 사용할 수 있다.

기본 full model 관찰값은 total 약 346.5M, trainable 약 334.9M parameters다.
EMA target 약 11.6M도 module state에 포함되지만 optimizer 대상은 아니다.

### 5.6 Trainer update 순서

[`Stage1Trainer`](../src/clad/training/stage1_trainer.py)의 한 successful update는
다음 순서다.

1. shuffled window batch를 device로 이동;
2. fp16 autocast forward와 loss 계산;
3. gradient accumulation 및 scaled backward;
4. unscale 후 global norm clip;
5. finite하면 AdamW update;
6. online state encoder에서 EMA target update;
7. learning-rate scheduler update;
8. metric logging 및 필요 시 atomic checkpoint.

AMP overflow이면 5--7을 실행하지 않고 attempt/skip counter만 증가한다.
`global_step=25,000`은 위 successful sequence를 25,000회 끝냈다는 뜻이다.

### 5.7 Stage 1 artifacts

```text
outputs/clad_stage1_official/
├── stage1_latest.pt          # model + targets + optimizer + resume state
├── train_metrics.jsonl
├── train_console.log
└── run_config_*.json
```

training checkpoint는 exact resume용이라 크다. Stage 2에는 다음 phase에서
inference subset만 별도로 내보낸다.

## 6. Stage 1 → Stage 2 artifact bridge

[`export_stage1_foresight.py`](../scripts/export_stage1_foresight.py)는 full Stage 1
checkpoint에서 다음 prefix만 고른다.

```text
inputs.*
transitions.*
dynamics.*
foresight_predictor.*
```

EMA target encoder, reconstruction head, loss, optimizer, scheduler, scaler는 제외한다.
출력 `stage1_foresight.pt`에는 model config, source path/step 및 선택된 tensor만
들어간다. write는 임시 파일 후 atomic rename이며 기존 파일은 명시적
`--overwrite` 없이 교체하지 않는다.

[`CLaDForesightBackbone`](../src/clad/models/clad_stage2.py)은 이 artifact를
strict load하고 모든 parameter를 freeze하며 `train()` 호출에도 eval mode를
유지한다. Stage 2에서는 action masking도 항상 끈다.

## 7. Phase 3: Stage 2 Diffusion Policy training

entry point는 [`train_clad_stage2.py`](../scripts/train_clad_stage2.py), 전체 실행
launcher는 [`train_stage2.sh`](../scripts/train_stage2.sh)다.

### 7.1 Dataset와 action statistics

Stage 2도 동일한 temporal window를 사용하지만 future visual/proprio target은
읽지 않는다. 대신 `[t,t+6)` `target_actions`가 필요하다.

[`action_stats.py`](../src/clad/data/action_stats.py)는 중복 window를 순회하지
않고 source episode의 각 action을 정확히 한 번 scan해 7개 dimension의 global
min/max를 구한다. policy의 `LinearActionNormalizer`가 이를 `[-1,1]` affine
mapping으로 저장한다.

### 7.2 Frozen foresight conditioning

Stage 1 backbone은 다음 history-only forward를 수행한다.

```text
(vision_prev, vision_now, text,
 proprio_prev, proprio_now, past_actions)
        │
        ▼
frozen Stage 1 encoders/transitions/dynamics/predictors
        │
        ├── z_hat_p [B,1024]
        ├── z_hat_s [B,1024]
        ├── proprio_now tokens [B,4,1024]
        └── semantic_now tokens [B,4,1024]
```

[`CLaDStage2Conditioner`](../src/clad/models/clad_stage2.py)은 current token을
평균하고 두 trainable FiLM을 적용한다.

```text
o_p = mean(proprio_now tokens)   [B,1024]
o_s = mean(semantic_now tokens)  [B,1024]

g_p = FiLM_p(z_hat_p, o_p)       [B,1024]
g_s = FiLM_s(z_hat_s, o_s)       [B,1024]
```

backbone은 frozen이고 `FiLM_p`, `FiLM_s`만 Stage 2 optimizer에 포함된다.

### 7.3 DDPM training objective

구현은 [`clad_diffusion.py`](../src/clad/models/clad_diffusion.py)에 있다.

1. target action `[B,6,7]`을 dimension별 `[-1,1]`로 normalize;
2. sample마다 timestep `k∈[0,99]`를 uniform sampling;
3. Gaussian noise `epsilon`을 뽑고 forward DDPM으로 `a_k` 생성;
4. timestep embedding과 `[g_p;g_s]`로 conditional 1D U-Net 실행;
5. predicted noise와 sampled noise의 element-mean MSE 계산.

```text
timestep embedding                    [B,256]
g_p, g_s                              [B,1024] each
residual-block global condition       [B,2304]
noisy/predicted action                [B,6,7]
```

U-Net은 temporal length를 6에서 8로 내부 padding한 뒤 8→4→2로 downsample하고,
skip connection과 upsample 후 다시 앞 6 step만 반환한다.

### 7.4 Trainable 범위와 policy EMA

기본 Stage 2 trainable parameter는 다음 두 집합뿐이다.

- observation FiLM 두 개: 약 4.2M;
- conditional denoiser: 227,412,743.

합계는 231,611,143으로 약 0.23B다. frozen CLaD를 포함해 policy object 전체는
약 563.4M이지만 frozen parameter는 optimizer와 policy EMA에서 제외한다.

[`Stage2Trainer`](../src/clad/training/stage2_trainer.py)는 Stage 1과 같은
successful-step, AMP, exact-resume contract를 쓰며 successful AdamW update 뒤
trainable parameter EMA를 갱신한다.

### 7.5 Stage 2 checkpoint

```text
outputs/clad_stage2_official/stage2_latest.pt
```

이 파일은 다음을 포함한다.

- trainable FiLM/U-Net raw weights;
- trainable weights의 EMA shadow;
- action normalizer;
- optimizer, scheduler, GradScaler;
- RNG와 data cursor;
- policy/conditioner/trainer config;
- frozen foresight artifact의 path, size, SHA-256.

frozen CLaD tensor는 중복 저장하지 않는다. `stage2_latest.pt`는 checkpoint
interval마다 atomic replace되므로 특정 step 비교가 필요하면 다음 저장 전에
별도 snapshot을 보존해야 한다.

## 8. Phase 4: checkpoint restoration과 LIBERO rollout

Python entry point는
[`evaluate_clad_libero.py`](../scripts/evaluate_clad_libero.py), GPU launcher는
[`evaluate_libero.sh`](../scripts/evaluate_libero.sh)다.

### 8.1 Inference-only policy load

[`checkpoint.py`](../src/clad/evaluation/checkpoint.py)는 다음 순서로 복원한다.

1. Stage 2 checkpoint SHA-256 계산 및 memory-mapped CPU load;
2. referenced Stage 1 foresight size/SHA-256 검증;
3. checkpoint 안의 config로 frozen backbone, FiLM, denoiser 재구성;
4. 기본적으로 EMA shadow를 trainable parameter에 복사;
5. action normalizer 복원;
6. 전체 model freeze/eval 후 지정 GPU로 이동.

optimizer state는 GPU에 올리지 않는다. `--weights raw`만 명시했을 때 online
training weight를 사용한다.

### 8.2 Live DecisionNCE와 train/eval identity

[`OnlineDecisionNCEEncoder`](../src/clad/evaluation/online_policy.py)는 feature-cache
manifest에서 model name, source revision, checkpoint hash, camera view를 읽는다.
로컬 DecisionNCE checkpoint hash가 다르면 rollout을 시작하지 않는다.

- task text는 training cache의 feature를 그대로 읽고 instruction 문자열도
  LIBERO benchmark와 정확히 대조한다.
- live RGB frame만 동일한 DecisionNCE adapter로 매 environment step encode한다.
- camera view 집합이 training cache와 정확히 같아야 한다.
- live proprioception은 checkpoint에 기록된 named contract를 따른다. 신규
  checkpoint는 `robot0_joint_pos` 7D 뒤에 `robot0_gripper_qpos` 2D를 연결한다.
  필드가 없는 기존 checkpoint만 gripper qpos, EEF position, EEF quaternion의
  레거시 9D 순서를 사용한다.

### 8.3 Online history buffer

[`OnlineHistoryBuffer`](../src/clad/evaluation/online_policy.py)는 최근 7개
observation과 6개 action을 보관한다. CLaD가 실제로 읽는 것은 6 step 전
observation, 현재 observation, 그리고 그 사이 6개 action이다.

episode reset 직후에는 initial observation을 7번, zero action을 6번 넣어
left padding한다. 실제 action을 실행할 때마다 action과 새 observation을 같이
append하므로 history가 environment와 동기화된다.

### 8.4 Reverse DDPM과 action chunk

`CLaDOnlinePolicy.plan()`은 history를 batch size 1로 만들고 다음을 실행한다.

```text
history → frozen CLaD → g_p/g_s
        → Gaussian action [1,6,7]
        → 100 reverse DDPM steps
        → environment-scale action [1,6,7]
```

기본 `execution_steps=6`이면 여섯 action을 차례로 실행한 뒤 다시 계획한다.
각 action 뒤 새 observation을 DecisionNCE로 encode해 history에 추가한다.

### 8.5 LIBERO episode

[`libero_rollout.py`](../src/clad/evaluation/libero_rollout.py)는 official benchmark
registry에서 task, BDDL, fixed initial state를 읽는다. 동일 task의 pending
rollout을 기본 4개씩 묶고, 한 개면 `DummyVectorEnv`, 여러 개면 공식
`SubprocVectorEnv`로 offscreen environment를 만든다.

1. 각 slot의 seed, reset, fixed initial state 설정;
2. live RGB를 batch DecisionNCE encode하고 slot별 policy/history/generator reset;
3. active slot에 zero action 5회로 physics warmup;
4. active history를 batch로 묶어 한 번의 DDPM sampling으로 action chunk 생성;
5. chunk action을 vector step하고 종료 slot을 active set에서 제거;
6. reward 또는 `check_success()`로 slot별 성공 확인;
7. 성공, termination 또는 policy action 600 steps에서 episode별 종료;
8. wave 결과를 episode 단위로 append/fsync하고 summary 갱신.

task당 rollout `i`는 fixed state `i % num_initial_states`를 쓴다. 중간 중단 후
같은 command를 실행하면 기록된 `(task_id, rollout_id)`를 건너뛴다.
각 slot의 DDPM generator는 해당 episode seed만 사용하므로 batch 위치와 다른
slot의 조기 종료에 의해 random stream이 바뀌지 않는다. 정책 weight는 부모 GPU
process에 한 번만 존재하고 environment subprocess에는 복제하지 않는다.

### 8.6 Evaluation outputs

```text
outputs/clad_evaluation_official/
├── run_identity.json       # checkpoint/cache hash와 protocol
├── episode_results.jsonl   # episode별 append-only record
├── summary.json            # task별 및 전체 success rate
├── eval_console.log        # shell launcher console log
└── videos/                 # --save-videos일 때 MP4
```

`run_identity.json`이 다르면 기존 directory에 다른 checkpoint나 protocol 결과를
섞지 않는다. video frame만 OpenGL row order를 보정하며 policy observation은
원본을 유지한다.

## 9. Artifact dependency와 무결성

| artifact | 생성자 | 직접 의존성 | 검증 방식 |
| --- | --- | --- | --- |
| DecisionNCE cache | cache script | LIBERO HDF5, VLM source/checkpoint | task fingerprint, manifest |
| `stage1_latest.pt` | Stage 1 trainer | cache, model/train config | schema, exact resume state |
| `stage1_foresight.pt` | export script | Stage 1 checkpoint | strict tensor prefixes/schema |
| `stage2_latest.pt` | Stage 2 trainer | foresight artifact, cache, action data | foresight size/SHA-256 |
| evaluation result | evaluator | Stage 2, foresight, cache, LIBERO config | run identity와 세 SHA-256 |

따라서 Stage 2 checkpoint만 복사해서는 평가가 완결되지 않는다. byte-identical
`stage1_foresight.pt`, 일치하는 DecisionNCE checkpoint/cache manifest,
LIBERO source/config도 필요하다.

## 10. Configuration 흐름

각 entry point의 값은 일반적으로 다음 우선순위를 따른다.

```text
dataclass defaults < YAML config < explicit CLI override
```

shell launcher는 dataset/GPU/output처럼 host마다 달라지는 값을 환경변수로
받아 명시적 CLI로 전달한다.

| 영역 | canonical config |
| --- | --- |
| dataset/window | [`configs/data/libero_long.yaml`](../configs/data/libero_long.yaml) |
| cache | [`configs/data/decisionnce_cache.yaml`](../configs/data/decisionnce_cache.yaml) |
| DecisionNCE | [`configs/model/decisionnce.yaml`](../configs/model/decisionnce.yaml) |
| Stage 1 model | [`configs/model/clad_stage1.yaml`](../configs/model/clad_stage1.yaml) |
| Stage 1 trainer | [`configs/train/stage1.yaml`](../configs/train/stage1.yaml) |
| Stage 2 model | [`configs/model/clad_stage2.yaml`](../configs/model/clad_stage2.yaml) |
| Stage 2 trainer | [`configs/train/stage2.yaml`](../configs/train/stage2.yaml) |
| rollout | [`configs/eval/libero_long.yaml`](../configs/eval/libero_long.yaml) |

checkpoint는 resolved config를 내장한다. Stage 2 resume는 trainer, policy,
conditioner config가 정확히 같지 않으면 거부한다. 평가도 checkpoint에 저장된
model config로 architecture를 재구성하므로 현재 YAML을 바꿔도 기존 checkpoint
shape가 조용히 바뀌지 않는다.

## 11. 실행 entry point 요약

| 목적 | 명령/스크립트 |
| --- | --- |
| dataset 검사 | `python scripts/inspect_dataset.py ...` |
| DecisionNCE cache | `python scripts/cache_decisionnce_features.py ...` |
| Stage 1 본 학습 | `./scripts/train_stage1.sh` |
| compact foresight export | `python scripts/export_stage1_foresight.py ...` |
| Stage 2 본 학습 | `./scripts/train_stage2.sh` |
| LIBERO path 설정 | `python scripts/configure_libero.py ...` |
| GPU 지정 평가 | `./scripts/evaluate_libero.sh GPU_ID [options]` |

설치 명령은 [`libero_installation.md`](libero_installation.md), Stage 1 실행은
[`stage1_training.md`](stage1_training.md), Stage 2 실행은
[`stage2_training.md`](stage2_training.md), 평가 protocol은
[`libero_evaluation.md`](libero_evaluation.md)에 상세히 나뉘어 있다.

## 12. 방어적 검증과 실패 방식

이 구현은 잘못된 실험이 조용히 진행되는 것보다 시작 전에 실패하는 쪽을
선택한다.

- dataset temporal length와 tensor rank 불일치 거부;
- episode 경계를 넘는 window 생성 금지;
- cache task/camera/checkpoint identity 불일치 거부;
- Stage 1/2 horizon, hidden, action dimension 불일치 거부;
- frozen CLaD parameter와 eval mode 강제;
- unfitted action normalizer로 Stage 2 시작 금지;
- NaN/Inf image feature, proprioception, action, gradient 감지;
- 반복 AMP overflow fail-fast;
- checkpoint와 evaluation JSON atomic write;
- 서로 다른 run identity의 resume 거부.

이 검증은 논문의 알고리즘 요소는 아니지만 긴 학습 후 잘못된 결과를 발견하는
위험을 줄이는 핵심 재현 인프라다.

## 13. 확장 지점

### Multi-camera

data/cache config에 두 번째 camera key를 추가해 cache를 다시 만들면 dataset과
batch interface는 그대로 유지된다. 현재 model fusion은 mean이므로 다른 fusion을
연구하려면 `FeatureFiLM.fuse_views()`와 config validation을 확장해야 한다.
기존 one-camera checkpoint와 multi-camera cache는 identity 검증 때문에 섞이지
않는다.

### Ablation

Stage 1 forward가 transition, dynamics, modality loss를 모두 반환하므로 attention
방향, reconstruction weight, action mask 실험을 분리하기 쉽다. Stage 2
conditioner도 `g_p`, `g_s`를 독립적으로 반환하므로 modality ablation을 추가할
수 있다. 다만 논문 표를 정확히 재현하려면 checkpoint selection과 동일 rollout
protocol도 함께 정의해야 한다.

### 다른 LIBERO suite

rollout config는 suite name을 받을 수 있지만 학습 cache와 task instruction이
suite를 완전히 포함해야 한다. 단순히 평가 suite 이름만 바꾸면 cache coverage
검증에서 실패한다. Spatial/Object/Goal 결과를 재현하려면 각 suite data cache와
학습 run이 별도로 필요하다.

## 14. 현재 완료 범위와 다음 작업

core data/cache, Stage 1, compact bridge, Stage 2, EMA restore, LIBERO rollout은
구현됐다. 남은 연구 구현은 다음과 같다.

1. Stage 2 최종 checkpoint까지 학습 완료;
2. single-checkpoint 10×50 본 평가;
3. 필요 시 top-3 snapshot retention/selection 자동화;
4. 논문 ablation과 다른 LIBERO suite 확장;
5. 공식 구현 정보가 공개되면
   [`reproduction_assumptions.md`](reproduction_assumptions.md)의 high-impact
   결정을 대조하고 config를 갱신.
