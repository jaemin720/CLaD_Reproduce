# CLaD 재현 가정과 미명시 구현 결정

이 문서는 논문 **CLaD: Planning with Grounded Foresight via Cross-Modal
Latent Dynamics**의 arXiv v1을 기준으로, 논문이 명시한 사실과 이 저장소가
추가로 결정한 구현을 분리해 기록한다. 목적은 현재 코드를 논문 저자의
공식 구현과 동일하다고 오해하지 않게 하고, 결과가 다를 때 어떤 선택을
우선 검증해야 하는지 남기는 것이다.

여기서 “미명시”는 선택이 잘못됐다는 뜻이 아니다. 논문의 식과 parameter
budget을 실행 가능한 시스템으로 만들기 위해 필요한 정보가 공개되지 않았고,
그 빈 부분을 재현 가능한 기본값으로 고정했다는 뜻이다.

## 1. 표기와 판단 기준

- **논문 명시**: 본문, supplementary, 식 또는 표에 직접 적힌 내용이다.
- **수식 해석**: 문장과 식 사이의 모호성을 한 가지 방식으로 해석한 것이다.
- **재현 가정**: 성능이나 checkpoint shape에 영향을 줄 수 있는 선택이다.
- **엔지니어링 선택**: cache, hash, resume처럼 실험 의미를 보존하기 위한
  구현이다. 일반적으로 모델 의미에는 영향을 주지 않는다.
- **미구현**: 논문에 결과는 있지만 현재 저장소가 자동화하지 않은 실험이다.

논문이 숫자를 명시했더라도 주변 세부사항까지 명시한 것은 아니다. 예를 들어
`H=1024`는 명시됐지만 이를 처리하는 attention layer 수와 head 수는 미명시다.

## 2. 논문이 직접 명시한 기준

| 항목 | 논문 기준 | 현재 구현 |
| --- | --- | --- |
| benchmark | LIBERO-LONG, 10 tasks | `libero_10` |
| latent hidden dimension | `H=1024` | 1024 |
| state token 수 | `Np=Ns=4` | 각 4개 |
| action/foresight horizon | `τ=6` | 6 |
| stochastic action mask | `r=0.3` | token별 확률 0.3 |
| Stage 1 EMA momentum | `m=0.995` | 0.995 |
| reconstruction weight | `λrecon=0.1` | 0.1 |
| Stage 1 학습량 | 25K steps, batch 128 | 25,000 successful updates, batch 128 |
| Stage 2 학습량 | 200K steps, batch 128 | 200,000 successful updates, batch 128 |
| 저자 보고 학습 환경 | RTX 4090, Stage 1 약 2시간, Stage 2 약 20시간 | 실행 host에 따라 달라지며 보장하지 않음 |
| VLM 계열 | DecisionNCE | DecisionNCE-T를 선택 |
| parameter budget | VLM 0.1B, CLaD 0.33B, policy 0.23B | 해당 budget에 가깝게 architecture를 결정 |
| single-checkpoint 평가 | task당 50 rollouts | 기본값 50 |
| top-3 평가 | 상위 3 checkpoints 각각 20 rollouts 평균 | 개별 평가는 가능, 자동 선정은 미구현 |

논문은 Stage 1/2 “step”을 optimizer update로 볼지 AMP overflow를 포함한
시도 횟수로 볼지 설명하지 않는다. 이 구현은 실제 parameter가 갱신된 횟수를
step으로 센다. 논문의 시간과 GPU 수치는 저자 환경의 보고값이며 architecture가
같다는 증거나 다른 GPU에서의 예상 시간으로 사용하지 않는다.

## 3. 데이터와 DecisionNCE

### A01. 학습 sample의 시간 인덱스

- **논문 명시**: `t-τ`, `t`, `t+τ` 상태와 `a(t-τ):t` action history를
  사용해 `t+τ`를 예측한다.
- **현재 선택**: 한 anchor `t`에 대해 상태는 정확히 `t-6`, `t`, `t+6`,
  과거 action은 Python 구간 `[t-6,t)`, Stage 2 target action은 `[t,t+6)`을
  사용한다.
- **추가 결정**: episode 경계 padding 없이 세 상태와 두 action 구간이 모두
  존재하는 anchor만 사용한다. 짧은 episode는 strict mode에서 거부한다.
- **영향**: 경계 frame을 padding해서 사용하는 구현보다 window 수와 초기/말기
  상태 분포가 다르다.

구현은 [`sequence_sampler.py`](../src/clad/data/sequence_sampler.py)와
[`libero_dataset.py`](../src/clad/data/libero_dataset.py)에 있다.

### A02. proprioception 정의

- **논문 명시**: `p_t`를 joint angle과 velocity를 포함하는 proprioceptive
  state로 서술하지만 정확한 `Dp`와 LIBERO field 조합은 제시하지 않는다.
- **현재 선택**: demonstration의 `robot_states` 9차원 vector를 사용한다.
  online rollout에서는 `gripper qpos(2) + EEF position(3) + EEF
  quaternion(4)`을 연결해 같은 9차원을 복원한다.
- **정합성 상태**: **미해결 차이**다. 현재 9차원 값에는 논문 문장의 full arm
  joint velocity가 포함되지 않으므로 그 서술과 문자 그대로 같지 않다. 다만
  논문은 `Dp`, 정확한 field, 좌표계와 전처리를 공개하지 않았고 현재 HDF5의
  `robot_states` contract는 online에서 정확히 재구성 가능하므로, 추측으로
  입력 schema와 이미 학습된 checkpoint를 바꾸지 않는다.
- **중요성**: 현재 state는 full arm joint angle/velocity vector가 아니다.
  저자 구현이 다른 proprio schema를 사용했다면 직접적인 재현 차이다.

### A03. action 정의

- **논문 명시**: horizon은 6이지만 action dimension과 LIBERO action field의
  세부 의미는 제시하지 않는다.
- **현재 선택**: HDF5의 7차원 `actions`를 그대로 사용한다. 별도 action
  conversion이나 delta/absolute 변환은 하지 않는다.
- **검증**: dataset, Stage 1, Stage 2, rollout 모두 `Da=7`을 강제한다.

### A04. 카메라와 영상 전처리

- **논문 명시**: semantic state에 image/VLM embedding을 사용하지만 카메라
  이름, view 수, image resolution, augmentation은 제시하지 않는다.
- **현재 선택**: `obs/agentview_rgb` 한 개가 기본이다. 학습 cache와 online
  rollout 모두 이 view를 사용하며, online render 크기는 128×128이다.
- **전처리**: resize, crop, normalization은 upstream DecisionNCE에 맡기고
  이 저장소는 RGB `uint8` layout만 맞춘다. 학습 augmentation은 적용하지 않는다.
- **확장**: 여러 view cache를 만들면 view별로 DecisionNCE를 실행한 뒤 feature를
  산술 평균한다. attention 기반 camera fusion은 구현하지 않았다.
- **영상 저장**: OpenGL row order 보정은 MP4 copy에만 적용한다. policy 입력
  frame은 뒤집지 않는다.

### A05. demonstration sampling과 split

- **논문 명시**: standard LIBERO protocol을 따른다고만 설명한다.
- **현재 선택**: 발견된 모든 task file과 모든 `demo_N`을 학습 source로 쓰며,
  validation split은 만들지 않는다. valid window 전체를 섞어 균등 sampling한다.
- **영향**: task-balanced sampling이 아니므로 더 긴 demonstration 또는 더 많은
  valid window를 가진 task가 더 자주 등장한다. top checkpoint를 validation으로
  선정하는 절차도 없다.

### A06. DecisionNCE 변형과 checkpoint

- **논문 명시**: DecisionNCE를 VLM으로 사용한다고만 적고 `P/T` 변형이나
  downstream checkpoint를 식별하지 않는다.
- **현재 선택**: transition direction과 language를 맞추는 `DecisionNCE-T`를
  기본으로 사용한다. upstream source는 Git revision
  `ebdc585c5e6833ec3a2ba77f801b15c74d7a28f8`, 현재 사용한 공식 checkpoint
  SHA-256은
  `75242de7e60c4007b186ddcdde220f6ca981e814775df564673af7782afcd03f`다.
- **관찰된 interface**: image/text feature가 모두 1024차원이다. 이는 논문이
  명시한 값이 아니라 선택한 upstream model에서 확인한 값이다.
- **높은 민감도**: 저자가 DecisionNCE-P 또는 별도 Robo-MUTUAL checkpoint를
  사용했다면 semantic representation 전체가 달라진다.

### A07. offline feature cache

- **논문 명시 여부**: 없음.
- **현재 선택**: frozen VLM을 Stage 1/2 학습 중 반복 실행하지 않고 모든 image와
  task text feature를 float16 HDF5 cache로 미리 계산한다.
- **동등성 전제**: upstream encoder가 eval mode이고 augmentation이 없으므로
  같은 checkpoint/input에 대한 online encoding과 의미상 동일하다고 본다.
- **보호 장치**: dataset identity, camera keys, source revision, checkpoint hash,
  dtype을 manifest에 기록하고 불일치 cache를 거부한다.

## 4. Stage 1: Cross-Modal Latent Dynamics

### B01. semantic FiLM의 형태

- **논문 명시**: `s_t = FiLM(v_t,l)`.
- **현재 선택**: language feature 하나가 visual dimension별 `delta_scale`과
  `shift`를 생성하고 `v*(1+delta_scale)+shift`를 계산한다.
- **초기화**: affine weight와 bias를 0으로 초기화해 시작 시 identity가 되게 한다.
- **미명시 부분**: FiLM network 깊이, activation, initialization은 논문에 없다.

### B02. state/action tokenizer

- **논문 명시**: `f_p`, `f_s`는 MLP이고 출력은 각각 4개 token이다. action도
  `f_a`로 encode한다.
- **현재 선택**: `Linear → GELU → Dropout → Linear` 두 층 MLP가 한 vector를
  `N×H`로 직접 펼친다. 별도 learned query나 transformer tokenizer는 없다.
- **action**: 각 7D action을 같은 형태의 MLP로 하나의 `H` token으로 바꾸고,
  6개 위치에 learned positional embedding을 더한다.
- **mask**: 각 action token을 독립 Bernoulli(`p=0.3`)로 선택해 하나의 learned
  mask token으로 교체한다. sample마다 정확히 30%를 보장하거나 최소 한 token을
  강제하지 않는다.

### B03. cross-attention architecture

- **논문 명시**: 식 (7)--(9)의 세 cross-attention 관계와 `H=1024`.
- **현재 선택**: modality transition 두 개와 asymmetric transition에 각각
  독립적인 8-layer pre-norm stack을 사용한다. 각 layer는 16 heads, head dimension
  64, `4H` GELU FFN, dropout 0이다.
- **문맥 구성**: 현재 state token이 query이고 `[past state tokens; six action
  tokens]`가 key/value다. state token용 별도 temporal embedding은 없다.
- **선정 이유**: 세 stack을 이 크기로 구성하면 trainable CLaD가 논문의
  약 0.33B budget에 가까워진다. 저자 architecture를 확인한 결과는 아니다.
- **높은 민감도**: layer/head 수는 representation과 checkpoint shape를 모두
  바꾼다.

### B04. learnable pooling

- **논문 명시**: learnable `q_out`을 쓰는 Pool 연산과 Perceiver 계열 인용.
- **현재 해석**: 하나의 learned query가 `z_(p→s)` token을 대상으로 한 번의
  pre-norm cross-attention/FFN block을 수행하고 `[B,H]` `z_dyn`을 만든다.
- **대안**: query를 단순 weighted pooling에만 쓰거나 multi-layer readout을
  구성할 수도 있으나 논문은 구분하지 않는다.

### B05. foresight predictor와 reconstruction head

- **논문 명시**: modality별 lightweight MLP `g_p`, `g_s`, `h_p`, `h_s`.
- **현재 선택**: 네 module 모두 `Linear → GELU → Dropout → Linear`, hidden
  width 1024, dropout 0이다.
- **출력**: foresight는 modality별 `[B,1024]`; reconstruction은 proprio
  `[B,9]`, semantic visual feature `[B,1024]`다.

### B06. EMA target token pooling

- **논문 모호성**: 식 (15)--(16)의 target은 `[B,H]`지만 `f_p`, `f_s`는 앞에서
  `N×H` token을 출력한다. token을 vector로 바꾸는 방법이 없다.
- **현재 선택**: EMA encoder의 네 token을 단순 평균한다. target 전용 projection은
  추가하지 않는다.
- **EMA 범위**: online proprio tokenizer와 semantic FiLM/tokenizer의 frozen
  copy만 갱신한다. transition/dynamics/predictor의 target copy는 만들지 않는다.

### B07. 식 (17)의 정규화 해석

- **논문 모호성**: 본문은 “L2-normalized embeddings”라고 복수로 서술하지만,
  출력된 식 (17)은 EMA target `z_bar`만 norm으로 나눈 형태다.
- **현재 선택**: 수식을 문자 그대로 따라 stop-gradient target만 L2 normalize하고
  prediction은 normalize하지 않는다. 각 modality는 squared L2를 feature dimension에
  합산한 뒤 batch 평균한다.
- **높은 민감도**: prediction까지 normalize하는 대안은 scale gradient와 loss
  범위를 바꾼다. 공식 코드가 공개되면 가장 먼저 대조할 항목이다.

### B08. semantic reconstruction target

- **논문 명시/모호성**: 식 (18)은 `s_v^(t+τ)`를 복원하지만 pixel인지 VLM
  visual component인지 정확한 tensor contract를 제시하지 않는다.
- **현재 해석**: future DecisionNCE visual feature를 view-fuse한 값, 즉 language
  FiLM 이전의 1024D vector를 L1 target으로 사용한다. pixel reconstruction은
  하지 않는다.
- **reduction**: L1 norm을 feature dimension에 합산한 뒤 batch 평균한다.

## 5. Stage 2: foresight-conditioned Diffusion Policy

### C01. 식 (20)의 observation encoder

- **논문 명시**: modality별 `e_p`, `e_s`로 current observation을 encode한다.
- **현재 선택**: 새 encoder를 만들지 않고 학습된 Stage 1 online state encoder를
  frozen 상태로 재사용한다. 네 token은 평균해 각각 `[B,1024]`로 만든다.
- **장점/위험**: foresight와 observation이 같은 latent space에 있지만, 저자
  구현이 별도 encoder를 사용했다면 parameterization이 다르다.

### C02. 식 (21)의 modality 대응과 FiLM

- **논문 모호성**: typeset 식에서 두 branch의 `z_hat` modality subscript가
  사라져 있다.
- **현재 해석**: `z_hat_p`는 proprio observation으로, `z_hat_s`는 semantic
  observation으로 각각 조절한다.
- **FiLM 선택**: modality별 단일 affine layer, identity initialization,
  dropout 0을 사용한다. 두 결과를 연결하면 2048D global condition이 된다.

### C03. diffusion denoiser

- **논문 명시**: standard DDPM noise-prediction objective와 policy 약 0.23B.
- **현재 선택**: Diffusion Policy 형태의 conditional 1D U-Net을 이 저장소에서
  PyTorch로 구현했다. 외부 Diffusion Policy source를 복사하지 않았다.
- **구조**: widths `[512,1024,1536]`, kernel 5, GroupNorm 8 groups, Mish,
  sinusoidal timestep embedding 256D, residual block마다 global scale/shift
  conditioning을 사용한다.
- **선정 이유**: denoiser 227,412,743 parameters와 Stage 2 FiLM을 합쳐
  231,611,143 trainable parameters가 되어 논문의 0.23B budget에 맞는다.

### C04. horizon 6의 내부 padding

- **논문 명시**: 외부 action horizon은 6.
- **현재 선택**: 두 stride-2 downsample을 위해 U-Net 내부 temporal length를
  오른쪽 zero padding으로 6에서 8로 늘리고 출력은 다시 6으로 자른다.
- **영향**: 외부 tensor/loss/action contract는 `[B,6,7]`로 유지되지만 padded
  boundary의 convolution context는 구현별 차이가 될 수 있다.

### C05. DDPM schedule과 sampling

- **논문 미명시**: timestep 수, beta schedule, reverse variance, sample clipping.
- **현재 선택**: 100 steps, `squaredcos_cap_v2`, `cosine_s=0.008`,
  `max_beta=0.999`, fixed-small posterior variance를 사용한다. 학습 timestep은
  `[0,99]` uniform sampling한다.
- **추론**: Gaussian noise에서 시작해 100개 reverse step을 모두 실행하며
  predicted clean normalized action을 `[-1,1]`로 clip한다.

### C06. action normalization

- **논문 미명시**: 없음.
- **현재 선택**: 모든 training demonstration action을 한 번씩 scan해 dimension별
  global min/max를 구하고 `[-1,1]`로 선형 변환한다. quantile normalization이나
  task별 통계는 사용하지 않는다.
- **checkpoint**: min/max/scale/bias를 Stage 2 checkpoint에 저장하고 평가 시
  같은 값으로 unnormalize한다.

### C07. Stage 2에서 action masking

- **현재 선택**: frozen CLaD에 실제 관측된 action history를 넣고 stochastic
  mask는 항상 끈다. action masking은 Stage 1 representation 학습에만 사용한다.
- **근거**: 평가 때 미래 action을 가리는 목적이 아니며, Stage 2 conditioning은
  deterministic해야 한다고 해석했다.

## 6. 최적화와 checkpoint

### D01. optimizer와 learning-rate schedule

논문은 두 stage의 steps와 batch size 외 optimizer를 제시하지 않는다.

| 항목 | Stage 1 기본값 | Stage 2 기본값 |
| --- | --- | --- |
| optimizer | AdamW | AdamW |
| learning rate | `1e-4` | `1e-4` |
| betas | `0.9/0.95` | `0.95/0.999` |
| weight decay | `0.01` | `1e-6` |
| epsilon | PyTorch 기본값 | `1e-8` |
| warmup | 500 updates | 500 updates |
| decay | cosine to 1% of base LR | cosine to 1% of base LR |
| grad clipping | global norm 1.0 | global norm 1.0 |

### D02. precision과 step 의미

- CUDA에서는 fp16 autocast와 dynamic GradScaler를 사용하고 initial scale은
  2048이다. frozen Stage 1 backbone도 Stage 2 CUDA 실행에서 기본 fp16이다.
- AMP overflow로 optimizer가 실행되지 않은 attempt는 paper step에 포함하지
  않는다. scheduler, Stage 1 target EMA, Stage 2 policy EMA도 성공한 update에만
  진행한다.
- 16회 연속 skip이면 잘못된 학습을 계속하지 않고 중단한다.

### D03. Stage 2 policy EMA

- **논문 미명시**: Stage 2 policy EMA 사용 여부와 decay가 없다. 식 (14)의
  EMA target encoder와는 별개의 문제다.
- **현재 선택**: trainable FiLM/U-Net에 inverse-gamma warmup EMA를 적용한다.
  `inv_gamma=1`, `power=0.75`, 최대 decay `0.9999`이며 평가 기본 weight는 EMA다.
- **비교 가능성**: `--weights raw`로 online parameter를 평가할 수 있다.

### D04. reproducible resume

RNG, optimizer, scheduler, scaler, successful/attempt step, shuffled batch cursor를
checkpoint에 저장한다. Stage 1 checkpoint에서 Stage 2에 필요 없는 optimizer,
EMA target, reconstruction state는 compact foresight artifact를 내보낼 때 제거한다.
Stage 2 checkpoint는 이 artifact를 내장하지 않고 size와 SHA-256으로 참조한다.
이는 엔지니어링 선택이며 모델 수학을 바꾸지 않는다.

## 7. LIBERO rollout과 metric

### E01. checkpoint protocol

- single-checkpoint 기본 평가는 EMA weight로 task당 50 rollouts를 수행한다.
- 논문의 top-3×20 protocol은 checkpoint 보존/선정 기준이 공개되지 않아 자동화하지
  않았다. 사용자가 선정한 세 checkpoint를 별도 output directory에서 각각 20회
  평가할 수만 있다.

### E02. initial state와 seed

- LIBERO benchmark가 제공하는 fixed initial state를 순서대로 사용하고, rollout
  수가 더 많으면 modulo로 순환한다.
- episode seed는 `42 + task_id*100000 + rollout_id`다.
- paper는 이 state 선택 순서와 seed 공식을 명시하지 않는다.

### E03. warmup과 online history

- environment reset 및 fixed state 적용 후 zero action 5회를 실행한다.
- episode 시작 history는 initial observation 반복과 zero action으로 왼쪽
  padding하고, warmup 중 실제 observation을 계속 반영한다.
- paper에는 warmup/history padding이 없으며 standard LIBERO 관행과 online
  tensor contract를 맞추기 위한 선택이다.

### E04. replanning 간격

- **논문 명시**: policy가 6-action chunk를 생성한다.
- **논문 미명시**: 몇 action을 실행한 뒤 다시 계획하는지 없다.
- **현재 기본값**: 6개를 모두 실행한 뒤 재계획한다. `execution_steps=1`이면
  매 control step마다 100-step DDPM을 다시 수행한다.
- **높은 민감도**: 이 값은 closed-loop feedback 빈도, 속도, 성공률을 크게
  바꿀 수 있다.

### E05. 종료와 성공 판정

- policy action 기준 최대 600 steps이며 5개 warmup step은 이 수치에 포함하지
  않는다.
- action은 environment 전달 전 `[-1,1]`로 clip한다.
- reward가 양수이거나 environment `check_success()`가 참이면 성공으로 종료한다.
- paper는 최대 step, clipping, 성공 API를 구체적으로 제시하지 않는다.

### E06. 병렬 평가

- **논문 미명시**: rollout environment 수와 evaluation process 구조가 없다.
- **현재 기본값**: 같은 task의 rollout을 `num_envs=4`인 synchronous
  `SubprocVectorEnv` wave로 실행한다. 하나의 부모 process만 GPU의 DecisionNCE와
  CLaD policy를 소유하고 observation/action inference를 batch 처리한다.
- environment seed, initial-state ID, history와 diffusion generator는 episode별로
  독립이다. 종료한 slot은 다음 vector step과 policy batch에서 제거한다.
- `num_envs`는 metric 정의가 아니라 실행 protocol이지만 batched GPU 연산의
  부동소수점 차이를 추적할 수 있도록 run identity에 포함한다.
- `num_envs=1`은 `DummyVectorEnv`를 사용하는 reference/debug 경로다.

### E07. latency의 범위

현재 `inference_seconds`는 history tensor가 준비된 뒤 batch 전체의 diffusion
policy 100-step sampling과 CUDA synchronization에 걸린 wall time이다. 같은
batch의 active episode에는 동일한 planning latency를 기록한다. 다음은 제외한다.

- live DecisionNCE image encoding;
- LIBERO environment step과 rendering;
- 영상 저장과 JSON 기록.

따라서 이 값은 논문 표의 end-to-end planning latency와 동일하다고 단정할 수
없다. 공정한 latency 비교가 필요하면 VLM encoding을 포함한 별도 wall-clock
측정이 필요하다.

### E08. 집계와 resume

- task별 success rate의 단순 평균을 `macro_task_success_rate`로 기록한다.
- 동시에 전체 episode 기준 weighted rate도 기록한다.
- checkpoint/cache/config hash가 같은 run만 같은 디렉터리에서 resume할 수 있다.
- MP4의 vertical flip은 시각화 artifact에만 적용되고 metric/policy에는 영향을
  주지 않는다.

## 8. 재현 결과에 민감한 우선 확인 항목

공식 코드나 저자 답변을 얻는다면 다음 순서로 대조하는 것이 좋다.

1. DecisionNCE `P/T` 변형과 실제 checkpoint;
2. proprioception field와 camera view/preprocessing;
3. cross-attention layer/head 수와 tokenizer 구조;
4. 식 (17)에서 prediction 정규화 여부와 EMA token pooling;
5. semantic reconstruction target이 pixel, VLM visual feature, semantic feature 중
   무엇인지;
6. diffusion U-Net, timestep/beta schedule, action normalization;
7. action chunk 중 실제 실행 개수와 평가 max steps;
8. optimizer, Stage 2 EMA, checkpoint selection protocol;
9. task-balanced sampling 또는 train/validation split 사용 여부.

앞의 1--7은 결과 차이를 크게 만들 가능성이 높다. cache 저장 dtype, atomic
checkpoint, JSONL logging과 같은 항목은 재현 안정성에는 중요하지만 모델 성능
차이의 첫 원인으로 볼 가능성은 낮다.

## 9. 아직 자동화하지 않은 논문 실험

- top-3 checkpoint 자동 보존, validation 및 선정;
- proprio-only, semantic-only, policy-only conditioning ablation;
- reconstruction loss 제거 ablation;
- symmetric/reversed cross-attention ablation;
- action-free/heavy-mask/curriculum ablation;
- LIBERO-Spatial/Object/Goal 전체 training/evaluation;
- integrated gradients 및 UMAP 분석.

이 항목들은 논문의 core two-stage training과 rollout을 실행하는 데 필수는
아니지만 논문의 ablation 및 supplementary 결과를 재현하려면 추가 구현이
필요하다.

## 10. 결과를 보고할 때의 권장 표현

현재 저장소의 결과는 “공식 CLaD 재현”보다 다음처럼 기술하는 것이 정확하다.

> CLaD 논문의 공개된 식과 hyperparameter를 따르되, 공개되지 않은 architecture,
> DecisionNCE checkpoint, optimization 및 deployment 세부사항을 문서화된
> 재현 가정으로 보완한 비공식 구현 결과다.

각 결과에는 최소한 다음 artifact identity를 함께 보존해야 한다.

- Git commit과 두 submodule revision;
- DecisionNCE checkpoint 및 feature-cache manifest SHA-256;
- Stage 1 foresight 및 Stage 2 policy checkpoint SHA-256;
- resolved model/train/eval configuration;
- raw 또는 EMA weight 여부;
- task IDs, rollout 수, execution steps, seed;
- software lock 또는 container identity.

전체 코드 흐름과 각 artifact가 만들어지는 위치는
[`framework_implementation.md`](framework_implementation.md)에 설명한다.
