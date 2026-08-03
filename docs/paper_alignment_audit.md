# CLaD 논문 정합성 감사

이 문서는 **CLaD: Planning with Grounded Foresight via Cross-Modal Latent
Dynamics**, arXiv:2603.29409v1 (2026-03-31)을 기준으로 현재 기본 구현을
대조한 결과다. 검토 기준은 논문 본문 식 (5)--(22), Section 5.1, supplementary
Sections 8--10이다.

- 감사일: 2026-08-03
- 코드 기준점: `9ec0ee3` (`docs: clarify camera-view reproduction assumption`)
- 논문 원문: <https://arxiv.org/abs/2603.29409>
- 구현 가정 상세: [`reproduction_assumptions.md`](reproduction_assumptions.md)
- tensor 흐름: [`framework_implementation.md`](framework_implementation.md)

이 감사에서 `일치`는 논문이 공개한 범위 안에서 tensor 관계와 기본값이
같다는 뜻이다. 저자 코드와 checkpoint가 공개되지 않았으므로 미명시 내부
구조까지 동일하다는 뜻은 아니다.

## 1. 명시사항 대조표

| 논문 항목 | 현재 구현 | 판정 |
| --- | --- | --- |
| semantic state `s=FiLM(v,l)`; 식 (5)--(6) | frozen DecisionNCE image/text feature, language-conditioned FiLM, modality MLP tokenization | 일치 |
| `H=1024`, `Np=Ns=4` | 기본 config가 1024, 4, 4 | 일치 |
| horizon `tau=6` | data window, CLaD history, future target, policy action chunk가 모두 6 | 일치 |
| transition 식 (7)--(8) | current state가 query, `[past state; past actions]`가 key/value | 일치 |
| stochastic action masking | Stage 1 train에서 learned mask token으로 교체 | 일치 |
| mask ratio `r=0.3` | token별 Bernoulli 확률 0.3 | 일치 |
| proprio transition이 semantic transition을 query; 식 (9) | `CrossModalDynamicsEncoder(proprio, semantic)` | 일치 |
| learned `q_out` pooling; 식 (10) | 한 learned query의 cross-attention readout | 호환 해석 |
| modality별 foresight `z_hat_p`, `z_hat_s`; 식 (11)--(13) | `z_dyn`에서 두 MLP predictor | 일치 |
| EMA targets에 `f_p`, `f_s`, semantic FiLM 포함; 식 (14)--(16) | semantic FiLM/tokenizer와 proprio tokenizer의 EMA copy | 일치 |
| EMA momentum `m=0.995` | 0.995, successful optimizer update 뒤 갱신 | 일치 |
| 식 (17) normalized-target MSE | stop-gradient target을 L2 normalize한 squared L2 | 수식 그대로 일치 |
| 식 (18) modality reconstruction L1 | future proprio와 pre-language-FiLM VLM visual feature 복원 | 호환 해석 |
| `L=L_latent+0.1 L_recon`; 식 (19) | reconstruction weight 0.1 | 일치 |
| frozen CLaD foresight와 observation FiLM; 식 (20)--(21) | frozen Stage 1 encoder를 observation encoder로 재사용하고 modality별 FiLM | 호환 해석 |
| DDPM epsilon prediction; 식 (22) | noisy action과 timestep, `g_p`, `g_s` 조건의 MSE noise objective | 일치 |
| Stage 1: 25K steps, batch 128 | 25,000 successful updates, batch 128 | 일치 |
| Stage 2: 200K steps, batch 128 | 200,000 successful updates, batch 128 | 일치 |
| model budget: VLM 0.1B, CLaD 0.33B, policy 0.23B | selected VLM 약 0.1B, trainable Stage 1 약 0.335B, Stage 2 약 0.232B | 반올림 budget 일치 |
| LIBERO-LONG, 10 tasks | `libero_10` data/benchmark suite | 일치 |
| single checkpoint: 50 rollouts | task당 50 기본값 | 일치 |
| top-3 checkpoints: 20-rollout protocol | 개별 checkpoint 평가는 가능하나 선정/집계 자동화 없음 | 부분 구현 |

## 2. 확인된 미해결 차이

### 2.1 Proprioception은 논문 문장과 문자 그대로 같지 않다

논문은 `p_t`가 joint angles와 velocities를 포함한다고 서술한다. 현재 구현은
공식 LIBERO HDF5의 9D `robot_states`, 즉 gripper qpos 2D, EEF position 3D,
EEF quaternion 4D를 사용한다. full arm joint velocity는 포함하지 않는다.

이 차이는 숨기지 않고 `configs/model/clad_stage1.yaml`과
[`reproduction_assumptions.md`](reproduction_assumptions.md)에 표시했다. 그러나
논문에는 `Dp`, field 순서, velocity 정의, 좌표계가 없어 안전한 대체 tensor를
구성할 수 없다. 이미 생성한 Stage 1/2 checkpoint의 입력 shape도 바뀌므로 이번
감사에서 모델 코드는 변경하지 않았다. 저자 정보가 생기면 가장 먼저 확인해야
할 항목이다.

### 2.2 논문이 공개하지 않은 고영향 선택

다음은 불일치로 단정할 수 없지만 성능 재현에 큰 영향을 줄 수 있다.

- DecisionNCE-P/T 및 실제 checkpoint;
- camera view 수와 image resolution/augmentation;
- state/action MLP 깊이, attention layer/head 수, target-token pooling;
- 식 (17) 문장의 “normalized embeddings”를 prediction에도 적용하는지;
- 식 (18)의 정확한 semantic reconstruction target;
- Stage 2 observation encoder와 FiLM 내부 구조;
- diffusion U-Net, noise schedule, inference step 수, action normalization;
- optimizer, learning-rate schedule, Stage 2 policy EMA;
- 한 action chunk에서 실제 실행하는 action 수, episode horizon, seeds;
- checkpoint selection과 task/window sampling 방식.

현재 구현은 이 항목들을 configurable하고 별도 재현 가정으로 문서화한다.
논문 parameter budget에 맞추기 위해 architecture를 고른 경우도 저자 구조와
동일하다는 증거로 취급하지 않는다.

## 3. 아직 재현하지 않은 논문 결과

- top-3 checkpoint 자동 보존, validation, 선정 및 집계;
- proprio-only, semantic-only, policy-only modality ablation;
- reconstruction-loss 제거와 attention 방향 ablation;
- action-free, heavy-mask, curriculum ablation;
- LIBERO-Spatial, Object, Goal 학습 및 50-rollout 평가;
- UMAP 및 integrated-gradients 시각화;
- 논문의 25 Hz, 0.012 s planning time, 4 GB memory를 동일한 측정 범위로
  재현하는 benchmark.

따라서 core two-stage algorithm과 LIBERO-LONG rollout 경로는 구현됐지만,
논문의 모든 표와 supplementary 분석이 재현 완료된 것은 아니다.

## 4. 이번 감사의 파일별 변경 기록

| 파일 | 변경 |
| --- | --- |
| `README.md` | 정합성 감사 문서 링크 추가 |
| `configs/data/libero_long.yaml` | 9D `robot_states`와 논문 proprio 서술의 차이 표시 |
| `configs/model/clad_stage1.yaml` | 논문 명시값과 data/upstream 유래 dimension 주석 분리, proprio 차이 표시 |
| `configs/eval/libero_long.yaml` | 논문의 50 rollouts와 LIBERO/재현 평가 선택 분리 |
| `docs/reproduction_assumptions.md` | proprioception을 미해결 차이로 명시 |
| `docs/libero_evaluation.md` | 600 steps, 128x128, fixed-state/seed가 논문 명시값이 아님을 명확화 |
| `docs/stage2_conditioning.md` | 이미 구현된 downstream trainer/evaluator 상태 반영 |
| `docs/stage2_diffusion.md` | 완료된 full-width GPU smoke 상태 반영 |
| `docs/stage2_training.md` | 구현 및 smoke 완료 상태 반영, 평가 문서 연결 |
| `docs/implementation_plan.md` | 오래된 “다음 trainer 단계” 제거, LIBERO 관행과 논문 protocol 분리 |
| `docs/paper_alignment_audit.md` | 본 감사표, 미해결 차이, 미구현 범위와 변경 기록 추가 |
| `tests/test_paper_alignment.py` | 논문이 직접 제시한 기본 수치의 regression test 추가 |

모델 수학이나 기존 checkpoint contract는 이번 감사에서 변경하지 않았다.

## 5. 후속 확인 순서

저자 코드, appendix 개정판 또는 저자 답변이 공개되면 다음 순서로 갱신한다.

1. proprioception schema와 camera/view contract;
2. DecisionNCE variant/checkpoint;
3. Stage 1 tokenizer, attention, pooling, normalization, reconstruction target;
4. Stage 2 encoder, FiLM, diffusion, normalization;
5. optimizer, EMA, rollout 및 checkpoint-selection protocol.

각 변경은 기존 checkpoint와 호환되는지 먼저 판단하고, 호환되지 않으면 config
schema와 artifact 이름을 분리해 과거 결과와 섞이지 않게 해야 한다.
