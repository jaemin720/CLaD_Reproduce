# OpenVLA 방식 LIBERO 고해상도 재렌더링

이 저장소는 LIBERO 데이터를 RLDS나 LeRobot으로 바꾸지 않는다. 원본
demonstration의 simulator state와 7D action을 LIBERO 환경에서 다시 실행하고,
현재 CLaD loader가 그대로 읽는 task별 HDF5를 새 디렉터리에 생성한다.

이 선택은 CLaD 논문이 명시한 구현사항은 아니다. 논문의 “standard training
protocol [22]”에서 [22]가 OpenVLA인 점과 OpenVLA 비교 조건을 더 가깝게
맞추기 위한 재현 실험이다. 따라서 기존 128px 결과와 별도의 experiment로
관리해야 한다.

## 참고한 OpenVLA 절차

공식 OpenVLA의
[`regenerate_libero_dataset.py`](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py)는
다음 작업을 수행한다.

1. 저장된 초기 simulator state와 action을 다시 실행한다.
2. 외부 카메라와 wrist 카메라를 native 256×256으로 렌더링한다.
3. 로봇 motion이 없고 gripper 명령도 변하지 않은 no-op을 제거한다.
4. replay 후 task 성공 조건을 만족한 demonstration만 남긴다.
5. 후속 RLDS 변환에서 이미지 방향을 180° 회전한다.
6. fixed state를 적용하더라도 object placement에 영향을 줄 수 있어 모든
   task environment를 seed 0으로 초기화한다.

CLaD 구현은 위 동작을 독립적으로 재작성했으며 OpenVLA source를 vendoring하거나
복사하지 않았다. OpenVLA는 MIT, 이 저장소의 새 코드는 Apache-2.0이다. 포맷을
바꾸지 않기 때문에 5번의 방향 보정을 HDF5 저장 시점에 적용한다. 필요하면
`--image-transform none` 또는 `flip_vertical`로 명시적으로 바꿀 수 있다.

## 출력 계약

`scripts/rerender_libero_dataset.py`는 원본 파일을 절대 수정하지 않는다. 각
task는 임시 파일에 완전히 기록된 뒤 원자적으로 설치되며, 기존 출력이 현재
source/config fingerprint와 다르면 `--overwrite` 없이는 중단한다.

각 retained timestep에는 다음 값이 저장된다.

- `actions`, `states`, `rewards`, `dones`, `robot_states`
- `obs/agentview_rgb`, `obs/eye_in_hand_rgb`
- `obs/joint_states`, `obs/joint_velocities`, `obs/gripper_states`
- `obs/ee_pos`, `obs/ee_ori`(3D axis-angle), `obs/ee_states`
- 원본과의 대응을 위한 `source_action_indices`

현재 CLaD의 기본 proprioception은 joint position 7D + gripper position 2D이다.

생성된 dataset의 online 평가도 같은 초기화 protocol을 사용해야 한다. 현재
evaluator 기본값은 metadata와 같은 `environment_seed=0`, settle 10 steps,
그리고 Panda gripper를 open 상태로 유지하는 dummy action
`[0, 0, 0, 0, 0, 0, -1]`이다. simulator seed는 episode별 DDPM seed와 별도로
기록된다.
새 파일은 논문 표현과 후속 ablation을 위해 joint velocity도 보존하지만, 현재
모델 입력 차원을 자동으로 바꾸지는 않는다.

HDF5 `data` attributes와 `rerender_manifest.json`에는 source SHA-256, output
해상도, 이미지 변환, no-op threshold, settle step, 성공 선별 여부 및 episode
통계를 기록한다. `env_args`의 카메라 해상도도 출력 해상도로 갱신한다.

## 1. 소규모 smoke test

원본과 분리된 임시 또는 smoke 디렉터리를 사용한다. 아래 예시는 physical GPU
1만 process에 노출하므로 LIBERO 내부에서는 보이는 GPU가 0번이다.

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \
python scripts/rerender_libero_dataset.py \
  --source-dir /path/to/libero_datasets/libero_10 \
  --output-dir /path/to/libero_datasets/libero_10_openvla256_smoke \
  --task-ids 0 \
  --max-demos-per-task 5 \
  --resolution 256 \
  --environment-seed 0 \
  --log-interval 1
```

일부 원본 episode가 replay에서 실패하는 것은 오류가 아니다. 실제 local smoke
test에서 task 0의 처음 5개 중 2개가 성공하여 591 timestep이 기록되었다.
첫 episode처럼 선택 범위의 모든 replay가 실패하면 불완전한 HDF5를 설치하지
않고 명시적으로 종료한다.

결과를 기존 loader로 확인한다.

```bash
python scripts/inspect_dataset.py \
  --dataset-dir /path/to/libero_datasets/libero_10_openvla256_smoke \
  --horizon 6
```

출력에서 image shape가 `(256, 256, 3)`인지 확인한다. 두 번째 카메라도
검사하려면 `--camera-key obs/eye_in_hand_rgb`를 추가한다.

## 2. 전체 LIBERO-10 변환

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \
python scripts/rerender_libero_dataset.py \
  --source-dir /path/to/libero_datasets/libero_10 \
  --output-dir /path/to/libero_datasets/libero_10_openvla256 \
  --resolution 256 \
  --environment-seed 0
```

기본값은 모든 task에 environment seed 0, settle 10 steps, no-op threshold
`1e-4`, 성공 replay만 유지,
`rotate_180`, lossless LZF compression이다. 중단 후 같은 명령을 실행하면
fingerprint가 일치하는 완료 task는 건너뛴다. 설정이 달라진 출력은 새
디렉터리를 사용하는 것을 권장한다. `--overwrite`는 의도적으로 같은 task
파일을 교체할 때만 사용한다.

2026-08-05 이전 구현은 task별로 `environment.seed(task_id)`를 사용했다.
OpenVLA는 모든 task에 seed 0을 사용하므로, `rerender_manifest.json`의
`protocol`에 `environment_seed`가 없는 출력은 다시 생성해야 한다. 기존
디렉터리를 교체할 때는 같은 명령에 `--environment-seed 0 --overwrite`를
명시한다. 각 기존 HDF5는 대응 task의 새 파일이 완성될 때까지 유지된다.

256px 두 뷰는 원본 128px 한/두 뷰보다 디스크 사용량이 크게 증가한다. 전체
실행 전 smoke 결과의 파일 크기와 남은 공간을 확인한다.

## 3. DecisionNCE cache 재생성

변환된 픽셀과 retained timestep이 달라지므로 기존 128px cache를 재사용할 수
없다. 한 뷰 실험은 다음처럼 새 cache를 만든다.

```bash
python scripts/cache_decisionnce_features.py \
  --dataset-dir /path/to/libero_datasets/libero_10_openvla256 \
  --cache-dir .cache/decisionnce/libero_long_openvla256 \
  --model-name DecisionNCE-T \
  --device cuda
```

두 뷰 실험에서는 camera key를 둘 다 전달한다.

```bash
python scripts/cache_decisionnce_features.py \
  --dataset-dir /path/to/libero_datasets/libero_10_openvla256 \
  --cache-dir .cache/decisionnce/libero_long_openvla256_two_view \
  --model-name DecisionNCE-T \
  --device cuda \
  --camera-key obs/agentview_rgb \
  --camera-key obs/eye_in_hand_rgb
```

DecisionNCE RN50은 입력을 최종적으로 224px에 맞추지만, simulator에서 128px로
먼저 렌더링한 뒤 확대하는 것과 native 256px를 224px로 축소하는 것은 픽셀
정보량이 다르다.

cache manifest는 HDF5의 해상도와 이미지 방향 메타데이터를 보존한다. 온라인
encoder는 rollout의 raw image에 같은 변환을 자동 적용한다. 서로 다른 변환
메타데이터의 task를 한 cache에 섞으면 오류가 발생한다.

## 4. 학습과 평가

새 dataset/cache 경로로 Stage 1과 Stage 2 또는 Policy-only를 처음부터 다시
학습해야 한다. 기존 checkpoint는 128px 데이터에서 만든 feature와 episode
분포를 학습했기 때문에 공정한 256px 실험 checkpoint가 아니다.

평가 simulator도 256×256으로 렌더링해야 한다.

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \
python scripts/evaluate_clad_libero.py \
  --checkpoint /path/to/new/stage2_latest.pt \
  --foresight-checkpoint /path/to/new/stage1_foresight.pt \
  --cache-dir .cache/decisionnce/libero_long_openvla256 \
  --output-dir outputs/clad_evaluation_openvla256 \
  --device cuda:0 \
  --camera-height 256 \
  --camera-width 256
```

평가 해상도와 cache에 기록된 학습 해상도가 다르면 evaluator가 실행 전에
중단한다. legacy 128px cache에는 새 메타데이터가 없으므로 기존 평가 동작은
`none` 변환으로 그대로 유지된다.

## 주요 CLI 기본값

| 인자 | 기본값 |
| --- | --- |
| `--suite-name` | `libero_10` |
| `--resolution` | `256` |
| `--settle-steps` | `10` |
| `--environment-seed` | `0` (모든 task에 동일 적용) |
| `--noop-threshold` | `1e-4` |
| no-op | 제거 (`--keep-noops`로 보존) |
| replay 실패 | 제거 (`--keep-failed-replays`로 보존) |
| `--image-transform` | `rotate_180` |
| `--compression` | `lzf` |
| `--render-gpu-device-id` | `-1` |
| `--log-interval` | `25` demonstrations |
