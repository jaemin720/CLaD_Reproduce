"""Named proprioception contracts shared by training and LIBERO rollout."""

from __future__ import annotations

from dataclasses import dataclass

LIBERO_JOINT_GRIPPER = "libero_joint_gripper"
LEGACY_ROBOT_STATE = "robot_states"


@dataclass(frozen=True, slots=True)
class ProprioceptionSpec:
    """Offline HDF5 and live observation fields for one state vector."""

    name: str
    hdf5_keys: tuple[str, ...]
    hdf5_component_dims: tuple[int, ...]
    observation_keys: tuple[str, ...]
    observation_component_dims: tuple[int, ...]

    @property
    def dimension(self) -> int:
        return sum(self.hdf5_component_dims)


PROPRIOCEPTION_SPECS = {
    LIBERO_JOINT_GRIPPER: ProprioceptionSpec(
        name=LIBERO_JOINT_GRIPPER,
        hdf5_keys=("obs/joint_states", "obs/gripper_states"),
        hdf5_component_dims=(7, 2),
        observation_keys=("robot0_joint_pos", "robot0_gripper_qpos"),
        observation_component_dims=(7, 2),
    ),
    LEGACY_ROBOT_STATE: ProprioceptionSpec(
        name=LEGACY_ROBOT_STATE,
        hdf5_keys=("robot_states",),
        hdf5_component_dims=(9,),
        observation_keys=(
            "robot0_gripper_qpos",
            "robot0_eef_pos",
            "robot0_eef_quat",
        ),
        observation_component_dims=(2, 3, 4),
    ),
}


def proprioception_spec(name: str) -> ProprioceptionSpec:
    """Resolve a stable contract name or fail before training/evaluation."""

    try:
        return PROPRIOCEPTION_SPECS[name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported proprioception contract {name!r}; "
            f"available={sorted(PROPRIOCEPTION_SPECS)}"
        ) from error
