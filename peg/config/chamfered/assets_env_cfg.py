import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from ... import mdp
from ...assembly_env_cfg import AssemblyEnvCfg

peg = ArticulationCfg(
    prim_path="/World/envs/env_.*/Peg",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.expanduser("~/Documents/USD/Peg_v1.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005, rest_offset=0.0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.2, 0.2, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)


@configclass
class ChamferedPegInsertEnvCfg(AssemblyEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.peg = peg.replace(prim_path="{ENV_REGEX_NS}/Peg")

        self.scene.peg_bottom_frame.target_frames = [
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Peg/Peg",
                name="peg_bottom",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.04],
                ),
            ),
        ]

        self.events.peg_reset = EventTerm(
            func=mdp.reset_peg_in_hand,
            mode="reset",
            params={
                "tf_pos": [0.0, 0.0, 0.005],
                "tf_quat": [0.7071, 0.0, 0.0, 0.7071],
            },
        )


@configclass
class ChamferedPegInsertEnvCfg_PLAY(ChamferedPegInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
