# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math
from typing import Optional


import json
from pathlib import Path


import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from pxr import PhysxSchema


class MobileAloha(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "aloha",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
        scale: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]"""

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([0.8, 0.25, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation
        self._scale = torch.tensor([1.0, 1.0, 1.0]) if scale is None else scale

        if self._usd_path is None:
            project_root = Path(__file__).resolve().parents[2]
            
            config_path = project_root / "config.json"  # Relative path to config.json

            # Load USD path from config.json
            with open(config_path, 'r') as file:
                config = json.load(file)
                self._usd_path = str(project_root / config.get('mobile_aloha_usd_path')) 
            
        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            scale=self._scale,
            articulation_controller=None,
        )

        dof_paths = [
            "fl_base_link/fl_joint1",
            "fl_link1/fl_joint2",
            "fl_link2/fl_joint3",
            "fl_link3/fl_joint4",
            "fl_link4/fl_joint5",
            "fl_link5/fl_joint6",
            "fl_link6/fl_joint7",  # First prismatic joint for gripper
            "fl_link6/fl_joint8",  # Second prismatic joint for gripper
        ]


        drive_type = ["angular"] * 6 + ["linear"] * 2
        default_dof_pos = [0,0,0,0,0,0] + [0,0]
        stiffness = [400 * np.pi / 180] * 6 + [10000] * 2
        damping = [80 * np.pi / 180] * 6 + [100] * 2
        max_force = [87, 87, 87, 12, 12, 12, 100, 100]
        max_velocity = [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.175, 2.61, 2.61]] + [0.2, 0.2]

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i],
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(
                max_velocity[i]
            )

    def set_franka_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)