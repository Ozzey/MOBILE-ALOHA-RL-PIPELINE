from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class AlohaView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "AlohaView",
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        self._hands = RigidPrimView(
            prim_paths_expr="/World/envs/.*/aloha/link6", name="hands_view", reset_xform_properties=False
        )
        self._lfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/aloha/link7", name="lfingers_view", reset_xform_properties=False
        )
        self._rfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/aloha/link8",name="rfingers_view", reset_xform_properties=False,
        )

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)
        self._gripper_indices = [self.get_dof_index("joint7"), self.get_dof_index("joint8")]

    @property
    def gripper_indices(self):
        return self._gripper_indices