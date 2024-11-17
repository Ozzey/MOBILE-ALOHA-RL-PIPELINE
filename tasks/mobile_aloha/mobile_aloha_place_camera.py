from gym import spaces
import numpy as np
import torch
import omni.usd
from pxr import UsdGeom

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from tasks.mobile_aloha.mobile_aloha_place import MobileAlohaPlaceTask

from robots.articulations.mobile_aloha import MobileAloha
from robots.articulations.views.mobile_aloha_view import MobileAlohaView

class MobileAlohaPlaceCameraTask(MobileAlohaPlaceTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)
        self._max_episode_length = 500
        self.distX_offset = 0.04
        self.dt = 1 / 60.0
        self._num_observations = self.camera_width * self.camera_height * 3
        self._num_actions = 8

        # use multi-dimensional observation for camera RGB
        self.observation_space = spaces.Box(
            np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * -np.Inf, 
            np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * np.Inf)

        RLTask.__init__(self, name, env)

        # Extra info for TensorBoard
        self.extras = {}
        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"cube_cube_dist": torch_zeros(), "finger_to_cube_dist": torch_zeros(), "is_stacked": torch_zeros(), "success_rate": torch_zeros()}

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]

        # Flag for testing
        self.is_test = False
        self.initial_test_value = None
        self.is_action_noise = False

        self.camera_type = self._task_cfg["env"].get("cameraType", 'rgb')
        self.camera_width = self._task_cfg["env"]["cameraWidth"]
        self.camera_height = self._task_cfg["env"]["cameraHeight"]
        
        self.camera_channels = 3
        self._export_images = self._task_cfg["env"]["exportImages"]

    def cleanup(self) -> None:
        # initialize remaining buffers
        RLTask.cleanup(self)

        # override observation buffer for camera data
        self.obs_buf = torch.zeros(
            (self.num_envs, self.camera_width, self.camera_height, 3), device=self.device, dtype=torch.float)

    def add_camera(self) -> None:
        stage = get_current_stage()
        camera_path = f"/World/envs/env_0/Camera"
        camera_xform = stage.DefinePrim(f'{camera_path}_Xform', 'Xform')
        # set up transforms for parent and camera prims
        position = (2.27, 0.0, 2.0)
        rotation = (0.0, -30.0, 0.0)
        UsdGeom.Xformable(camera_xform).AddTranslateOp()
        UsdGeom.Xformable(camera_xform).AddRotateXYZOp()
        camera_xform.GetAttribute('xformOp:translate').Set(position)
        camera_xform.GetAttribute('xformOp:rotateXYZ').Set(rotation)
        camera = stage.DefinePrim(f'{camera_path}_Xform/Camera', 'Camera')
        UsdGeom.Xformable(camera).AddRotateXYZOp()
        camera.GetAttribute("xformOp:rotateXYZ").Set((90, 0, 90))
        # set camera properties
        camera.GetAttribute('focalLength').Set(24)
        camera.GetAttribute('focusDistance').Set(400)
        # hide other environments in the background
        camera.GetAttribute("clippingRange").Set((0.01, 20.0))


    def set_up_scene(self, scene) -> None:
        # Franka
        self.get_franka()
        self.get_cube()
        self.get_target_cube()
        self.get_cabinet()

        self.get_kitchen()

        self.add_camera()

        RLTask.set_up_scene(self, scene, filter_collisions=False)

        # start replicator to capture image data
        self.rep.orchestrator._orchestrator._is_started = True

        #set up cameras
        self.render_products = []
        env_pos = self._env_pos.cpu()
        camera_paths = [f"/World/envs/env_{i}/Camera_Xform/Camera" for i in range(self._num_envs)]
        for i in range(self._num_envs):
            render_product = self.rep.create.render_product(camera_paths[i], resolution=(self.camera_width, self.camera_height))
            self.render_products.append(render_product)
                # set up cameras

        # initialize pytorch writer for vectorized collection
        self.pytorch_listener = self.PytorchListener()
        self.pytorch_writer = self.rep.WriterRegistry.get("PytorchWriter")
        self.pytorch_writer.initialize(listener=self.pytorch_listener, device="cuda")
        self.pytorch_writer.attach(self.render_products)

        # Add Franka
        self._frankas = MobileAlohaView(prim_paths_expr="/World/envs/.*/aloha", name="aloha_view")

        # Add cube
        self._cube = RigidPrimView(prim_paths_expr="/World/envs/.*/cube", name="cube_view", reset_xform_properties=False)

        # Add location_ball
        self._target_cube = RigidPrimView(prim_paths_expr="/World/envs/.*/target_cube", name="target_cube_view", reset_xform_properties=False)

        
        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._cube)
        scene.add(self._target_cube)

        self.init_data()
        return
    

    def get_observations(self) -> dict:
        # Franka
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.franka_dof_pos = franka_dof_pos

        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)

        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )
        # cube
        cube_pos, cube_rot = self._cube.get_world_poses(clone=False)

        # target cube
        tar_cube_pos, tar_cube_rot = self._target_cube.get_world_poses(clone=False)   # tool position
        to_target = cube_pos - tar_cube_pos  

        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                franka_dof_vel * self.dof_vel_scale,
                cube_pos,
                cube_rot,
                to_target,
            ),
            dim=-1,
        )

        observations = {self._frankas.name: {"obs_buf": self.obs_buf}}

        # retrieve RGB data from all render products
        images = self.pytorch_listener.get_rgb_data()
        if images is not None:
            if self._export_images:
                from torchvision.utils import save_image, make_grid
                img = images/255
                save_image(make_grid(img, nrows = 2), 'cartpole_export.png')

            self.obs_buf = torch.swapaxes(images, 1, 3).clone().float()/255.0
        else:
            print("Image tensor is NONE!")

        return self.obs_buf