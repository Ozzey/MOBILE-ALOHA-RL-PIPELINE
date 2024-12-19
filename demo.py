import datetime
import os

import numpy as np
import gym
import hydra
import torch
from omegaconf import DictConfig
import omniisaacgymenvs
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path, get_experience
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from utils import initialize_task
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


def run_demo(env):
    enable_cameras = env.task.add_camera
    task = env.task
    
    for _ in range(3):
        obs = env.reset()
        
        if not task.camera_initialized:
            from omni.isaac.core.utils.extensions import enable_extension
            enable_extension("omni.isaac.sensor")
            task.set_up_camera()
        
        buf = []
        for i in range(10000):
            action = env.action_space.sample()
            # -_- TypeError: clamp() ...
            action = torch.as_tensor(action)

            action = torch.zeros_like(action)
            
            obs, r, done, info = env.step(action)
            
            if enable_cameras:
                image_1 = env.task.render("hand_cam_0")
                image_2 = env.task.render("base_cam_0")
                if image_1 is not None and image_2 is not None:
                    image = np.hstack((image_1, image_2))
                    buf.append(image)
                print(f"Adding image at step {i}")
            else:
                print(f"Got obs {obs['obs'][0, :5]}...")
            
            if enable_cameras and (i+1) % 100 == 0:
                import imageio as iio
                iio.mimwrite("test.mp4", buf)
                print("Saved video.")
                buf.clear()
    
    print("Demo is DONE.")


class RLGTrainer:
    def __init__(self, cfg, cfg_dict):
        self.cfg = cfg
        self.cfg_dict = cfg_dict

    def launch_rlg_hydra(self, env):
        # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
        # We use the helper function here to specify the environment config.
        self.cfg_dict["task"]["test"] = self.cfg.test

        # register the rl-games adapter to use inside the runner
        vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": lambda **kwargs: env})

        self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)

    def run(self, module_path, experiment_dir):
        self.rlg_config_dict["params"]["config"]["train_dir"] = os.path.join(module_path, "runs")

        # create runner and set the settings
        runner = Runner(RLGPUAlgoObserver())
        runner.load(self.rlg_config_dict)
        runner.reset()

        # dump config dict
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        runner.run(
            {"train": not self.cfg.test, "play": self.cfg.test, "checkpoint": self.cfg.checkpoint, "sigma": None}
        )


@hydra.main(version_base=None, config_name="config", config_path="cfg")
def parse_hydra_configs(cfg: DictConfig):

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    headless = cfg.headless

    # local rank (GPU id) in a current multi-gpu mode
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # global rank (GPU id) in multi-gpu multi-node mode
    global_rank = int(os.getenv("RANK", "0"))
    if cfg.multi_gpu:
        cfg.device_id = local_rank
        cfg.rl_device = f'cuda:{local_rank}'
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    # select kit app file
    experience = get_experience(headless, cfg.enable_livestream, enable_viewport, cfg.enable_recording, cfg.kit_app)

    env = VecEnvRLGames(
        headless=headless,
        sim_device=cfg.device_id,
        enable_livestream=cfg.enable_livestream,
        enable_viewport=enable_viewport or cfg.enable_recording,
        experience=experience
    )

    # parse experiment directory
    module_path = os.getcwd()
    print("Save_path:", module_path)
    experiment_dir = os.path.join(module_path, "runs", cfg.train.params.config.name)

    # use gym RecordVideo wrapper for viewport recording
    if cfg.enable_recording:
        if cfg.recording_dir == '':
            videos_dir = os.path.join(experiment_dir, "videos")
        else:
            videos_dir = cfg.recording_dir
        video_interval = lambda step: step % cfg.recording_interval == 0
        video_length = cfg.recording_length
        env.is_vector_env = True
        if env.metadata is None:
            env.metadata = {"render_modes": ["rgb_array"], "render_fps": cfg.recording_fps}
        else:
            env.metadata["render_modes"] = ["rgb_array"]
            env.metadata["render_fps"] = cfg.recording_fps
        env = gym.wrappers.RecordVideo(
            env, video_folder=videos_dir, step_trigger=video_interval, video_length=video_length
        )

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = cfg.seed + global_rank if cfg.seed != -1 else cfg.seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict["seed"] = cfg.seed

    task = initialize_task(cfg_dict, env)

    if cfg.wandb_activate and global_rank == 0:
        # Make sure to install WandB if you actually use this.
        import wandb

        run_name = f"{cfg.wandb_name}_{time_str}"

        wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            config=cfg_dict,
            sync_tensorboard=True,
            name=run_name,
            resume="allow",
        )

    torch.cuda.set_device(local_rank)
    
    run_demo(env)
    
    # rlg_trainer = RLGTrainer(cfg, cfg_dict)
    # rlg_trainer.launch_rlg_hydra(env)
    # rlg_trainer.run(module_path, experiment_dir)

    env.close()

    if cfg.wandb_activate and global_rank == 0:
        wandb.finish()


if __name__ == "__main__":
    parse_hydra_configs()
