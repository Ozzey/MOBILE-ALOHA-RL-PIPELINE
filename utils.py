# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def import_tasks():
    from tasks.aloha.aloha_pick import AlohaPickTask
    from tasks.aloha.aloha_place import AlohaPlaceTask
    
    from tasks.aloha.aloha_pick_camera import AlohaPickCameraTask
    from tasks.aloha.aloha_place_camera import AlohaPlaceCameraTask

    from tasks.mobile_aloha.mobile_aloha_pick import MobileAlohaPickTask
    from tasks.mobile_aloha.mobile_aloha_place import MobileAlohaPlaceTask

    from tasks.mobile_aloha.mobile_aloha_pick_camera import MobileAlohaPickCameraTask
    from tasks.mobile_aloha.mobile_aloha_place_camera import MobileAlohaPlaceCameraTask
    

    # Mappings from strings to environments
    task_map = {
        "AlohaPick": AlohaPickTask,
        "AlohaPlace": AlohaPlaceTask,

        "AlohaPickCamera": AlohaPickCameraTask,
        "AlohaPlaceCamera": AlohaPlaceCameraTask,

        "MobileAlohaPick": MobileAlohaPickTask,
        "MobileAlohaPlace": MobileAlohaPlaceTask,

        "MobileAlohaPickCamera": MobileAlohaPickCameraTask,
        "MobileAlohaPlaceCamera": MobileAlohaPlaceCameraTask,
    }
    return task_map

def initialize_task(config, env, init_sim=True):
    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig

    sim_config = SimConfig(config)
    task_map = import_tasks()
    cfg = sim_config.config

    task = task_map[cfg["task_name"]](
        name=cfg["task_name"], sim_config=sim_config, env=env
    )

    backend = "torch"

    rendering_dt = sim_config.get_physics_params()["rendering_dt"]

    env.set_task(
        task=task,
        sim_params=sim_config.get_physics_params(),
        backend=backend,
        init_sim=init_sim,
        rendering_dt=rendering_dt,
    )

    return task
