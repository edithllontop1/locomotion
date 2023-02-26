# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from legged_gym.envs.a1.a1_config_log import A1RoughCfg as A1RoughCfgLog
from legged_gym.envs.a1.a1_config_log import A1RoughCfgPPO as A1RoughCfgPPOLog

from legged_gym.envs.a1.a1_config_prior import A1RoughCfg as A1RoughCfgPrior
from legged_gym.envs.a1.a1_config_prior import A1RoughCfgPPO as A1RoughCfgPPOPrior

from legged_gym.envs.a1com.a1com_config import A1RoughCfg as A1RoughCfgCom
from legged_gym.envs.a1com.a1com_config import A1RoughCfgPPO as A1RoughCfgPPOCom


from legged_gym.envs.go1.go1_config import GO1RoughCfg, GO1RoughCfgPPO
from legged_gym.envs.go1.go1_config_log import GO1RoughCfg as GO1RoughCfgLog
from legged_gym.envs.go1.go1_config_log import GO1RoughCfgPPO as GO1RoughCfgPPOLog

from legged_gym.envs.a1wx.a1wx_config import A1WXRoughCfg, A1WXRoughCfgPPO
from legged_gym.envs.a1wxs.a1wxs_config import A1WXSRoughCfg, A1WXSRoughCfgPPO



from legged_gym.envs.go1k1.go1k1_config import GO1K1RoughCfg, GO1K1RoughCfgPPO
from legged_gym.envs.go1k1.go1k1_config_log import GO1K1RoughCfg as GO1K1RoughCfgLog
from legged_gym.envs.go1k1.go1k1_config_log import GO1K1RoughCfgPPO as GO1K1RoughCfgPPOLog
from legged_gym.envs.k1.k1_config import K1RoughCfg, K1RoughCfgPPO
from legged_gym.envs.z1.z1_config import Z1RoughCfg, Z1RoughCfgPPO
from legged_gym.envs.wx.wx_config import WXRoughCfg, WXRoughCfgPPO
from legged_gym.envs.wxoa1.wxoa1_config import WXOA1RoughCfg, WXOA1RoughCfgPPO
from legged_gym.envs.wxoa1p.wxoa1p_config import WXOA1PRoughCfg, WXOA1PRoughCfgPPO

from legged_gym.envs.wxrot.wxrot_config import WXROTRoughCfg,WXROTRoughCfgPPO

from .base.legged_robot import LeggedRobot
from .base.legged_robot_prior import LeggedRobot as LeggedRobotPrior
from .base.legged_robot_com import LeggedRobot as LeggedRobotCom
from .base.legged_robot_arm import LeggedRobot as LeggedRobotArm
# from .base.legged_robot_arm import LeggedArm, ArmPrior, LeggedRobotStudent
# from .base.legged_robot_arm import LeggedArmPrior

from legged_gym.envs.wx.arm_reach import Arm

from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
# from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO


import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "a1log", LeggedRobot, A1RoughCfgLog(), A1RoughCfgPPOLog() )
task_registry.register( "a1prior", LeggedRobotPrior, A1RoughCfgPrior(), A1RoughCfgPPOPrior())
task_registry.register( "a1com", LeggedRobotCom, A1RoughCfgCom(), A1RoughCfgPPOCom())


task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "go1", LeggedRobot, GO1RoughCfg(), GO1RoughCfgPPO() )
task_registry.register( "go1log", LeggedRobot, GO1RoughCfgLog(), GO1RoughCfgPPOLog() )
task_registry.register( "go1k1", LeggedRobot, GO1K1RoughCfg(), GO1K1RoughCfgPPO() )
task_registry.register( "go1k1log", LeggedRobot, GO1K1RoughCfgLog(), GO1K1RoughCfgPPOLog() )
task_registry.register( "k1", LeggedRobot, K1RoughCfg(), K1RoughCfgPPO() )
task_registry.register( "z1", LeggedRobot, Z1RoughCfg(), Z1RoughCfgPPO() )
task_registry.register( "wx", Arm, WXRoughCfg(), WXRoughCfgPPO() )

task_registry.register( "a1wx", LeggedRobotArm, A1WXRoughCfg(), A1WXRoughCfgPPO() )
# task_registry.register( "a1wxs", LeggedRobotStudent, A1WXSRoughCfg(), A1WXSRoughCfgPPO() )

# task_registry.register( "wxoa1", LeggedArm, WXOA1RoughCfg(), WXOA1RoughCfgPPO() )
# task_registry.register( "wxoa1p", LeggedArmPrior, WXOA1PRoughCfg(), WXOA1PRoughCfgPPO() )
# task_registry.register( "wxrot", ArmPrior, WXROTRoughCfg(), WXROTRoughCfgPPO() )




