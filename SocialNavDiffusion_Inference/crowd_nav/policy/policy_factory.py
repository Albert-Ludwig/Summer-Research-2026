from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_sim.envs.policy.linear import Linear
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.social_force import SFM
from crowd_nav.policy.diffusion_v3 import DiffusionV3
from crowd_nav.policy.diffusion_v1 import DiffusionV1
from crowd_nav.policy.diffusion_v2 import DiffusionV2
from crowd_nav.policy.diffusion_wf import DiffusionWF #V1 but world frame, not ego frame
from crowd_nav.policy.diffusion_v2wf import DiffusionV2WF #V2 but world frame, not ego frame

from crowd_nav.policy.diffusion_ConditionalUNet1D import DiffusionConditionalUNet1D
from crowd_nav.policy.diffusion_CondUNetTokens import DiffusionConditionalUNet1DTokens
from crowd_nav.policy.diffusion_CondUNetCFG import DiffusionConditionalUNet1DCFG
from crowd_nav.policy.diffusion_CondUNetCFG_FEASIBLE import DiffusionConditionalUNet1DCFG_FEASIBLE

from crowd_nav.policy.single_agent_proxy import SingleagentProxy
from crowd_nav.policy.multiagent_proxy import MultiagentProxy

policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['sfm'] = SFM
policy_factory['diffusion_v3'] = DiffusionV3
policy_factory['diffusion_v1'] = DiffusionV1
policy_factory['diffusion_v2'] = DiffusionV2
policy_factory['diffusion_wf'] = DiffusionWF
policy_factory['diffusion_v2wf'] = DiffusionV2WF

policy_factory['diffusion_conditional_unet1d'] = DiffusionConditionalUNet1D
policy_factory['diffusion_conditional_unet1dtokens'] = DiffusionConditionalUNet1DTokens
policy_factory['diffusion_conditional_unet1dcfg'] = DiffusionConditionalUNet1DCFG
policy_factory['diffusion_conditional_unet1dcfg_feasible'] = DiffusionConditionalUNet1DCFG_FEASIBLE
policy_factory['single_agent_proxy'] = SingleagentProxy
policy_factory['multiagent_proxy'] = MultiagentProxy