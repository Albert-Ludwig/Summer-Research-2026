# Original non-ROS smoke test status

Status:
- Test/inspect_checkpoint.py passed
- Test/smoke_import_policy.py passed
- Test/smoke_test_policy_load.py passed
- POLICY_CONFIGURE_OK reached

Checkpoint:
- SocialGuidedNavPlanner.pt
- checkpoint step: 478000
- style axes: prox, pass, yield, group

Environment:
- Python venv: .venv
- rvo2 installed from Python-RVO2 source into .venv
- acados cloned at commit dab96fc9b8ad486af8166331259834b33e93de37
- acados_template installed into .venv
- ACADOS_SOURCE_DIR=/home/ubuntu/acados
- t_renderer downloaded automatically
- acados projection solver built successfully

Notes:
- policy.config still points to ckpt_step478000_SOCIAL_NORMS8.pt
- smoke test used SocialGuidedNavPlanner.pt instead
- predict(state) has not been tested yet
- ROS wrapper has not been started yet
