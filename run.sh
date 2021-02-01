loss_type=${1:-hard-em}
task=${2:-race}


python run_multiple_choice.py --cfg config/common.yaml --loss_cfg config/${loss_type}.yaml --data_cfg config/${task}.yaml
