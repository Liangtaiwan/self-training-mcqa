# Unsupervised Multiple Choices Question Answering

This is the original implementation of the following paper.

Chi-Liang Liu, Hung-yi Lee. [Unsupervised Deep Learning based Multiple Choices Question Answering: Start Learning from Basic Knowledge](https://arxiv.org/abs/2010.11003)



## Quick Reproduce


```shell
pip install -r requirement.txt
```
Warning: If you want to use fp16, please install apex from https://github.com/NVIDIA/apex. Default config run with fp16.

Then, you can do

```shell
./preprocess.sh         # convert mctest and race format to squad format
./run.sh $task $type    # train and eval
```
task: mctest or race
type: gt, highest-only, mml, hard-em

Note: the extracted QA model results are in data/${task}/prediction*.json
If you want to use your own prediction, you can overwrite those files.

## Usage

For manual running:
```shell
python run_multiple_choice.py --cfg config/common.yaml --loss_cfg
config/${loss_type}d.yaml --data_cfg config/${task}.yaml ...(something you want to overwrite the default config varaible)
```

## Citation
```
@misc{liu2020unsupervised,
      title={Unsupervised Deep Learning based Multiple Choices Question Answering: Start Learning from Basic Knowledge}, 
      author={Chi-Liang Liu and Hung-yi Lee},
      year={2020},
      eprint={2010.11003},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact

For any question, please contact [Chi-Liang Liu](https://liangtaiwan.github.io) or post a Github issue.

