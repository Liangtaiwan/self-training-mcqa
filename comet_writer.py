import time

from comet_ml import Experiment
from torch.utils.tensorboard import SummaryWriter


class CometWriter(Experiment):
    def __init__(
        self, api_key=None, project_name=None, workspace=None,
        log_code=True, log_graph=True, auto_param_logging=True,
        auto_metric_logging=True, parse_args=True, auto_output_logging="default",
        log_env_details=True, log_git_metadata=True, log_git_patch=True,
        disabled=False, log_env_gpu=True, log_env_host=True, 
        display_summary=None, log_env_cpu=True, display_summary_level=1,
        exp_name=None, sync_tensorboard=False,
    ):
        super(CometWriter, self).__init__(
            api_key, project_name, workspace,
            log_code, log_graph, auto_param_logging,
            auto_metric_logging, parse_args, auto_output_logging,
            log_env_details, log_git_metadata, log_git_patch,
            disabled, log_env_gpu, log_env_host, 
            display_summary, log_env_cpu, display_summary_level
        )
        self.set_name(exp_name)
        self.sync_tensorboard = sync_tensorboard
        if self.sync_tensorboard:
            self.tb_writer = SummaryWriter(
                f"./runs/{exp_name}_{time.strftime('%Y%m%d%H%M%S')}"
            )
        
    def add_scalar(self, key, value, step):
        if not isinstance(step, int):
            try:
                step = int(step)
            except:
                step = None 
        self.log_metric(key, value, step)
        if self.sync_tensorboard:
            self.tb_writer.add_scalar(key, value, step)


    def flush(self):
        if self.sync_tensorboard:
            self.tb_writer.flush()
        
