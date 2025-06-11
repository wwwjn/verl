# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging

import hydra
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torchtitan.config_manager import ConfigManager, JobConfig as TorchTitanEngineConfig
from torchtitan.experiments.trainer.torchtitan_engine import TorchTitianEngine
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from torch.utils.data import Dataset

from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_to_local
from verl.utils.tracking import Tracking
from transformers import AutoTokenizer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class TorchTitanSFTTrainer:
    def __init__(self, config, engine_config: TorchTitanEngineConfig, tokenizer: AutoTokenizer, train_dataset: Dataset, val_dataset: Dataset):
        # Trainer config
        self.config = config

        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")
        

        # only initalize training related configs as part 
        self.engine = TorchTitianEngine(engine_config)
        self.engine.init_model_and_optimizer()

        self.device_mesh = self.engine.world_mesh
        self._build_dataloader(train_dataset, val_dataset)


    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size
        if self.engine.parallel_dims.dp_enabled:
            dp_mesh = self.device_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0
        
        if self.device_mesh.get_rank() == 0:
            print(f"Using DP rank {dp_rank} and size {dp_degree} for dataloader distribution")

        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True, num_replicas=dp_degree, rank=dp_rank, drop_last=True)
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,  # NOTE(jianiw): This is global batch size
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False, num_replicas=dp_degree, rank=dp_rank, drop_last=True)
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu, # NOTE(jianiw): This is local batch size
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}")

    def _normalize_config_bsz(self):
        dp_mesh = self.device_mesh["dp"]
        dp_degree = dp_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_degree}")

        # NOTE: self.config.data.train_batch_size is global batch size. train_batch_size = dp_degree * micro_batch_size_per_gpu
        assert self.config.data.train_batch_size % dp_degree == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_degree}"

        self.config.data.train_batch_size //= dp_degree

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def training_step(self, batch: TensorDict):

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.engine.optimizer_zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            outputs_loss = self.engine.forward_backward_step(batch=micro_batch)
            loss = outputs_loss / n_micro_batches
            step_loss += loss.item()

        self.engine.optimizer_step()
        lr = self.engine.lr_scheduler_step()
        step_loss = torch.tensor(step_loss).cuda()
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        return {"train/loss": step_loss.detach().item(), "train/lr(1e-3)": lr * 1e3}


    def fit(self):
        # rank = self.device_mesh.get_rank()
        rank = torch.distributed.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        global_step = 0
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        
        def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            """Common cross-entropy loss function for Transformer models training."""
            return torch.nn.functional.cross_entropy(
                pred.flatten(0, 1).float(), labels.flatten(0, 1)
            )


        self.engine.set_loss_fn(cross_entropy_loss)
        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
            ):
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                # for early exit validation
                if global_step >= self.total_training_steps:
                    return


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    # Use official HF tokenzier from https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    local_tokenzier_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer: AutoTokenizer = hf_tokenizer(local_tokenzier_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset: Dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset: Dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    # Currently, pass CONFIG_PATH to it, and initialize torchtian engine
    # TODO: Merge the torchtitan config as a subconfig of the sft_trainer main config
    engine_config_manager = ConfigManager()
    engine_config = engine_config_manager.parse_args([f"--job.config_file={config.trainer.torchtitan_config_file}"])
    trainer = TorchTitanSFTTrainer(config=config, engine_config=engine_config, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset)
    trainer.fit()


def create_sft_dataset(data_paths, data_config, tokenizer) -> Dataset:
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


if __name__ == "__main__":
    main()
