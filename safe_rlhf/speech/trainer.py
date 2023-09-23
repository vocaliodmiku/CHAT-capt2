# Copyright 2021 The ce Shen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from __future__ import annotations

import abc
import argparse
from typing import Any, ClassVar

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.ops.adam import FusedAdam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_scheduler
from transformers.deepspeed import HfDeepSpeedConfig

from safe_rlhf.configs import ADAM_BETAS
from safe_rlhf.datasets.raw.asr_ada import ASRADADataset
from safe_rlhf.models import load_pretrained_models_for_speech
from safe_rlhf.trainers.base import TrainerBase
from safe_rlhf.utils import get_optimizer_grouped_parameters, is_main_process, to_device
from transformers.modeling_outputs import CausalLMOutputWithPast
from fairseq2.models.sequence import SequenceBatch


class ASRTrainer(TrainerBase):
    """Trainer base class for ASR.

    Abstract methods:
        loss: Compute supervised training loss.
        train_step: Perform a single training step.
    """

    TRAINING_TYPE: ClassVar[str] = 'asr'
    DATASET_TYPE: ClassVar[type[ASRADADataset]]
    MODEL_TYPE = AutoModelForCausalLM

    model: deepspeed.DeepSpeedEngine
    ds_config: dict[str, Any]

    def __init__(self, args: argparse.Namespace, ds_config: dict[str, Any]) -> None:
        """Initialize trainer."""
        self.args = args
        self.ds_config = ds_config
        self.global_step = 0

        self.init_models()
        dist.barrier()
        self.init_datasets()
        dist.barrier()
        self.init_engines()
        dist.barrier()
        self.init_logger()

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_config is not None and self.ds_config['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.ds_config)

        self.model, self.tokenizer = load_pretrained_models_for_speech(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='right',
            auto_model_type=self.MODEL_TYPE,
            trust_remote_code=self.args.trust_remote_code,
            audio_token_num=0
            # using_llama2=self.args.using_llama2,
        )

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        if self.args.bf16:
            data_dtype = torch.bfloat16
        else:
            data_dtype = torch.float32
        train_dataset = ASRADADataset(
            self.args.train_datasets,
            tokenizer=self.tokenizer,
            dtype=data_dtype
        )

        if self.args.need_eval:
            if self.args.eval_datasets is None and self.args.eval_split_ratio is not None:
                raise ValueError('Unsupported.') 
            
            elif self.args.eval_datasets is not None and self.args.eval_split_ratio is None:
                eval_dataset = ASRADADataset(
                    self.args.eval_datasets,
                    tokenizer=self.tokenizer,
                    partition="dev"
                )
            else:
                raise ValueError('Either `eval_datasets` or `eval_split_ratio` should be provided.')

            self.eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=eval_dataset.get_collator(),
                sampler=DistributedSampler(eval_dataset, shuffle=True),
                batch_size=self.args.per_device_eval_batch_size,
            )
        else:
            self.eval_dataloader = None

        self.train_dataloader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.get_collator(),
            sampler=DistributedSampler(train_dataset, shuffle=True),
            batch_size=self.args.per_device_train_batch_size,
        )
        # speech adaptor  
        from seamless_communication.models.unit_extraction.wav2vec2_layer_output import load_wav2vec2_model, Wav2Vec2LayerOutputModel
        wav2vec2_model = load_wav2vec2_model(
            model_name_or_card="xlsr2_1b_v2", 
            device=self.args.device, 
            dtype=torch.float32)
        self.speech_encoder = Wav2Vec2LayerOutputModel(wav2vec2_model)
        for name, param in self.speech_encoder.named_parameters():
            param.requires_grad = False

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.args.num_update_steps_per_epoch = (
            len(self.train_dataloader) + self.args.gradient_accumulation_steps - 1
        ) // self.args.gradient_accumulation_steps
        self.args.total_training_steps = self.args.epochs * self.args.num_update_steps_per_epoch

        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            self.model,
            self.args.weight_decay,
        )

        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=ADAM_BETAS,
        )

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.total_training_steps,
        )
        print(f"self.ds_config: {self.ds_config}")
        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            args=self.args,
            config=self.ds_config,
            lr_scheduler=lr_scheduler,
            dist_init_required=True,
        )

        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def loss(
        self,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """Loss function for supervised finetuning."""
        outputs: CausalLMOutputWithPast = self.model(
           **kwargs
        )
        return {
            'loss': outputs.loss,
        }

    def train_step(
        self,
        **kwargs
    ) -> dict[str, Any]:
        """Performs a single training step.

        Args:
            input_ids (torch.LongTensor): input ids for causal inputs to complete with.
            labels (torch.LongTensor): labels for the full sequence.
            attention_mask (torch.BoolTensor): attention mask for the labels.

        Returns:
            dict[str, Any]: training loss, learning rate
        """
        loss = self.loss(**kwargs)['loss']
        self.model.backward(loss)
        self.model.step()

        # Synchronizes the final result.
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        return {
            'train/loss': loss,
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }

    def eval(self):
        """Eval the model."""
        self.logger.print('***** Running evaluating *****')

        self.model.eval()

        for batch in self.eval_dataloader:
            speech_feat = self.speech_encoder(SequenceBatch(
                seqs=batch['input'].to(self.args.device).to(torch.float32), 
                seq_lens=batch['lengths'].to(self.args.device).to(torch.float32)
                ), 34).transpose(0,1)
            if self.args.bf16:
                batch['input'] = speech_feat.clone().to(torch.bfloat16)
            else:
                batch['input'] = speech_feat.clone()

            info = self.model(**to_device(batch, self.args.device))
            preds = torch.argmax(info.logits, -1)
            pred = self.tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True) 
            self.logger.print(
                f'\n***** Evaluating *****\n target: If things are getting too much for you, and you feel you can\'t cope, ask for help. Your family or friends may be able to offer practical support or a listening ear. \n pred:{pred[0]}',
            )
        return {}

    def train(self) -> None:
        """Train the model."""
        self.logger.print('***** Running training *****')
        progress_bar = tqdm(
            total=self.args.epochs * len(self.train_dataloader),
            desc=f'Training 1/{self.args.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )
        
        for epoch in range(self.args.epochs):
            self.model.train()
            for batch in self.train_dataloader:
                speech_feat = self.speech_encoder(SequenceBatch(
                    seqs=batch['input'].to(self.args.device).to(torch.float32), 
                    seq_lens=batch['lengths'].to(self.args.device).to(torch.float32)
                    ), 34).transpose(0,1)
                if self.args.bf16:
                    batch['input'] = speech_feat.clone().to(torch.bfloat16)
                else:
                    batch['input'] = speech_feat.clone()
                info = self.train_step(**to_device(batch, self.args.device))
                # torch.cuda.empty_cache()
                info['train/epoch'] = epoch

                self.global_step += 1
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.args.epochs} epoch '
                    f'(loss {info["train/loss"]:.4f})',
                )
                progress_bar.update(1)

                # progress_bar.set_description(
                #     f'Training {epoch + 1}/{self.args.epochs} epoch (loss {info["train/loss"]:.4f})',
                # )
                # progress_bar.update()

                if self.global_step % self.args.save_interval == 0:
                    self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                    self.model.save_checkpoint(self.args.output_dir, tag=self.global_step)
                    self.logger.print('Checkpoint saved.')

                if (
                    self.args.need_eval
                    and self.args.eval_strategy == 'steps'
                    and self.global_step % self.args.eval_interval == 0
                ):
                    self.logger.print(f'\n***** Evaluating at step {self.global_step} *****')
                    self.logger.log(self.eval(), step=self.global_step)

            if self.args.need_eval and self.args.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.args.epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)

            self.model.tput_timer.update_epoch_count()

    def set_train(self, mode: bool = True) -> None:
        """Set training mode for model."""
        if mode:
            self.model.train()
            if self.args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
        else:
            self.model.eval()
            if self.args.gradient_checkpointing:
                self.model.gradient_checkpointing_disable()
