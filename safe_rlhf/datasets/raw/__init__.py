# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
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
# ==============================================================================
"""Raw datasets."""

from safe_rlhf.datasets.raw.alpaca import AlpacaDataset,AlpacaZhDataset
from safe_rlhf.datasets.raw.firefly import FireflyDataset
from safe_rlhf.datasets.raw.hh_rlhf import (
    HhRLHFDialogueDataset,
    HhRLHFHarmlessDialogueDataset,
    HhRLHFHelpfulDialogueDataset,
    rlhf_rewardDataSet
)
from safe_rlhf.datasets.raw.moss import MOSS002SFT, MOSS003SFT, MOSS002ENHarmlessnessPrompt,MOSS002ENHonestyPrompt,MOSS002ENHelpfulnessPrompt,MOSS002ZHHelpfulnessPrompt,MOSS002ZHHonestynessPrompt
from safe_rlhf.datasets.raw.safe_rlhf import (
    SafeRLHF10KTrainDataset,
    SafeRLHFDataset,
    SafeRLHFTestDataset,
    SafeRLHFTrainDataset,
)
from safe_rlhf.datasets.raw.custom import CustomTrainDataset, CustomTestDataset,CValuesTestDataset, CValuesTrainDataset
from safe_rlhf.datasets.raw.gpt4_llm import gpt4_llm_reward
from safe_rlhf.datasets.raw.pku_safe import pku_safe_reward
from safe_rlhf.datasets.raw.share_gpt import ShareGpt,ShareGptDialog
from safe_rlhf.datasets.raw.callAnnie import CallAnnieTest, CallAnnieTrain
from safe_rlhf.datasets.raw.asr import CustomAsrTrainDataset, CustomAsrEvalDataset
from safe_rlhf.datasets.raw.asr_ada import AsrAdaTrainDataset, AsrAdaEvalDataset
__all__ = [
    'AlpacaDataset',
    'FireflyDataset',
    'HhRLHFDialogueDataset',
    'HhRLHFHarmlessDialogueDataset',
    'HhRLHFHelpfulDialogueDataset',
    'MOSS002SFT',
    'MOSS003SFT',
    'SafeRLHFDataset',
    'SafeRLHFTrainDataset',
    'SafeRLHFTestDataset',
    'SafeRLHF10KTrainDataset',
    'CustomTestDataset',
    'CustomTrainDataset',
    'AlpacaZhDataset',
    'gpt4_llm_reward',
    'pku_safe_reward',
    'ShareGptDialog',
    'CallAnnieTest',
    'CallAnnieTrain',
    'CValuesTestDataset',
    'CValuesTrainDataset',
    'CustomAsrTrainDataset',
    'CustomAsrEvalDataset', 
    'AsrAdaTrainDataset',
    'AsrAdaEvalDataset'
]
