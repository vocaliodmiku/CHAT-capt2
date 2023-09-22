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
"""Helpful and Harmless Dialogue Datasets from Anthropic."""

from __future__ import annotations

import os
from typing import ClassVar

from datasets import load_dataset
from safe_rlhf.datasets.base import RawDataset, RawSample


__all__ = [
    'HhRLHFDialogueDataset',
    'HhRLHFHarmlessDialogueDataset',
    'HhRLHFHelpfulDialogueDataset',
    'HhRLHFPreferenceDataset',
    'HhRLHFHarmlessPreferenceTrainDataset',
    'HhRLHFHarmlessPreferenceTestDataset',
    'HhRLHFHelpfulPreferenceTrainDataset',
    'HhRLHFHelpfulPreferenceTestDataset',
]


class HhRLHFDialogueDataset(RawDataset):
    NAME: ClassVar[str] = 'hh-rlhf-dialogue'
    ALIASES: tuple[str, ...] = ('hh-dialogue',)
    DATA_DIR: ClassVar[str | None] = None

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(
            path or 'PKU-Alignment/processed-hh-rlhf',
            data_dir=self.DATA_DIR,
            split='train',
        )

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        dialogue = [content['text'] for content in data['context']]
        dialogue.append(data['chosen']['text'])
        return RawSample(dialogue=dialogue)

    def __len__(self) -> int:
        return len(self.data)


class HhRLHFHarmlessDialogueDataset(HhRLHFDialogueDataset):
    NAME: str = 'hh-rlhf-harmless-dialogue'
    ALIASES: tuple[str, ...] = (
        'hh-rlhf-dialogue/harmless-base',
        'hh-harmless-dialogue',
        'hh-dialogue/harmless-base',
    )
    DATA_DIR: str = 'harmless-base'


class HhRLHFHelpfulDialogueDataset(HhRLHFDialogueDataset):
    NAME: str = 'hh-rlhf-helpful-dialogue'
    ALIASES: tuple[str, ...] = (
        'hh-rlhf-dialogue/helpful-base',
        'hh-helpful-dialogue',
        'hh-dialogue/helpful-base',
    )
    DATA_DIR: str = 'helpful-base'


def filter(example):
    prompt = example['prompt']
    chosen = example['chosen']
    rejected = example['rejected']
    return chosen != rejected and len(prompt) < 2000

class rlhf_rewardDataSet(RawDataset):
    NAME: str = 'rlhf_reward'
    ALIASES: tuple[str, ...] = ('rlhf_reward_new',)
    PATH: str = 'yitingxie/rlhf-reward-datasets'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(self.PATH, split='train')
        self.data = self.data.filter(filter)
        # self.data = load_dataset(path or 'tatsu-lab/alpaca', split='train')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['prompt']
        chosen = data['chosen']
        rejected = data['rejected']
        prompt = prompt + "Assistant: "
        chosen = str(chosen).replace("Assistant: ", "")
        rejected = str(rejected).replace("Assistant: ", "")
        if chosen == rejected:
            print(f"rlhf 数据集{index}错误， chose 和 rejected 一样")
            if index == len(self.data) - 1:
                return self.__getitem__(0)
            else:
                return self.__getitem__(index+1)
        assert prompt is not None and chosen is not None and rejected is not None
        return RawSample(input=prompt, answer=chosen, other_answer=rejected, safer=True, better=True)


    def __len__(self) -> int:
        return len(self.data)

class HhRLHFPreferenceDataset(RawDataset):
    NAME: ClassVar[str] = 'hh-rlhf-preference'
    ALIASES: tuple[str, ...] = ('hh-preference',)
    DATA_DIR: ClassVar[str | None] = None
    SPLIT: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(
            path or 'PKU-Alignment/processed-hh-rlhf',
            data_dir=self.DATA_DIR,
            split=self.SPLIT,
        )

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        dialogue = [content['text'] for content in data['context']]
        answer = data['chosen']['text']
        other_answer = data['rejected']['text']

        return RawSample(
            input=dialogue,
            answer=answer,
            other_answer=other_answer,
            better=True,
        )

    def __len__(self) -> int:
        return len(self.data)


class HhRLHFHarmlessPreferenceTrainDataset(HhRLHFPreferenceDataset):
    NAME: str = 'hh-rlhf-harmless-preference/train'
    ALIASES: tuple[str, ...] = (
        'hh-rlhf-preference/harmless-base/train',
        'hh-harmless-preference/train',
        'hh-preference/harmless-base/train',
    )
    DATA_DIR: str = 'harmless-base'
    SPLIT: str = 'train'


class HhRLHFHarmlessPreferenceTestDataset(HhRLHFPreferenceDataset):
    NAME: str = 'hh-rlhf-harmless-preference/test'
    ALIASES: tuple[str, ...] = (
        'hh-rlhf-preference/harmless-base/test',
        'hh-harmless-preference/test',
        'hh-preference/harmless-base/test',
    )
    DATA_DIR: str = 'harmless-base'
    SPLIT: str = 'test'


class HhRLHFHelpfulPreferenceTrainDataset(HhRLHFPreferenceDataset):
    NAME: str = 'hh-rlhf-helpful-preference/train'
    ALIASES: tuple[str, ...] = (
        'hh-rlhf-preference/helpful-base/train',
        'hh-helpful-preference/train',
        'hh-preference/helpful-base/train',
    )
    DATA_DIR: str = 'helpful-base'
    SPLIT: str = 'train'


class HhRLHFHelpfulPreferenceTestDataset(HhRLHFPreferenceDataset):
    NAME: str = 'hh-rlhf-helpful-preference/test'
    ALIASES: tuple[str, ...] = (
        'hh-rlhf-preference/helpful-base/test',
        'hh-helpful-preference/test',
        'hh-preference/helpful-base/test',
    )
    DATA_DIR: str = 'helpful-base'
    SPLIT: str = 'test'

