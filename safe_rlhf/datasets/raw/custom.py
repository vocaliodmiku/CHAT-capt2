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
"""Stanford Alpaca dataset for supervised instruction fine-tuning."""

from __future__ import annotations

import os.path

from datasets import load_dataset
from safe_rlhf.datasets.base import RawDataset, RawSample


__all__ = ['CustomTrainDataset', 'CustomTestDataset']


def filter(example):
    prompt = example['prompt']
    chosen = example['pos_resp']
    rejected = example['neg_resp']

    return chosen != rejected and prompt is not None and chosen is not None and rejected is not None
class CustomTrainDataset(RawDataset):
    NAME: str = 'custom_train'
    ALIASES: tuple[str, ...] = ('my_custom_train',)
    PATH: str = '/ssd9/exec/shenchengen/data/models/llm_data/rl_0727'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset("json", data_files=[
            path or os.path.join(self.PATH, 'train.json')
        ])['train']

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['prompt']
        chosen = data['chosen']
        rejected = data['rejected']
        prompt = "Human: " + prompt + "\n\nAssistant: "
        # chosen = "Assistant: " + chosen
        # rejected = "Assistant: " + rejected
        return RawSample(input=prompt, answer=chosen, other_answer=rejected, safer=True, better=True)

    def __len__(self) -> int:
        return len(self.data)


class CustomTestDataset(RawDataset):
    NAME: str = 'custom_test'
    ALIASES: tuple[str, ...] = ('my_custom_test',)
    PATH: str = '/ssd9/exec/shenchengen/data/models/llm_data/rl_test'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset("json", data_files=[
            path or os.path.join(self.PATH, 'test.json')
        ])['train']
        # self.data = load_dataset(path or 'tatsu-lab/alpaca', split='train')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['prompt']
        chosen = data['chosen']
        rejected = data['rejected']
        prompt = "Human: " + prompt + "\n\nAssistant: "
        # chosen = "" + chosen
        # rejected = "Assistant: " + rejected
        return RawSample(input=prompt, answer=chosen, other_answer=rejected, safer=True, better=True)

    def __len__(self) -> int:
        return len(self.data)

    NAME: str = 'custom_test'
    ALIASES: tuple[str, ...] = ('my_custom_test',)
    PATH: str = '/ssd9/exec/shenchengen/data/models/llm_data/rl_test'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset("json", data_files=[
            path or os.path.join(self.PATH, 'test.json')
        ])['train']

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['prompt']
        chosen = data['chosen']
        rejected = data['rejected']
        prompt = "Human: " + prompt + "\n\nAssistant: "
        # chosen = "Assistant: " + chosen
        # rejected = "Assistant: " + rejected
        return RawSample(input=prompt, answer=chosen, other_answer=rejected, safer=True, better=True)

    def __len__(self) -> int:
        return len(self.data)

class CValuesTrainDataset(RawDataset):
    NAME: str = 'cvalue_train'
    ALIASES: tuple[str, ...] = ('my_cvalue_train',)
    PATH: str = '/ssd9/exec/shenchengen/data/models/llm_data/ali_CValues'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset("json", data_files=[
            path or os.path.join(self.PATH, 'train.jsonl')
        ])['train']
        self.data = self.data.filter(filter)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['prompt']
        chosen = data['pos_resp']
        rejected = data['neg_resp']
        prompt = "Human: " + prompt + "\n\nAssistant: "
        # chosen = "Assistant: " + chosen
        # rejected = "Assistant: " + rejected
        return RawSample(input=prompt, answer=chosen, other_answer=rejected, safer=True, better=True)

    def __len__(self) -> int:
        return len(self.data)


class CValuesTestDataset(RawDataset):
    NAME: str = 'cvalue_test'
    ALIASES: tuple[str, ...] = ('my_cvalue_test',)
    PATH: str = '/ssd9/exec/shenchengen/data/models/llm_data/ali_CValues'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset("json", data_files=[
            path or os.path.join(self.PATH, 'test.jsonl')
        ])['train']
        self.data = self.data.filter(filter)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['prompt']
        chosen = data['pos_resp']
        rejected = data['neg_resp']
        prompt = "Human: " + prompt + "\n\nAssistant: "
        # chosen = "Assistant: " + chosen
        # rejected = "Assistant: " + rejected
        return RawSample(input=prompt, answer=chosen, other_answer=rejected, safer=True, better=True)

    def __len__(self) -> int:
        return len(self.data)
