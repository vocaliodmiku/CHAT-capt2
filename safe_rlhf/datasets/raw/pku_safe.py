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

import os.path

from datasets import load_dataset
from safe_rlhf.datasets.base import RawDataset, RawSample


def filter(example):
    prompt = example['prompt']
    if example['safer_response_id'] == 1:
        chosen = example['response_1']
        rejected = example['response_0']
    else:
        chosen = example['response_0']
        rejected = example['response_1']
    return chosen != rejected and len(prompt) <= 2000

class pku_safe_reward(RawDataset):
    NAME: str = 'pku_safe_reward'
    ALIASES: tuple[str, ...] = ('pku_safe_reward1',)
    PATH: str = '/ssd9/exec/shenchengen/data/models/llm_data/GPT-4-LLM/data/PKU-SafeRLHF-10K'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(self.PATH, split='train')
        # self.data = load_dataset(path or 'tatsu-lab/alpaca', split='train')
        self.data = self.data.filter(filter)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['prompt']
        prompt = "Human: " + prompt + "\n\nAssistant: "
        # prompt = data['user_input']
        # prompt = str(prompt).replace('Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n', '')
        # prompt = prompt.replace('\n\n### Input:\n', '')
        if data['safer_response_id'] == 1:
            chosen = data['response_1']
            rejected = data['response_0']
        else:
            chosen = data['response_0']
            rejected = data['response_1']
        # chosen = "Assistant: " + chosen
        # rejected = "Assistant: " + rejected
        assert prompt is not None and chosen is not None and rejected is not None
        return RawSample(input=prompt, answer=chosen, other_answer=rejected, safer=True, better=True)

    def __len__(self) -> int:
        return len(self.data)
