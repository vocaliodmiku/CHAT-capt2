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

import re
def filter(example):
    chosen = None
    rejected = None
    prompt = example['user_input']
    gpt4 = None
    icm = None
    gpt3 = None
    for _ in example['responses_and_scores']:
        if _['source'] == 'gpt4':
            gpt4 = _['response']
        if _['source'] == 'icm-1.3b':
            icm = _['response']
        if _['source'] == 'text-davinci-003':
            gpt3 = _['response']
    if gpt4 is None:
        return False

    chosen = gpt4
    if icm is not None:
        rejected = icm
    else:
        rejected = gpt3
    return chosen != rejected and len(prompt) <= 2000

class gpt4_llm_reward(RawDataset):
    NAME: str = 'gpt4_llm_reward'
    ALIASES: tuple[str, ...] = ('gpt4_llm_reward1',)
    PATH: str = '/ssd9/exec/shenchengen/data/models/llm_data/GPT-4-LLM/data/comparison_data_v2.json'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset("json", data_files=[
            self.PATH
        ])['train']
        self.data = self.data.filter(filter)
        # self.data = load_dataset(path or 'tatsu-lab/alpaca', split='train')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]

        prompt = data['user_input']

        # 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n'
        # prompt = re.sub(r'Below is an instruction.*### Instruction:\n', '', prompt)
        # s = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy."
        to_replace = "Below is an instruction that describes a task.*### Instruction:\n"

        # 使用 re.DOTALL 使 '.' 匹配包括换行在内的所有字符
        prompt = re.sub(to_replace, '', prompt, flags=re.DOTALL)

        # print(s)
        # prompt = str(prompt).replace('Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n', '')
        prompt = prompt.replace('\n\n### Input:', '')
        prompt = "Human: " + prompt + "\n\nAssistant: "
        chosen = None
        rejected = None
        gpt4 = None
        icm = None
        gpt3 = None
        for _ in data['responses_and_scores']:
            if _['source'] == 'gpt4':
                gpt4 = _['response']
            if _['source'] == 'icm-1.3b':
                icm = _['response']
            if _['source'] == 'text-davinci-003':
                gpt3 = _['response']
        if gpt4 is None:
            if index == len(self.data) - 1:
                return self.__getitem__(0)
            else:
                return self.__getitem__(index + 1)

        chosen = gpt4
        if icm is not None:
            rejected = icm
        else:
            rejected = gpt3
        # chosen = "Assistant: " + chosen
        # rejected = "Assistant: " + rejected
        if not (prompt is not None and chosen is not None and rejected is not None):
            if index == len(self.data) - 1:
                return self.__getitem__(0)
            else:
                return self.__getitem__(index+1)
        assert prompt is not None and chosen is not None and rejected is not None
        return RawSample(input=prompt, answer=chosen, other_answer=rejected, safer=True, better=True)

    def __len__(self) -> int:
        return len(self.data)
