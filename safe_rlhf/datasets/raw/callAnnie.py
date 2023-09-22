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


class CallAnnieTrain(RawDataset):
    NAME: str = 'call_annie_train'
    ALIASES: tuple[str, ...] = ('call_annie_train_dialog',)
    PATH: str = '/ssd9/exec/shenchengen/data/models/llm_data/call_annie/v5_train.jsonl'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset("json", data_files=[
            path or self.PATH
        ])['train']

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['input']
        chosen = data['response']
        # rejected = data['rejected']
        # prompt = "Human: " + prompt + "\n\nAssistant: "
        # chosen = "Assistant: " + chosen
        # rejected = "Assistant: " + rejected
        return RawSample(input=prompt, answer=chosen)

    def __len__(self) -> int:
        return len(self.data)


class CallAnnieTest(RawDataset):
    NAME: str = 'call_annie_test'
    ALIASES: tuple[str, ...] = ('call_annie_test_dialog',)
    PATH: str = '/ssd9/exec/shenchengen/data/models/llm_data/call_annie/v5_dev.jsonl'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset("json", data_files=[
            path or self.PATH
        ])['train']

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['input']
        chosen = data['response']
        # rejected = data['rejected']
        # prompt = "Human: " + prompt + "\n\nAssistant: "
        # chosen = "Assistant: " + chosen
        # rejected = "Assistant: " + rejected
        return RawSample(input=prompt, answer=chosen)

    def __len__(self) -> int:
        return len(self.data)

