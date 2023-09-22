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

from datasets import load_dataset
from safe_rlhf.datasets.base import RawDataset, RawSample


__all__ = ['AlpacaDataset']


class ShareGpt(RawDataset):
    NAME: str = 'share_gpt'
    ALIASES: tuple[str, ...] = ('share_gpt_single',)
    PATH: str = "/ssd9/exec/shenchengen/data/models/llm_data/Alpaca-CoT/ShareGPT/sharegpt.json"

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('json', data_files=[self.PATH])['train']

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = (  # pylint: disable=redefined-builtin
            ' '.join((data['instruction'], data['input'])) if data['input'] else data['instruction']
        )
        answer = data['output']
        input = "Human: " + input + "\n\nAssistant: "
        # answer = "Assistant: " + answer
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)

class ShareGptDialog(RawDataset):
    NAME: str = 'share_gpt_dialogue'
    ALIASES: tuple[str, ...] = ('share_gpt_multiturn',)
    PATH: str = "/ssd9/exec/shenchengen/data/models/llm_data/Alpaca-CoT/ShareGPT/sharegpt_context.json"

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('json', data_files=[self.PATH])['train']

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = (  # pylint: disable=redefined-builtin
            ' '.join((data['instruction'], data['input'])) if data['input'] else data['instruction']
        )
        answer = data['output']
        input = input.replace("[HM]:", "Human:").replace("[AI]:", "\nAssistant:")+"\n\nAssistant: "
        # input = "Human: "+input+"\n\n"
        # answer = "Assistant: "+answer
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)
