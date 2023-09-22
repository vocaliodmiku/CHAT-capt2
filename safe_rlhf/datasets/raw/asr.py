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
from typing import List

from datasets import load_dataset, Dataset
from safe_rlhf.datasets.asr import SpeechRawDataset, SpeechRawSample
import asrp


__all__ = ['CustomAsrTrainDataset', 'CustomAsrEvalDataset']




class CustomAsrTrainDataset(SpeechRawDataset):
    NAME: str = 'custom_asr_train'
    ALIASES: tuple[str, ...] = ('custom_sft_train',)
    PATH: str = '/ssd5/exec/liyj/seamless_communication/scripts/m4t/audio_to_units/train_save/train'

    def __init__(self, path: str | None = None) -> None:
        self.data = Dataset.load_from_disk(path or self.PATH)
        print(len(self.data))

    def __getitem__(self, index: int) -> SpeechRawSample:
        data = self.data[index]
        instruction = data['instruction']
        input = data['input']
        output = data['output']
        return SpeechRawSample(input=input,
                               output=output,
                               instruction=instruction)

    def __len__(self) -> int:
        return len(self.data)

class CustomAsrEvalDataset(SpeechRawDataset):
    NAME: str = 'custom_asr_eval'
    ALIASES: tuple[str, ...] = ('custom_sft_eval',)
    PATH: str = '/ssd5/exec/liyj/seamless_communication/scripts/m4t/audio_to_units/valid_save/train'

    def __init__(self, path: str | None = None) -> None:
        self.data = Dataset.load_from_disk(path or self.PATH)
        print(len(self.data))

    def __getitem__(self, index: int) -> SpeechRawSample:
        data = self.data[index]
        instruction = data['instruction']
        input = data['input']
        output = data['output']
        return SpeechRawSample(input=input,
                               output=output,
                               instruction=instruction)

    def __len__(self) -> int:
        return len(self.data)


