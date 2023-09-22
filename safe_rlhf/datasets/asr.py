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
import copy
import os
from fractions import Fraction
from typing import Any, Callable, ClassVar, Collection, Dict, Iterable, Iterator, List
from typing_extensions import NotRequired  # Python 3.11+
from typing_extensions import TypedDict  # Python 3.10+
from weakref import WeakValueDictionary

import numpy as np
import torch
import transformers
from torch.utils.data import ConcatDataset, Dataset, Subset, default_collate
from tqdm import tqdm
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy


from safe_rlhf.configs import (
    IGNORE_INDEX,
    PROMPT_ASSISTANT,
    PROMPT_BEGIN,
    PROMPT_INPUT,
    PROMPT_USER,
)
from safe_rlhf.utils import is_main_process
from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import right_padding


__all__ = [
    'RawDataset',
    'RawSample',
    'TokenizedDataset',
    'CollatorBase',
    'parse_dataset',
]


def parse_dataset(string: str) -> tuple[str, dict[str, Any]]:
    """Parse dataset path and its proportion and optionally additional arguments from a string.

    Args:
        string (str): Dataset string in the format of ``dataset_name[:proportion[:dataset_path]]``.
    """
    if string.count(':') > 2:
        raise ValueError(
            f'Invalid dataset string `{string}`, '
            'should be in the format of `dataset_name[:proportion[:dataset_path]]`.',
        )
    name, colon, proportion = string.partition(':')
    if not colon:
        return name, {'proportion': 1.0}
    proportion, colon, path = proportion.partition(':')
    if '/' in proportion:
        left, right = proportion.split('/')
        left, right = left.strip(), right.strip()
        if right == '-1':
            proportion = Fraction(int(left), -1)
            if proportion == 0:
                proportion = 0.0
        else:
            proportion = float(left) / float(right)
    elif proportion != '':
        proportion = float(proportion)
    else:
        proportion = 1.0
    if not colon:
        return name, {'proportion': proportion}
    if not path:
        raise ValueError(f'Invalid dataset path `{path}`.')
    path = os.path.expanduser(path)
    return name, {'proportion': proportion, 'path': path}



class SpeechRawSample(TypedDict, total=False):
    """Speech Raw sample type.

    For SupervisedDataset, should provide (input, answer) or (dialogue).
    For PreferenceDataset, should provide (input, answer, other_answer, better).
    For SafetyPreferenceDataset, should provide (input, answer, other_answer, safer, is_safe, is_other_safe).
    For PromptOnlyDataset, should provide (input).
    """

    
    input: NotRequired[str]  # either `input` or `dialogue` should be provided
    
    output: NotRequired[str]
    
    instruction: NotRequired[str]


class SpeechRawDataset(Dataset[SpeechRawSample]):
    """Dataset that provides raw speech samples."""

    NAME: ClassVar[str]
    """Name of the dataset."""
    ALIASES: ClassVar[Collection[str]]
    """Name aliases of the dataset."""
    __REGISTRY: ClassVar[WeakValueDictionary[str, SpeechRawDataset]] = WeakValueDictionary()
    """Registry of all subclasses."""
    __ALIAS_NAME_MAPPING: ClassVar[dict[str, str]] = {}
    """Mapping from aliases to names."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        name = getattr(cls, 'NAME', None)
        if name is None:
            return

        if not isinstance(name, str):
            raise ValueError('`NAME` should be a string.')
        if not name:
            raise ValueError('`NAME` should not be empty.')
        if name in cls.__REGISTRY or name in cls.__ALIAS_NAME_MAPPING:
            raise ValueError(f'Duplicate dataset name `{name}`.')
        cls.__REGISTRY[name] = cls

        aliases = set(cls.__dict__.get('ALIASES', ()))  # do not inherit aliases from parent class
        for alias in aliases:
            if not isinstance(alias, str):
                raise ValueError('`ALIASES` should be a set of strings.')
            if not alias:
                raise ValueError('`ALIASES` should not contain empty strings.')
            if alias in cls.__REGISTRY:
                raise ValueError(f'Duplicate dataset alias `{alias}`.')
            if alias in cls.__ALIAS_NAME_MAPPING:
                print(alias)
                print(cls.__ALIAS_NAME_MAPPING)
                raise ValueError(f'Duplicate dataset alias `{alias}`.')
        cls.__ALIAS_NAME_MAPPING.update(dict.fromkeys(aliases, name))

    @staticmethod
    def load(name: str, /, *args: Any, **kwargs: Any) -> SpeechRawDataset:
        """Load a raw dataset by name."""
        normalized_name = SpeechRawDataset.__ALIAS_NAME_MAPPING.get(name, name)
        try:
            cls = SpeechRawDataset.__REGISTRY[normalized_name]
        except KeyError as ex:
            raise ValueError(
                f'Unknown dataset name `{name}`. '
                'You should implement a raw dataset with that name and import it in your code. '
                f'Available datasets are: {", ".join(sorted(SpeechRawDataset.__REGISTRY.keys()))}.',
            ) from ex
        return cls(*args, **kwargs)

    make = load  # alias

    @abc.abstractmethod
    def __getitem__(self, index: int) -> SpeechRawSample:
        """Get a data sample by index."""
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[SpeechRawSample]:
        """Iterate over the dataset."""
        return (self[index] for index in range(len(self)))

    def split_train_test(
        self,
        split_ratio: float = 0.2,
        shuffle: bool = True,
        seed: int = 42,
    ) -> tuple[Dataset[SpeechRawSample], Dataset[SpeechRawSample]]:
        """Split the dataset into train dataset and test dataset by split ratio."""
        num_samples = len(self)
        indices = list(range(num_samples))
        num_test_samples = round(num_samples * split_ratio)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        train_dataset = Subset(self, indices[:-num_test_samples])
        test_dataset = Subset(self, indices[-num_test_samples:])
        return train_dataset, test_dataset


class SpeechTokenizedDataset(Dataset[Dict[str, torch.Tensor]]):
    """Dataset that provides tokenized samples."""

    raw_datasets: list[SpeechRawDataset]
    """Raw datasets."""
    tokenizer: transformers.PreTrainedTokenizerBase
    """Tokenizer."""
    rawdata: list[SpeechRawSample]
    """Merged raw data samples."""
    data: list[dict[str, torch.Tensor]]
    """Tokenized data samples."""

    _SENTINEL: Any = object()

    def __init__(  # pylint: disable=too-many-branches
        self,
        dataset_names_and_attributes: dict[str, float | dict[str, Any]]
        | Iterable[tuple[str, float | dict[str, Any]]],
        tokenizer: transformers.PreTrainedTokenizerBase,
        lazy_tokenization: bool = True,
        seed: int = 42,
    ) -> None:
        if not isinstance(dataset_names_and_attributes, dict):
            dataset_names_and_attributes = tuple(dataset_names_and_attributes)
            dataset_names = [name for name, _ in dataset_names_and_attributes]
            if len(dataset_names) != len(set(dataset_names)):
                raise ValueError(
                    f'Dataset names should be unique, but got {dataset_names}.',
                )

        super().__init__()
        self.dataset_names_and_proportion: dict[str, float | Fraction] = {}
        self.raw_datasets = []
        for name, attributes in dict(dataset_names_and_attributes).items():
            if isinstance(attributes, float):
                kwargs = {'proportion': attributes}
            elif isinstance(attributes, dict):
                kwargs = dict(attributes)  # copy
            else:
                raise TypeError(
                    f'Dataset `{name}` attributes should be a float or a dict, '
                    f'got {type(attributes).__name__}.',
                )
            proportion = kwargs.pop('proportion', 1.0)
            if isinstance(proportion, Fraction):
                if not (proportion < 0 and proportion.denominator == 1):
                    raise ValueError(
                        f'Dataset `{name}` proportion should be a negative integer '
                        f'represents `num_samples / -1`, got {proportion}.',
                    )
            else:
                proportion = float(proportion)
                if proportion < 0.0:
                    raise ValueError(
                        f'Dataset `{name}` proportion should be no less than 0.0, '
                        f'got {proportion}.',
                    )
            if proportion == 0.0:
                continue
            self.dataset_names_and_proportion[name] = proportion
            self.raw_datasets.append(SpeechRawDataset.load(name, **kwargs))

        self.tokenizer = tokenizer
        self.seed = seed
        self.data = self.raw_datasets[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index.""" 
        raw_sample = self.data[index]
        data = self.preprocess(raw_sample) 
        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.data)

    def preprocess(self, raw_sample: SpeechRawSample) -> SpeechRawSample:
        PROMPT_TEMPLATE = (
            "[INST] <<SYS>>\n"
            "You are a helpful assistant. 你是一个乐于助人的助手。\n"
            "<</SYS>>\n\n{instruction} [/INST]"
        )
        prompt = PROMPT_TEMPLATE

        instruction = raw_sample['instruction']
        input = raw_sample['input']
        output = raw_sample['output']
        
        if input is not None and input !="":
            instruction = instruction+'\n'+input
        source = prompt.format_map({'instruction':instruction})
        target = f"{output}{self.tokenizer.eos_token}"
        
        tokenized_source = self.tokenizer(source,return_attention_mask=False)['input_ids']
        tokenized_target = self.tokenizer(target,return_attention_mask=False,add_special_tokens=False)['input_ids']
        max_length = self.tokenizer.model_max_length
        text_id = torch.LongTensor(tokenized_source + tokenized_target)[:max_length]
        
        labels = text_id.clone() 
        return {
           'input_ids': text_id,
           'labels': labels,
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator(self.tokenizer.pad_token_id)

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation: bool | str | TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
        max_length: int | None = None,
    ) -> torch.LongTensor:  # size = (L,)
        """Tokenize a text string into a tensor representation."""
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        #print(f"***max_length: {max_length}***")
        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors='pt',
        )['input_ids'][0]

    # def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
    #     """Get a collator function for the dataset."""
    #     return default_collate

    def resize(self, size: int) -> None:
        """Resize the dataset to the given size."""
        if size < 0:
            raise ValueError(f'Invalid dataset size: {size}')
        old_size = len(self)
        if size == old_size:
            return
        if size > old_size:
            num_replicates = (size + old_size - 1) // old_size
            self.data = self.data * num_replicates
            self.rawdata = self.rawdata * num_replicates
        self.data = self.data[:size]
        self.rawdata = self.rawdata[:size]

    def split_train_test(
        self,
        split_ratio: float = 0.2,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> tuple[Dataset[dict[str, torch.Tensor]], Dataset[dict[str, torch.Tensor]]]:
        """Split the dataset into train dataset and test dataset by split ratio."""
        if seed is None:
            seed = self.seed

        num_samples = len(self)
        indices = list(range(num_samples))
        num_test_samples = round(num_samples * split_ratio)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        memo = {id(attr): attr for attr in self.__dict__.values()}
        train_dataset = copy.deepcopy(self, memo=memo.copy())
        test_dataset = copy.deepcopy(self, memo=memo.copy())
        train_dataset.data = [self.data[i] for i in indices[:-num_test_samples]]
        train_dataset.rawdata = [self.rawdata[i] for i in indices[:-num_test_samples]]
        test_dataset.data = [self.data[i] for i in indices[-num_test_samples:]]
        test_dataset.rawdata = [self.rawdata[i] for i in indices[-num_test_samples:]]

        return train_dataset, test_dataset


class SpeechSupervisedBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)


class SupervisedCollator(CollatorBase):
    def __call__(self, samples: list[SpeechRawSample]) -> SpeechSupervisedBatch:
        input_ids = right_padding(
            [sample['input_ids'] for sample in samples],
            padding_value=self.pad_token_id,
        )
        labels = right_padding(
            [sample['labels'] for sample in samples],
            padding_value=IGNORE_INDEX,
        )
        attention_mask = input_ids.ne(self.pad_token_id)
        return {
            'input_ids': input_ids,  # size = (B, L)
            'labels': labels,  # size = (B, L)
            'attention_mask': attention_mask,  # size = (B, L)
        }

