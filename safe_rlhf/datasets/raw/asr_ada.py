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

__all__ = ['AsrAdaTrainDataset', 'AsrAdaEvalDataset']

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List
from .prompts import PROMPTS_ASR
import random
import soundfile as sf    

PROMPT_TEMPLATE = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. \n"
    "<</SYS>>\n\n{instruction} [/INST]"
)

# https://github.com/huggingface/blog/blob/main/llama2.md#how-to-prompt-llama-2
def read_audio(f):
    postfix = f.split('.')[-1]
    if not os.path.exists(f):
        return None
    if postfix == "flac":
        data, samplerate = sf.read(f)  
        assert samplerate == 16000 
        return data
    elif postfix == "mp3" or postfix == "wav":
        try:
            data, samplerate = sf.read(f)  
            assert samplerate == 16000 
            return data
        except:
            return None
    else:
        debug = 1
        return None

class ASRADADataset(Dataset):
    def __init__(self, data_path, tokenizer, partition="train", max_audio_length=30, dtype=torch.float32):
        if partition == "train":
            data_file = os.path.join(data_path[0][0], "train_manifest.json")
        else:
            # data_file = os.path.join(data_path[0][0], "train_manifest.json")
            data_file = os.path.join(data_path[0][0], "dev_manifest.json")
            #data_file = "dev_manifest.json"
        self.ann = {}
        self.utts = {}
        idx = 0
        with open(data_file, "r") as f:
            for line in f.readlines():
                line = json.loads(line)
                self.ann[line['target']['id']] = line   
                self.utts[idx] = line['target']['id']
                idx += 1
        self.max_audio_length = max_audio_length
        self.tokenizer = tokenizer
        self.dtype = dtype

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        index = self.utts[index]
        ann = self.ann[index]
        ASR_PRMPT = random.choice(PROMPTS_ASR)
        pre_msg = PROMPT_TEMPLATE
        pre_msg = pre_msg.format_map({'instruction':ASR_PRMPT})
        pre_msg += " <SEP> "
        post_msg = " <SEP> " + ann['target']['text']
        pre_tok, post_tok = self.tokenizer.encode(pre_msg), self.tokenizer.encode(post_msg)
        post_tok.append(self.tokenizer.eos_token_id)
        post_tok.pop(0)
        pre_tok, post_tok = torch.tensor(pre_tok, dtype=torch.int64), torch.tensor(post_tok, dtype=torch.int64)

        waveform_npy = read_audio(ann["target"]["audio_local_path"])
        waveform = torch.from_numpy(waveform_npy).to(torch.float32)[:self.max_audio_length*16000].to(self.dtype)
        # 30s 等价于1500个speech token
        return {
            "waveform": waveform,
            "pre_tok": pre_tok,
            "post_tok": post_tok,
        }

    @staticmethod
    def get_mask(seq_lens):
        mask = torch.BoolTensor(size=(len(seq_lens), max(seq_lens))) 
        for idx, length in enumerate( seq_lens ):
            mask[idx,:] = False
            mask[idx,:length] = True 
        return ~mask
    
    @staticmethod
    def collator(items):
        batch = {}
        batch["source"] = items
        waveforms, lengths = [], []
        pre_toks, post_toks = [], []
        pre_toks_lens, post_toks_lens = [], []
        for sample in items:
            waveform, length = sample["waveform"], sample["waveform"].shape[0]
            waveforms.append(waveform)
            lengths.append(length)
            pre_toks.append(sample["pre_tok"])
            post_toks.append(sample["post_tok"])
            pre_toks_lens.append(len(sample["pre_tok"]))
            post_toks_lens.append(len(sample["post_tok"]))
        batch["input"] = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        batch["input"] = F.layer_norm(batch["input"], batch["input"].shape)
        batch["lengths"] = torch.tensor(lengths)
        batch["seq_lens"] = (batch["lengths"] * 50 / 16000).int() + 2 # 写多两个，后面会在forward用真实的切片裁剪
        batch["src_key_padding_mask"] = ASRADADataset.get_mask(batch["seq_lens"])

        batch["pre_tok"] = torch.nn.utils.rnn.pad_sequence(pre_toks, batch_first=True)
        batch["pre_tok_padding_mask"] =  ASRADADataset.get_mask(pre_toks_lens)

        batch["post_tok"] = torch.nn.utils.rnn.pad_sequence(post_toks, batch_first=True)
        batch["post_tok_padding_mask"] =  ASRADADataset.get_mask(post_toks_lens)
        return batch

    def get_collator(self):
        return ASRADADataset.collator
    
class AsrAdaTrainDataset(ASRADADataset):
    NAME: str = 'asr_ada_train'  
    def __init__(self, dataset_config, tokenizer, max_audio_length):
        super.__init__(dataset_config, tokenizer, partition="train", max_audio_length=max_audio_length)


class AsrAdaEvalDataset(ASRADADataset):
    NAME: str = 'asr_ada_eval'  
    def __init__(self, dataset_config, tokenizer, max_audio_length):
        super.__init__(dataset_config, tokenizer, partition="eval", max_audio_length=max_audio_length)
