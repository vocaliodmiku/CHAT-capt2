import os
from datasets import load_dataset, concatenate_datasets

data_path='/ssd5/exec/liyj/seamless_communication/scripts/m4t/audio_to_units/valid.json'
max_seq_length=2048

data_cache_dir = str(os.path.dirname(data_path))
cache_path = os.path.join(data_cache_dir,os.path.basename(data_path).split('.')[0])
os.makedirs(cache_path, exist_ok=True)

raw_dataset = load_dataset("json", data_files=data_path, cache_dir=cache_path)

os.makedirs(cache_path+'_save', exist_ok=True)
raw_dataset.save_to_disk(cache_path+'_save')
