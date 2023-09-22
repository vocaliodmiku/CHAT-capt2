from safe_rlhf.models import load_pretrained_models_for_speech
from transformers import AutoModelForCausalLM 

model_name_or_path = "/ssd9/exec/penglinkai/chatcapt/output"
max_length  = 1024
MODEL_TYPE = AutoModelForCausalLM
trust_remote_code = True
model, tokenizer = load_pretrained_models_for_speech(
    model_name_or_path,
    model_max_length=max_length,
    padding_side='right',
    auto_model_type=MODEL_TYPE,
    trust_remote_code=trust_remote_code,
    audio_token_num=0
    # using_llama2=self.args.using_llama2,
)
debug = 1