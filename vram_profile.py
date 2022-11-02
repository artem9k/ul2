# use t5 small for profile
"""
t5 small
without opt:
527 mb 
with modules:
384 mb 
"""

from t5.test import Test
from t5.t5 import T5ForConditionalGeneration
from t5.tokenization_t5 import T5Tokenizer
from transformers import PretrainedConfig
import accelerate
import nvidia_smi

MODEL = 'google/t5-v1_1-small'

def load_tokenizer():
    tok = T5Tokenizer.from_pretrained(MODEL, use_fast=True)
    return tok

def profile_without_opt():
    # just load the whole ass model
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    used_before_load = info.used

    # create tokenizer
    tok = load_tokenizer()

    # create model from pretrained
    model = T5ForConditionalGeneration.from_pretrained(MODEL).to('cuda')
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    used_after_load = info.used

    del model
    del tok

    used_before_load *= 0.000001
    used_after_load *= 0.000001
    model_vram = used_after_load - used_before_load

    print(f'used before load: {used_before_load} MB')
    print(f'used after load: {used_after_load} MB')
    print(f'model vram with overhead: {model_vram} MB')

    nvidia_smi.nvmlShutdown()

def profile_with_modules():
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    used_before_load = info.used

    # create model from pretrained
    model = T5ForConditionalGeneration.from_pretrained(MODEL)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    model.encoder.to('cuda')
    #model.run_encoder()
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    used_after_encoder = info.used
    model.encoder.to('cpu')

    model.decoder.to('cuda')
    #model.run_decoder()
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    used_after_decoder = info.used
    model.decoder.to('cpu')

    model.lm_head.to('cuda')
    #model.run_head()
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    used_after_head = info.used
    model.lm_head.to('cpu')

    used_max = max(used_after_encoder, used_after_decoder, used_after_head)

    used_before_load *= 0.000001
    used_max *= 0.000001
    model_vram = used_max - used_before_load

    print(f'used before load: {used_before_load} MB')
    print(f'used max: {used_max} MB')
    print(f'max model vram with overhead: {model_vram} MB')

    nvidia_smi.nvmlShutdown()
    
def profile_with_split_attn():
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    used_before_load = info.used

    # create model from pretrained
    model = T5ForConditionalGeneration.from_pretrained(MODEL).to('cuda')
    model.enable_split_attention()
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    used_after_load = info.used
    del model

    used_before_load *= 0.000001
    used_after_load *= 0.000001
    model_vram = used_after_load - used_before_load

    print(f'used before load: {used_before_load} MB')
    print(f'used after load: {used_after_load} MB')
    print(f'model vram with overhead: {model_vram} MB')

    nvidia_smi.nvmlShutdown()


if __name__ == "__main__":
    #profile_without_opt()
    profile_with_modules()
    #profile_with_split_attn()
