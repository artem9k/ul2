# use t5 small for profile
"""
t5 small
without opt:
527 mb 

"""

from t5.test import Test
from t5.t5 import T5ForConditionalGeneration
from transformers import PretrainedConfig
import nvidia_smi

MODEL = 'google/t5-v1_1-small'

def profile_without_opt():
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    used_before_load = info.used

    # create model from pretrained
    model = T5ForConditionalGeneration.from_pretrained(MODEL).to('cuda')
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
    profile_without_opt()
