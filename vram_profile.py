# use t5 large for profile

from t5.test import Test
from t5.t5 import T5ForConditionalGeneration
from transformers import PretrainedConfig
import nvidia_smi

nvidia_smi.nvmlInit()

handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)


used_before_load = info.used

# create model from pretrained
model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-large').to('cuda')

info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

used_after_load = info.used

del model

print(f'used before load: {used_before_load}')
print(f'used after load: {used_after_load}')

nvidia_smi.nvmlShutdown()



