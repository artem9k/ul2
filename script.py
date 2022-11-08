from t5.test import Test
from t5.t5 import T5ForConditionalGeneration
from t5.tokenization_t5 import T5Tokenizer
from t5.profiler import Profiler
import accelerate
import nvidia_smi
import transformers
from transformers import AutoModelForSeq2SeqLM
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import psutil

SWAP_PATH = './swap'
MODEL = 'google/t5-v1_1-small'

profiler = Profiler(do_swap=False)
profiler.profile_start()

CKPT = 'google/t5-v1_1-small'

# create model from pretrained
with init_empty_weights():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, torch_dtype=torch.float16)
#model = load_checkpoint_and_dispatch(model, CKPT, device_map="auto", offload_folder='SWAP_PATH')
profiler.profile_end()

del model

profiler.show_results()
