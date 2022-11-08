from t5.test import Test
from t5.t5 import T5ForConditionalGeneration
from t5.tokenization_t5 import T5Tokenizer
import accelerate
import nvidia_smi
import transformers
from transformers import AutoModelForSeq2SeqLM
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import psutil

SWAP_PATH = './swap'

class Profiler:
    def __init__(self, swap_folder=None):
        self.vram = []
        self.ram = []
        self.swap = []
        self.do_swap = False
        
    def profile_start(self):
        # ram
        self.ram.append(psutil.virtual_memory()[3])

        # vram
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        self.info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        self.vram.append(self.info.used)

        if self.do_swap:
            # swap
            for path, dirs, files in os.walk(SWAP_PATH):
                for f in files:
                    fp = os.path.join(path, f)
                    size += os.path.getsize(fp)
            
            self.swap.append(size)
        
    def profile_end(self):
        # ram
        self.ram.append(psutil.virtual_memory()[3])

        # vram
        self.vram.append(self.info.used)
        nvidia_smi.nvmlShutdown()

        if self.do_swap:

            # swap
            for path, dirs, files in os.walk(SWAP_PATH):
                for f in files:
                    fp = os.path.join(path, f)
                    size += os.path.getsize(fp)
            
            self.swap.append(size)
        

    def show_results(self):

        self.vram = list(map(self.to_mb, self.vram))
        model_vram = self.vram[1] - self.vram[0]

        self.ram = list(map(self.to_mb, self.ram))
        model_ram = self.ram[1] - self.ram[0]

        mem_before = self.vram[0] + self.ram[0]
        mem_after = self.vram[1] + self.ram[1]
        mem_total = mem_after - mem_before

        if self.do_swap:
            self.swap = list(map(self.to_mb, self.swap))
            model_ram = self.swap[1] - self.swap[0]
            mem_before = self.vram[0] + self.ram[0] + self.swap[0]
            mem_after = self.vram[1] + self.ram[1] + self.swap[1]

        else:
            mem_before = self.vram[0] + self.ram[0]
            mem_after = self.vram[1] + self.ram[1]

        mem_total = mem_after - mem_before

        print('--------------------PROFILER--------------------')
        print(f'vram used before load: {self.vram[0]} MB')
        print(f'vram used after load: {self.vram[1]} MB')
        print(f'model vram: {model_vram} MB')
        print()

        print(f'vram used before load: {self.vram[0]} MB')
        print(f'vram used after load: {self.vram[1]} MB')
        print(f'model ram: {model_ram} MB')
        print()

        if self.do_swap:
            print(f'swap used before load: {self.vram[0]} MB')
            print(f'swap used after load: {self.vram[1]} MB')
            print(f'model swap: {model_vram} MB')
            print()
        
        # total
        print(f'total mem before load: {mem_before} MB')
        print(f'total mem after load: {mem_after} MB')
        print(f'total mem usage: {mem_total} MB')
        print()
 
        print('------------------------------------------------')
    
    @staticmethod
    def to_mb(num):
        return num * 0.000001

MODEL = 'google/t5-v1_1-small'

profiler = Profiler()
profiler.profile_start()

CKPT = 'google/t5-v1_1-small'

# create model from pretrained
with init_empty_weights():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, torch_dtype=torch.float16)
#model = load_checkpoint_and_dispatch(model, CKPT, device_map="auto", offload_folder='./offload')
profiler.profile_end()

del model

profiler.show_results()
