class Profiler:
    def __init__(self, do_swap, swap_folder=None):
        self.do_swap = do_swap
        self.vram = []
        self.ram = []
        self.swap = []
        
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

