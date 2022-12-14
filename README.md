### T5 with Memory Optimizations
This model was originally taken from Huggingface. 

Broke: run LLMs on expensive server GPUs

Woke: run LLMs on consumer hardware

The eventual goal is to run the google UL2 model on a RTX3060 12GB. This should be [possible](https://github.com/basujindal/stable-diffusion/pull/103).
Maybe inference can even be kinda fast

Reference model: ```t5-v1_1-large```
info in ```vram_profile.py```
but ul2 is just a larger t5 with SiLU

TPUs are for losers 

### Metrics
I also want to use this repo to analyze the VRAM/performance tradeoff for language models up to 20B. 
Performance meaning both inference speed and benchmark performance
I will use either BIG-bench or [this](https://github.com/EleutherAI/lm-evaluation-harness)

###Optimizations Todo
- [x] Model works
- [x] Seperately load the model parts
    Model has 3 parts: encoder, decoder, lm_head (linear layer). can be loaded separately enc -> dec -> lm head
- [ ] xformers attn instead of regular attn
- [ ] split-attention  (Model Parallel Split Attention from [OPT/Megatron](https://arxiv.org/pdf/1909.08053.pdf)
- [ ] many modules support. This can be achieved with weights splitting, which also solves the RAM issue
    - [ ] automatic weight splitter for pt and bin
- [ ] Memory efficient [attention](https://github.com/basujindal/stable-diffusion/pull/103)

### Optimizations (Stolen from Automatic1111)
A number of optimization can be enabled by [commandline arguments](Run-with-Custom-Parameters):

### Running inference 
```
python3 setup.py develop
python3 run.py
```

| commandline argument           | explanation                                                                                                                                                                                                                                                                                                                                                                                                                          |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--xformers`                   | Use [xformers](https://github.com/facebookresearch/xformers) library. Great improvement to memory consumption and speed. Windows version installs binaries mainained by [C43H66N12O12S2](https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases). Will only be enabled on small subset of configuration because that's what we have binaries for. [Documentation](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers)                                                                         |
| `--force-enable-xformers`      | Enables xformers above regardless of whether the program thinks you can run it or not. Do not report bugs you get running this.                                                                                                                                                                                                                                                                                                      |
| `--opt-split-attention`        | Cross attention layer optimization significantly reducing memory use for almost no cost (some report improved preformance with it).  Black magic. <br/>On by default for `torch.cuda`, which includes both NVidia and AMD cards.                                                                                                                                                                                                     |
| `--disable-opt-split-attention` | Disables the optimization above.                                                                                                                                                                                                                                                                                                                                                                                                     |
| `--opt-split-attention-v1`     | Uses an older version of the optimization above that is not as memory hungry (it will use less VRAM, but will be more limiting in the maximum size of pictures you can make).                                                                                                                                                                                                                                                        |
| `--medvram`                    | Makes the Stable Diffusion model consume less VRAM by splitting it into three parts - cond (for transforming text into numerical representation), first_stage (for converting a picture into latent space and back), and unet (for actual denoising of latent space) and making it so that only one is in VRAM at all times, sending others to CPU RAM. Lowers performance, but only by a bit - except if live previews are enabled. |
| `--lowvram`                    | An even more thorough optimization of the above, splitting unet into many modules, and only one module is kept in VRAM. Devastating for performance.                                                                                                                                                                                                                                                                                 |
| `*do-not-batch-cond-uncond`    | Prevents batching of positive and negative prompts during sampling, which essentially lets you run at 0.5 batch size, saving a lot of memory. Decreases performance. Not a command line option, but an optimization implicitly enabled by using `--medvram` or `--lowvram`.                                                                                                                                                          |
| `--always-batch-cond-uncond`   | Disables the optimization above. Only makes sense together with `--medvram` or `--lowvram`                                                                                                                                                                                                                                                                                                                                           |
| `--opt-channelslast`           | Changes torch memory type for stable diffusion to channels last. Effects not closely studied.    
