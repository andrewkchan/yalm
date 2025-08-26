yalm (Yet Another Language Model) is an LLM inference implementation in C++/CUDA, using no libraries except to load and save frozen LLM weights.
- This project is intended as an **educational exercise** in performance engineering and LLM inference implementation. 
- The codebase therefore emphasizes documentation, whether external or in comments, scientific understanding of optimizations, and readability where possible. 
- It is not meant to be run in production. See [limitations](#limitations) section at bottom.
- See my blog post [Fast LLM Inference From Scratch](https://andrewkchan.dev/posts/yalm.html) for more.

Latest benchmarks with Mistral-7B-Instruct-v0.2 in FP16 with 4k context, on RTX 4090 + EPYC 7702P:

| Engine      | Avg. throughput (~120 tokens) tok/s | Avg. throughput (~4800 tokens) tok/s |
| ----------- | ----------- | ----------- |
| huggingface transformers, GPU | 25.9 | 25.7 |
| llama.cpp, GPU | 61.0 | 58.8 |
| calm, GPU | 66.0 | 65.7 |
| yalm, GPU | 63.8 | 58.7 |

# Instructions

yalm requires a computer with a C++20-compatible compiler and the CUDA toolkit (including `nvcc`) to be installed. You'll also need a directory containing LLM safetensor weights and configuration files in huggingface format, which you'll need to convert into a `.yalm` file. Follow the below to download Mistral-7B-v0.2, build `yalm`, and run it:

```
# install git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get -y install git-lfs
# download Mistral
git clone git@hf.co:mistralai/Mistral-7B-Instruct-v0.2
# clone this repository
git clone git@github.com:andrewkchan/yalm.git

cd yalm
pip install -r requirements.txt
python convert.py --dtype fp16 mistral-7b-instruct-fp16.yalm ../Mistral-7B-Instruct-v0.2/
make && ./build/main mistral-7b-instruct-fp16.yalm -i "What is a large language model?" -m c
```

# Usage

See the CLI help documentation below for `./build/main`:

```
Usage:   main <checkpoint> [options]
Example: main model.yalm -i "Q: What is the meaning of life?" -m c
Options:
  -h Display this help message
  -d [cpu,cuda] which device to use (default - cuda)
  -m [completion,passkey,perplexity] which mode to run in (default - completion)
  -T <int> sliding window context length (0 - max)

Perplexity mode options:
  Choose one:
    -i <string> input prompt
    -f <filepath> input file with prompt
Completion mode options:
  -n <int>    number of steps to run for in completion mode, default 256. 0 = max_seq_len, -1 = infinite
  -t <float> temperature (default - 1.0)
  Choose one:
    -i <string> input prompt
    -f <filepath> input file with prompt
Passkey mode options:
  -n <int>    number of junk lines to insert (default - 250)
  -l <int>    passkey position (-1 - random)
```

# Tests and benchmarks

yalm comes with a basic test suite that checks implementations of attention, matrix multiplications, feedforward nets in the CPU and GPU backends. Build and run it like so:

```
make test
./build/test
```

The test binary also includes benchmarks for individual kernels (useful for profiling with `ncu`) and broader system tools such as 2 benchmarks to determine main memory bandwidth:

```
# Memory benchmarks
./build/test -b
./build/test -b2

# Kernel benchmarks
./build/test -k [matmul,mha,ffn]
```

# Limitations

- Only completions may be performed (in addition to some testing modes like computing perplexity on a prompt or performing a [passkey test](https://github.com/ggerganov/llama.cpp/pull/3856)). Chat interface has not been implemented.
- An NVIDIA GPU is required.
- The GPU backend only works with a single GPU and the entire model must fit into VRAM.
- As of Dec 31, 2024 only the following models have been tested:
  - Mistral-v0.2 
  - Mixtral-v0.1 (CPU only)
  - Llama-3.2

# Acknowledgements

- [calm](https://github.com/zeux/calm) - Much of my implementation is inspired by Arseny Kapoulkine’s inference engine. In a way, this project was kicked off by “understand calm and what makes it so fast.” I’ve tried to keep my code more readable for myself though, and as much as possible scientifically understanding optimizations, which means foregoing some advanced techniques used in calm like dynamic parallelism.
- [llama2.c](https://github.com/karpathy/llama2.c) - Parts of the CPU backend come from Andrej Karpathy’s excellent C implementation of Llama inference.