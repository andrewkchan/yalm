# Converts a model consisting of a huggingface config.json, tokenizer.json, and .safetensors weights into a .yalm file,
# which:
# - Normalizes the config to a common format in the header
# - Combines any safetensors shards
# - Reads the token vocabulary into a simpler format
# - Performs quantization to fp8 if specified

import argparse
import os
import json
import safetensors
from safetensors.torch import save_file
import torch

SUPPORTED_ARCHITECTURES = [
  "LlamaForCausalLM",
  "MistralForCausalLM",
  "MixtralForCausalLM"
]
SUPPORTED_DTYPES = ["fp32", "fp16", "fp8", "bf16"]

class Metadata:
  def __init__(self, config, dtype):
    arch = config["architectures"][0]
    if arch not in SUPPORTED_ARCHITECTURES:
      raise Exception(f"Architecture {arch} is not supported, must be one of {SUPPORTED_ARCHITECTURES}")
    self.arch = arch
    if dtype not in SUPPORTED_DTYPES:
      raise Exception(f"Data type {dtype} is not supported, must be one of {SUPPORTED_DTYPES}")
    self.dtype = dtype
    if arch in ["MistralForCausalLM", "LlamaForCausalLM", "MixtralForCausalLM"]:
      self.dim = config["hidden_size"]
      self.hidden_dim = config["intermediate_size"]
      self.head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
      self.n_layers = config["num_hidden_layers"]
      self.n_heads = config["num_attention_heads"]
      self.n_kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
      self.vocab_size = config["vocab_size"]
      self.max_seq_len = config["max_position_embeddings"]
      self.bos_token_id = config["bos_token_id"]
      self.eos_token_id = config["eos_token_id"]
      self.rope_theta = config.get("rope_theta", 10000.0)
      self.rotary_dim = int(self.head_dim * config.get("partial_rotary_factor", 1))
      self.norm_eps = config["rms_norm_eps"]
      self.norm_type = "rmsnorm"

      # TODO support bias
      assert config.get("attention_bias", False) == False
      assert config.get("mlp_bias", False) == False

      assert config["hidden_act"] in ["gelu", "silu"]
      self.act_type = config["hidden_act"]

      if arch in ["MixtralForCausalLM"]:
        self.n_experts = config["num_local_experts"]
        self.n_experts_active = config["num_experts_per_tok"]
  
  def to_dict(self):
    result = {}
    result["arch"] = self.arch
    result["dtype"] = self.dtype
    if self.arch in ["MistralForCausalLM", "LlamaForCausalLM", "MixtralForCausalLM"]:
      result["dim"] = str(self.dim)
      result["hidden_dim"] = str(self.hidden_dim)
      result["head_dim"] = str(self.head_dim)
      result["n_layers"] = str(self.n_layers)
      result["n_heads"] = str(self.n_heads)
      result["n_kv_heads"] = str(self.n_kv_heads)
      result["vocab_size"] = str(self.vocab_size)
      result["max_seq_len"] = str(self.max_seq_len)
      result["bos_token_id"] = str(self.bos_token_id)
      result["eos_token_id"] = str(self.eos_token_id)
      result["rope_theta"] = str(self.rope_theta)
      result["rotary_dim"] = str(self.rotary_dim)
      result["norm_eps"] = str(self.norm_eps)
      result["norm_type"] = str(self.norm_type)
      result["act_type"] = str(self.act_type)
      if self.arch in ["MixtralForCausalLM"]:
        result["n_experts"] = str(self.n_experts)
        result["n_experts_active"] = str(self.n_experts_active)
    return result

# this is a horrible gpt-2 unicode byte encoder hack from https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
# this has poisoned all HF tokenizer configs that use ByteLevel decoder/preprocessor
# as a result we get crazy UTF-8-as-bytes-as-UTF8 in the tokenizer data that we need to convert back
def gpt2_bytes_to_unicode():
  bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
  cs = bs[:]
  n = 0
  for b in range(2**8):
    if b not in bs:
      bs.append(b)
      cs.append(2**8+n)
      n += 1
  cs = [chr(n) for n in cs]
  return dict(zip(bs, cs))

def load_tokens(tokenizer_path, vocab_size):
  tokens = [""] * vocab_size
  with open(tokenizer_path, "r") as f:
    tokenizer = json.load(f)
  use_gpt2_byte_preprocessing = not tokenizer["model"].get("byte_fallback", False)
  
  vocab = tokenizer["model"]["vocab"]
  assert len(vocab) <= vocab_size

  for t, i in vocab.items():
    tokens[i] = t
  
  for added in tokenizer["added_tokens"]:
    tokens[added["id"]] = added["content"]
  
  gpt2_decode = {v: k for k, v in gpt2_bytes_to_unicode().items()}
  # Preprocess tokens into UTF-8 encoding
  for i, t in enumerate(tokens):
    if use_gpt2_byte_preprocessing:
      b = bytes([gpt2_decode.get(c, 0) for c in t])
    else:
      t = t.replace('\u2581', ' ') # sentencepiece uses this character as whitespace
      b = t.encode('utf-8')
    b = b.replace(b"\0", b"\7") # replace null bytes with bell characters
    assert b.count(0) == 0 # no null bytes allowed
    tokens[i] = b
  
  return tokens

def load_weights(model_files, dtype_str, metadata, tie_word_embeddings):
  """
  Load all weights from the model files in huggingface format into a dictionary of tensors,
  normalizing the attention weights, and casting all tensors (except for all layer norm weights,
  which are converted to float32) to the specified dtype.
  """
  weights = {}
  for model_path in model_files:
    ext = os.path.splitext(model_path)[1]
    if ext == ".safetensors":
      with safetensors.safe_open(model_path, framework="pt") as f:
        for k in f.keys():
          assert(k not in weights)
          weights[k] = f.get_tensor(k)
  
  # Stolen from https://github.com/zeux/calm/blob/86dfb56daba5052c260a2dd86d296309cfbd4324/tools/convert.py#L223
  # huggingface permutes WQ and WK, this function reverses it
  # see https://github.com/huggingface/transformers/blob/b132c1703eb1c8bd9dfa4ad6a9be2bfd6ef819e9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122
  def permute_reverse(w, heads, rotary_dim):
    head_dim = w.shape[0] // heads
    assert rotary_dim <= head_dim
    w = torch.unflatten(w, 0, (-1, head_dim))
    # wr is the rotary part, wk is the part kept unrotated
    wr = w[:, :rotary_dim]
    wk = w[:, rotary_dim:]
    # switch wr from outputting two rotary_dim/2 chunks to outputting values interleaved
    wr = torch.unflatten(wr, 1, (2, -1))
    wr = wr.transpose(1, 2)
    wr = wr.flatten(1, 2)
    # assemble the heads back
    w = torch.cat([wr, wk], dim=1)
    return torch.flatten(w, 0, 1)

  dtype = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "fp8": torch.float8_e5m2,
    "bf16": torch.bfloat16
  }[dtype_str]

  # convert weights
  progress = 0
  def conv(t):
    nonlocal progress
    progress += 1
    print(f"\rConverting tensor {progress}: {t.shape}", end="", flush=True)
    return t.to(dtype)

  tensors = {}
  tensors["model.embed.weight"] = conv(weights["model.embed_tokens.weight"])

  for l in range(config["num_hidden_layers"]):
    tensors[f"model.layers.{l}.attn.norm.weight"] = weights[f"model.layers.{l}.input_layernorm.weight"].float()

    rotary_dim = metadata.rotary_dim
    n_heads = metadata.n_heads
    n_kv_heads = metadata.n_kv_heads

    tensors[f"model.layers.{l}.attn.wq.weight"] = conv(permute_reverse(weights[f"model.layers.{l}.self_attn.q_proj.weight"], n_heads, rotary_dim))
    tensors[f"model.layers.{l}.attn.wk.weight"] = conv(permute_reverse(weights[f"model.layers.{l}.self_attn.k_proj.weight"], n_kv_heads, rotary_dim))

    tensors[f"model.layers.{l}.attn.wv.weight"] = conv(weights[f"model.layers.{l}.self_attn.v_proj.weight"])
    tensors[f"model.layers.{l}.attn.wo.weight"] = conv(weights[f"model.layers.{l}.self_attn.o_proj.weight"])

    tensors[f"model.layers.{l}.mlp.norm.weight"] = weights[f"model.layers.{l}.post_attention_layernorm.weight"].float()

    if metadata.arch in ["MixtralForCausalLM"]:
      tensors[f"model.layers.{l}.moegate.weight"] = conv(weights[f"model.layers.{l}.block_sparse_moe.gate.weight"])

      tensors[f"model.layers.{l}.mlp.w1.weight"] = torch.stack([conv(weights[f"model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight"]) for e in range(config["num_local_experts"])])
      tensors[f"model.layers.{l}.mlp.w2.weight"] = torch.stack([conv(weights[f"model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight"]) for e in range(config["num_local_experts"])])
      tensors[f"model.layers.{l}.mlp.w3.weight"] = torch.stack([conv(weights[f"model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight"]) for e in range(config["num_local_experts"])])
    else:
      tensors[f"model.layers.{l}.mlp.w1.weight"] = conv(weights[f"model.layers.{l}.mlp.gate_proj.weight"])
      tensors[f"model.layers.{l}.mlp.w2.weight"] = conv(weights[f"model.layers.{l}.mlp.down_proj.weight"])
      tensors[f"model.layers.{l}.mlp.w3.weight"] = conv(weights[f"model.layers.{l}.mlp.up_proj.weight"])

  tensors["model.norm.weight"] = weights["model.norm.weight"].float()
  if tie_word_embeddings == False:
    tensors["model.output.weight"] = conv(weights["lm_head.weight"])
  else:
    # Model output classifier just uses the word embeddings matrix
    pass
  
  print() # newline
  return tensors

if __name__ == "__main__":
  argp = argparse.ArgumentParser()
  argp.add_argument("output", type=str)
  argp.add_argument("input", type=str, nargs="?")
  argp.add_argument("--dtype", type=str, default="fp16", choices=SUPPORTED_DTYPES)
  args = argp.parse_args()

  if args.input is not None:
    # Input is a directory with HuggingFace layout, e.g. files:
    #   config.json
    #   tokenizer.json
    #   *.safetensors
    args.config = os.path.join(args.input, "config.json")
    if not os.path.exists(args.config):
      argp.error(f"config.json not found in {args.input}")
    
    args.tokenizer = os.path.join(args.input, "tokenizer.json")
    if not os.path.exists(args.tokenizer):
      argp.error(f"tokenizer.json not found in {args.input}")
    
    files = os.listdir(args.input)
    args.models = [os.path.join(args.input, fname) for fname in files if os.path.splitext(fname)[1] == ".safetensors"]
    if len(args.models) == 0:
      argp.error(f"no .safetensors files found in {args.input}")
  else:
    argp.error("argument input is required")

  with open(args.config, "r") as f:
    config = json.load(f)
    metadata = Metadata(config, args.dtype)

  tokens = load_tokens(args.tokenizer, metadata.vocab_size)
  tensors = load_weights(args.models, args.dtype, metadata, config.get("tie_word_embeddings", None))

  # add tokenizer tensors at the end (to maximize the chance of model tensor alignment)
  # note: we concatenate all bytes of all tokens into a single tensor
  tensors["tokenizer.tokens"] = torch.cat([torch.tensor([x for x in b] + [0], dtype=torch.uint8) for b in tokens])

  print(f"Saving {len(tensors)} tensors...")
  save_file(tensors, args.output, metadata.to_dict())