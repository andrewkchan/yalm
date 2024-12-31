#include "model.h"

#include "json.hpp"
#include <algorithm>
#include <array>
#include <cfloat>
#include "fmt/format.h"
#include <iostream>
#include <limits.h>
#include <string>

#include "immintrin.h"

using json = nlohmann::json;

void Config::from_yalm(YALMData& yalm, int context) {
  dim = std::stoi(yalm.metadata.at("dim").get<std::string>());
  hidden_dim = std::stoi(yalm.metadata.at("hidden_dim").get<std::string>());
  head_dim = std::stoi(yalm.metadata.at("head_dim").get<std::string>());
  n_layers = std::stoi(yalm.metadata.at("n_layers").get<std::string>());
  n_heads = std::stoi(yalm.metadata.at("n_heads").get<std::string>());
  n_kv_heads = std::stoi(yalm.metadata.at("n_kv_heads").get<std::string>());
  vocab_size = std::stoi(yalm.metadata.at("vocab_size").get<std::string>());
  // mixture of experts
  n_experts = std::stoi(yalm.metadata.at("n_experts").get<std::string>());
  n_experts_active = std::stoi(yalm.metadata.at("n_experts_active").get<std::string>());

  // for now limit seq_len to 4096 to avoid KV cache OOM for models like Mistral since window size isn't correctly specified
  max_seq_len = std::min(std::stoi(yalm.metadata.at("max_seq_len").get<std::string>()), 4096);
  if (context) {
    max_seq_len = context;
  }

  rope_theta = std::stof(yalm.metadata.at("rope_theta").get<std::string>());
  rotary_dim = std::stoi(yalm.metadata.at("rotary_dim").get<std::string>());

  norm_eps = std::stof(yalm.metadata.value("norm_eps", "1e-5"));

  std::string act_str = yalm.metadata.value("act_type", "gelu");
  if (act_str == "gelu") {
    act = ActivationType::GELU;
  } else if (act_str == "silu") {
    act = ActivationType::SILU;
  } else {
    std::cerr << "unsupported act_type, defaulting to gelu" << std::endl;
    act = ActivationType::GELU;
  }

  std::string norm_type_str = yalm.metadata.value("norm_type", "rmsnorm");
  if (norm_type_str == "rmsnorm") {
    norm_type = LayerNormType::RMSNorm;
  } else {
    std::cerr << "unsupported norm_type, defaulting to rmsnorm" << std::endl;
    norm_type = LayerNormType::RMSNorm;
  }

  qkv_clip = yalm.metadata.contains("qkv_clip") ? std::stof(yalm.metadata.at("qkv_clip").get<std::string>()) : FLT_MAX;

  std::string dtype = yalm.metadata.at("dtype").get<std::string>();
  // TODO: support fp8
  if (dtype == "fp32") {
    weight_dtype = DType::F32;
  } else if (dtype == "fp16") {
    weight_dtype = DType::F16;
  } else {
    std::cerr << "FATAL: unsupported dtype: " << dtype << std::endl;
    assert(false);
  }
}

size_t Config::active_bytes(size_t pos) const {
  size_t weight_size = dtype_size(weight_dtype);

  size_t bytes_per_block = 0;
  bytes_per_block += 2 * dim * sizeof(float); // rms_att_weight, rms_ffn_weight
  bytes_per_block += n_heads * head_dim * dim * weight_size; // wq
  bytes_per_block += 2 * n_kv_heads * head_dim * dim * weight_size; // wk, wv
  bytes_per_block += n_heads * head_dim * dim * weight_size; // wo
  bytes_per_block += 3 * dim * hidden_dim * weight_size; // w1, w2, w3
  size_t kv_len = std::min(static_cast<size_t>(max_seq_len), pos + 1);
  size_t kv_entry_size = sizeof(f16_t);
  bytes_per_block += 2 * kv_len * n_kv_heads * head_dim * kv_entry_size; // key_cache, value_cache

  size_t bytes = 0;
  bytes += dim * weight_size; // 1 row of token_embedding_table
  bytes += n_layers * bytes_per_block; // blocks
  bytes += dim * sizeof(float); // rms_final_weight
  bytes += vocab_size * dim * sizeof(float); // wcls

  return bytes;
}

void* check_tensor(const Tensor* tensor, DType weight_dtype, std::array<int, 4> shape) {
  if (tensor == nullptr) {
    std::cerr << "FATAL: missing tensor" << std::endl;
    assert(false);
    return nullptr;
  }
  if (tensor->dtype != weight_dtype || tensor->shape != shape) {
    std::cerr << "FATAL: tensor mismatch for " << tensor->name << std::endl;
    std::cerr 
      << fmt::format("expected: dtype={}, shape=[{},{},{},{}]", dtype_to_string(weight_dtype), shape[0], shape[1], shape[2], shape[3]) 
      << std::endl;
    std::cerr 
      << fmt::format("got: dtype={}, shape=[{},{},{},{}]", dtype_to_string(tensor->dtype), tensor->shape[0], tensor->shape[1], tensor->shape[2], tensor->shape[3]) 
      << std::endl;
    assert(false);
  }
  return tensor->data;
};

const Tensor* get_tensor(const YALMData& yalm, const std::string& key) {
  auto it = yalm.tensors.find(key);
  if (it == yalm.tensors.end()) {
    std::cerr << "FATAL: missing tensor: " << key << std::endl;
    assert(false);
    return nullptr;
  }
  const Tensor& tensor = it->second;
  return &tensor;
};

Block::Block(
  int layer_i,
  const std::shared_ptr<Config> config,
  const Tensor* rms_att_weight,
  const Tensor* rms_ffn_weight,
  const Tensor* wq,
  const Tensor* wk,
  const Tensor* wv,
  const Tensor* wo,
  const Tensor* w1,
  const Tensor* w2,
  const Tensor* w3
) {
  _layer_i = layer_i;
  _config = config;
  switch (config->weight_dtype) {
    case DType::F32:
    case DType::F16: {
      break;
    }
    default: {
      std::cerr << "FATAL: unsupported weight dtype " << dtype_to_string(config->weight_dtype) << std::endl;
      assert(false);
      break;
    }
  }

  _rms_att_weight = static_cast<float*>(check_tensor(
    rms_att_weight, DType::F32, {config->dim, 0, 0, 0}
  ));
  _rms_ffn_weight = static_cast<float*>(check_tensor(
    rms_ffn_weight, DType::F32, {config->dim, 0, 0, 0}
  ));

  _wq = check_tensor(
    wq, config->weight_dtype, {config->n_heads * config->head_dim, config->dim, 0, 0}
  );
  _wk = check_tensor(
    wk, config->weight_dtype, {config->n_kv_heads * config->head_dim, config->dim, 0, 0}
  );
  _wv = check_tensor(
    wv, config->weight_dtype, {config->n_kv_heads * config->head_dim, config->dim, 0, 0}
  );
  _wo = check_tensor(
    wo, config->weight_dtype, {config->dim, config->n_heads * config->head_dim, 0, 0}
  );

  _w1 = check_tensor(
    w1, config->weight_dtype, {config->hidden_dim, config->dim, 0, 0}
  );
  _w2 = check_tensor(
    w2, config->weight_dtype, {config->dim, config->hidden_dim, 0, 0}
  );
  _w3 = check_tensor(
    w3, config->weight_dtype, {config->hidden_dim, config->dim, 0, 0}
  );

  _key_cache = new f16_t[config->max_seq_len * config->n_kv_heads * config->head_dim]();
  _value_cache = new f16_t[config->max_seq_len * config->n_kv_heads * config->head_dim]();
}

Block::~Block() {
  if (_device == Device::CPU) {
    delete[] _key_cache;
    delete[] _value_cache;
  } else {
    free_cuda(_key_cache);
    free_cuda(_value_cache);
  }
}

void Block::cuda() {
  if (_device != Device::CPU) {
    return;
  }
  _device = Device::CUDA;
  size_t weight_size = dtype_size(_config->weight_dtype);
  // norms
  _rms_att_weight = static_cast<float*>(upload_cuda(_rms_att_weight, _config->dim * sizeof(float)));
  _rms_ffn_weight = static_cast<float*>(upload_cuda(_rms_ffn_weight, _config->dim * sizeof(float)));

  // self-attention
  _wq = upload_cuda(_wq, _config->n_heads * _config->head_dim * _config->dim * weight_size);
  _wk = upload_cuda(_wk, _config->n_kv_heads * _config->head_dim * _config->dim * weight_size);
  _wv = upload_cuda(_wv, _config->n_kv_heads * _config->head_dim * _config->dim * weight_size);
  _wo = upload_cuda(_wo, _config->dim * _config->n_heads * _config->head_dim * weight_size);

  // ffn
  _w1 = upload_cuda(_w1, _config->hidden_dim * _config->dim * weight_size);
  _w2 = upload_cuda(_w2, _config->dim * _config->hidden_dim * weight_size);
  _w3 = upload_cuda(_w3, _config->hidden_dim * _config->dim * weight_size);

  // kv cache
  _key_cache = static_cast<f16_t*>(upload_cuda(_key_cache, _config->max_seq_len * _config->n_kv_heads * _config->head_dim * sizeof(f16_t)));
  _value_cache = static_cast<f16_t*>(upload_cuda(_value_cache, _config->max_seq_len * _config->n_kv_heads * _config->head_dim * sizeof(f16_t)));
}

void Block::block(
  InferenceState& s,  // inference state
  int pos,            // index of the current token in the sequence
  int kv_sink,        // number of sink tokens currently in the KV cache
  int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
  int kv_len          // number of tokens in the kv cache that we will attend over
) const {
  if (_device == Device::CUDA) {
    switch (_config->weight_dtype) {
      case DType::F32: {
        _block_cuda<float>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      }
      case DType::F16: {
        _block_cuda<f16_t>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      }
      default: {
        assert(false && "unsupported weight dtype for cuda");
      }
    }
  } else {
    switch (_config->weight_dtype) {
      case DType::F32: {
        _block_cpu<float>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      }
      case DType::F16: {
#if defined(__AVX2__) && defined(__F16C__)
        _block_cpu<f16_t>(s, pos, kv_sink, kv_pos, kv_len);
#else
        assert(false && "float16 not supported on this platform");
#endif
        break;
      }
      default: {
        assert(false && "unsupported weight dtype for cpu");
      }
    }
  }

}

InferenceState::InferenceState(const std::shared_ptr<Config> config): 
  _config(config) {
  assert(config);
  _x = new float[config->dim]();
  _xb = new float[config->dim]();
  _xb2 = new float[config->dim]();
  _hb = new float[config->hidden_dim]();
  _hb2 = new float[config->hidden_dim]();
  _q = new float[config->n_heads * config->head_dim]();
  _k = new float[config->n_kv_heads * config->head_dim]();
  _v = new float[config->n_kv_heads * config->head_dim]();
  _att = new float[config->n_heads * config->max_seq_len]();
  _logits = new float[config->vocab_size]();
}

InferenceState::~InferenceState() {
  if (_device == Device::CPU) {
    delete[] _x;
    delete[] _xb;
    delete[] _xb2;
    delete[] _hb;
    delete[] _hb2;
    delete[] _q;
    delete[] _k;
    delete[] _v;
    delete[] _att;
    delete[] _logits;
  } else {
    free_cuda(_x);
    free_cuda(_xb);
    free_cuda(_xb2);
    free_cuda(_hb);
    free_cuda(_hb2);
    free_cuda(_q);
    free_cuda(_k);
    free_cuda(_v);
    free_cuda(_att);
    unregister_cuda_host(_logits);
    delete[] _logits;
  }
}

void InferenceState::cuda() {
  if (_device != Device::CPU) {
    return;
  }
  _device = Device::CUDA;
  init_cuda_stream(&_stream);
  _x = static_cast<float*>(upload_cuda(_x, _config->dim * sizeof(float)));
  _xb = static_cast<float*>(upload_cuda(_xb, _config->dim * sizeof(float)));
  _xb2 = static_cast<float*>(upload_cuda(_xb2, _config->dim * sizeof(float)));
  _hb = static_cast<float*>(upload_cuda(_hb, _config->hidden_dim * sizeof(float)));
  _hb2 = static_cast<float*>(upload_cuda(_hb2, _config->hidden_dim * sizeof(float)));
  _q = static_cast<float*>(upload_cuda(_q, _config->n_heads * _config->head_dim * sizeof(float)));
  _k = static_cast<float*>(upload_cuda(_k, _config->n_kv_heads * _config->head_dim * sizeof(float)));
  _v = static_cast<float*>(upload_cuda(_v, _config->n_kv_heads * _config->head_dim * sizeof(float)));
  _att = static_cast<float*>(upload_cuda(_att, _config->n_heads * _config->max_seq_len * sizeof(float)));
  register_cuda_host(_logits, _config->vocab_size * sizeof(float));
}

Model::Model(YALMData& yalm, int context) {
  config = std::make_shared<Config>();
  config->from_yalm(yalm, context);
  std::cout << "loading model with dtype: " << dtype_to_string(config->weight_dtype) << std::endl;

  token_embedding_table = check_tensor(
    get_tensor(yalm, "model.embed.weight"), 
    config->weight_dtype,
    {config->vocab_size, config->dim, 0, 0}
  );

  for (int i = 0; i < config->n_layers; ++i) {
    blocks.emplace_back(std::make_shared<Block>(
      i,
      config,
      get_tensor(yalm, fmt::format("model.layers.{}.attn.norm.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.norm.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wq.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wk.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wv.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wo.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.w1.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.w2.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.w3.weight", i))
    ));
  }

  rms_final_weight = static_cast<float*>(check_tensor(
    get_tensor(yalm, "model.norm.weight"), 
    DType::F32, 
    {config->dim, 0, 0, 0}
  ));
  bool tie_word_embeddings = yalm.tensors.count("model.output.weight") == 0;
  if (tie_word_embeddings) {
    wcls = token_embedding_table;
  } else {
    wcls = check_tensor(
      get_tensor(yalm, "model.output.weight"), 
      config->weight_dtype, 
      {config->vocab_size, config->dim, 0, 0}
    );
  }
}

void Model::cuda() {
  if (_device != Device::CPU) {
    return;
  }
  _device = Device::CUDA;
  // TODO: support multiple CUDA devices
  set_cuda_device(0);
  size_t weight_size = dtype_size(config->weight_dtype);
  token_embedding_table = upload_cuda(token_embedding_table, config->vocab_size * config->dim * weight_size);
  for (auto& block : blocks) {
    block->cuda();
  }
  rms_final_weight = static_cast<float*>(upload_cuda(rms_final_weight, config->dim * sizeof(float)));
  wcls = upload_cuda(wcls, config->vocab_size * config->dim * weight_size);
}

void Model::forward(InferenceState& s, int token, int pos, InferenceMode mode) {
  if (s.device() != _device) {
    std::cerr << "FATAL: inference state device mismatch" << std::endl;
    assert(false);
    return;
  }
  if (_device == Device::CUDA) {
    _forward_cuda(s, token, pos, mode);
  } else {
    _forward_cpu(s, token, pos, mode);
  }
}

#if DEBUG_MODEL
DebugTensor::DebugTensor(const std::vector<float>& data) {
  data_f32 = data;
  data_type = DataType::F32;
}
DebugTensor::DebugTensor(const std::vector<f16_t>& data) {
  data_f16 = data;
  data_type = DataType::F16;
}

float DebugTensor::max_err(const DebugTensor& other) const {
  if (data_type != other.data_type) {
    return -1;
  }
  if (data_type == DataType::F32) {
    float max_err = 0;
    for (size_t i = 0; i < data_f32.size(); i++) {
      max_err = std::max(max_err, std::abs(data_f32[i] - other.data_f32[i]));
    }
    return max_err;
  } else {
#if defined(__F16C__)
    float max_err = 0;
    for (size_t i = 0; i < data_f16.size(); i++) {
      max_err = std::max(max_err, std::abs(_cvtsh_ss(data_f16[i]) - _cvtsh_ss(other.data_f16[i])));
    }
    return max_err;
#else
  assert(false && "float16 not supported on this platform");
#endif
  }
}
#endif