#include "model.h"

#include <assert.h>
#include <cfloat>
#include <math.h>

#include "immintrin.h"
#include "f16cintrin.h"

#if defined(__AVX2__) && defined(__F16C__)
inline float half_to_float(f16_t x) {
  return _cvtsh_ss(x);
}
inline f16_t float_to_half(float x) {
  return _cvtss_sh(x, 0);
}
#else
inline float half_to_float(f16_t x) {
  assert(false && "float16 not supported on this platform");
  return 0.0f;
}
inline f16_t float_to_half(float x) {
  assert(false && "float16 not supported on this platform");
  return 0;
}
#endif

union FP32 {
  unsigned int u;
  float f;
};

// Un-optimized bfloat16 conversion routines.
inline float bfloat16_to_float(bf16_t x) {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(&x);
  uint16_t q[2];
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  q[0] = *p;
  q[1] = 0;
#else
  q[0] = 0;
  q[1] = *p;
#endif
  float out;
  memcpy(&out, q, sizeof(float));
  return out;
}
// Convert a float32 to bfloat16 with round-to-nearest-even rounding.
// See:
// https://hhhhhojeihsu.github.io/tensorflow_1.8_woboq/tensorflow_1.8_xla/tensorflow/tensorflow/core/lib/bfloat16/bfloat16.h.html
// for a long explanation of what this is doing.
inline bf16_t float_to_bfloat16(float v) {
  uint32_t input;
  FP32 f;
  f.f = v;
  input = f.u;
  uint16_t output;
  if (std::isnan(v)) {
    output = 0x7fc0;
  } else {
    uint32_t lsb = (input >> 16) & 1;
    uint32_t rounding_bias = 0x7fff + lsb;
    input += rounding_bias;
    output = static_cast<uint16_t>(input >> 16);
  }
  bf16_t result;
  memcpy(&result, &output, sizeof(bf16_t));
  return result;
}

#if DEBUG_MODEL
#include "fmt/format.h"
static std::map<std::string, DebugTensor> _debug_map;
std::map<std::string, DebugTensor>& debug_map_cpu() {
  return _debug_map;
}
template <typename T>
static std::vector<T> copy_debug_tensor(T* x, size_t size) {
  std::vector<T> out(size);
  for (size_t i = 0; i < size; i++) {
    out[i] = x[i];
  }
  return out;
}
template <typename T>
static void save_debug_tensor(const std::string& name, T* x, size_t size) {
  _debug_map[name] = DebugTensor(copy_debug_tensor<T>(x, size));
}
#endif

// matmul supporting float32 weights. This is typically auto-vectorized by gcc.
static void matmul(float* xout, float* x, float* w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
}

// matmul supporting float16 weights via the F16C extension, which allows
// conversion into float32 values before calculations.
static void matmul(float* xout, float* x, f16_t* w, int n, int d) {
#if defined(__AVX2__) && defined(__F16C__)
  // W (d,n) @ x (n,) -> xout (d,)
  assert(n % 16 == 0);
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    // Vectorized dot product of w[i][:] and x[:] where w is a packed float16 array.
    __m256 sumlo = _mm256_setzero_ps();
    __m256 sumhi = _mm256_setzero_ps();
    for (int j = 0; j < n; j+=16) {
      // Extract the next set of 16 float16 weights from `w` and store them
      // to two separate float32 vectors of width 8 (`wveclo_ps`, `wvechi_ps`)
      __m256i wvec = _mm256_loadu_si256((__m256i*)&w[i * n + j]);
      __m128i wveclo = _mm256_extractf128_si256(wvec, 0);
      __m128i wvechi = _mm256_extractf128_si256(wvec, 1);
      __m256 wveclo_ps = _mm256_cvtph_ps(wveclo);
      __m256 wvechi_ps = _mm256_cvtph_ps(wvechi);
      // Extract the next two float32 vectors of width 8 `xveclo`, `xvechi` from `x`
      __m256 xveclo = _mm256_loadu_ps(&x[j]);
      __m256 xvechi = _mm256_loadu_ps(&x[j + 8]);
      // Compute vectorized FMAs: sumlo += wveclo * xveclo, sumhi += wvechi * xvechi
      sumlo = _mm256_fmadd_ps(wveclo_ps, xveclo, sumlo);
      sumhi = _mm256_fmadd_ps(wvechi_ps, xvechi, sumhi);
    }
    // Horizontally reduce width-8 float32 vectors sumlo, sumhi to a scalar.
    __m256 sum8 = _mm256_add_ps(sumlo, sumhi);              // sum8[0:8] = sumlo[0:8] + sumhi[0:8]
    __m128 sum4 = _mm_add_ps(                               // sum4[0:4] = sum8[0:4] + sum8[4:8]
      _mm256_extractf128_ps(sum8, 0), 
      _mm256_extractf128_ps(sum8, 1)
    );
    __m128 sum1 = _mm_dp_ps(sum4, _mm_set1_ps(1.0f), 0xf1); // sum1[0] = dot(sum4, [1,1,1,1])
    xout[i] = _mm_cvtss_f32(sum1);
  }
#else
  assert(false && "float16 not supported on this platform");
#endif
}

// Un-optimized matmul supporting bfloat16 weights.
// TODO: Vectorize.
static void matmul(float* xout, float* x, bf16_t* w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += bfloat16_to_float(w[i * n + j]) * x[j];
    }
    xout[i] = val;
  }
}

static void moe_gate(float* moe_weights, int* active_experts, float* x, int n_experts, int n_active_experts) {
  // Set moe_weights[:n_active_experts] to the weights of the top K experts.
  // Set active_experts[:n_active_experts] to the indices of the top K experts.

  // get the max weight for later softmax computation
  float max_val = -FLT_MAX;
  for (int j = 0; j < n_experts; ++j) {
    if (x[j] > max_val) {
      max_val = x[j];
    }
  }
  
  // top k
  uint64_t mask = 0;
  float wsum = 0.0f;
  for (int k = 0; k < n_active_experts; ++k) {
    int best = -1;
    for (int j = 0; j < n_experts; ++j) {
      if ((mask & (1ull << j)) == 0 && (best == -1 || x[j] > x[best])) {
        best = j;
      }
    }

    active_experts[k] = best;
    wsum += expf(x[active_experts[k]] - max_val);
    mask |= 1ull << best;
  }

  // normalize top k weights to obtain the softmax result
  for (int k = 0; k < n_active_experts; ++k) {
    moe_weights[k] = expf(x[active_experts[k]] - max_val) / wsum;
  }
}

static void rmsnorm(float* o, float* x, float* weight, int size, float eps) {
  float rms = 0.0f;
  for (int i = 0; i < size; ++i) {
    rms += x[i] * x[i];
  }
  rms = sqrtf(rms / size + eps);
  float scale = 1.0f / rms;
  for (int i = 0; i < size; ++i) {
    o[i] = x[i] * scale * weight[i];
  }
}

[[maybe_unused]] static void layernorm(float* o, float* x, float* weight, float* bias, int size, float eps) {
  float mean = 0.0f;
  for (int i = 0; i < size; ++i) {
    mean += x[i];
  }
  mean /= size;
  float var = 0.0f;
  for (int i = 0; i < size; ++i) {
    var += (x[i] - mean) * (x[i] - mean);
  }
  var /= size;
  float scale = 1.0f / sqrtf(var + eps);
  if (bias) {
    for (int i = 0; i < size; ++i) {
      o[i] = (x[i] - mean) * scale * weight[i] + bias[i];
    }
  } else {
    for (int i = 0; i < size; ++i) {
      o[i] = (x[i] - mean) * scale * weight[i];
    }
  }
}

// Compute the softmax of an input vector `x` of length `size` and store it in `o`.
static void softmax(float* o, float* x, int size) {
  float score_max = -FLT_MAX;
  for (int i = 0; i < size; ++i) {
    if (x[i] > score_max) {
      score_max = x[i];
    }
  }
  float score_sum = 0.0f;
  for (int i = 0; i < size; ++i) {
    o[i] = expf(x[i] - score_max);
    score_sum += o[i];
  }
  for (int i = 0; i < size; ++i) {
    o[i] /= score_sum;
  }
}

inline float gelu(float x) {
  return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

inline float silu(float x) {
  return x / (1.0f + expf(-x));
}

inline float clip(float x, float v) {
  return x < -v ? -v : (x > v ? v : x);
}

// TODO annotate me
static void rope(float* vec, int d, int head_dim, int pos, float theta, int rotary_dim) {
  for (int i = 0; i < d; i += 2) {
    int j_head = i % head_dim;
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    float v0 = vec[i];
    float v1 = vec[i + 1];
    vec[i] = v0 * fcr - v1 * fci;
    vec[i + 1] = v0 * fci + v1 * fcr;
  }
}

// Compute next value in a sequence for a single causal self-attention head.
void attn(
  float* xout,    // (dim,) - output vector
  float* atth,    // (kv_len,) - scratch space to hold attention scores of the sequence
  float* qh,      // (head_dim,) - query vector for this head
  f16_t* kh,      // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
  f16_t* vh,      // (kv_len, n_kv_heads, head_dim) - buffer containing value vectors of the sequence for all KV heads
  int head_dim,   // size of the "key-space"
  int n_kv_heads, // number of kv heads, can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
  int kv_len      // number of tokens of the sequence we will attend over
) {
  int kv_stride = n_kv_heads * head_dim; // stride per token in this kv head
  // calculate attention scores as dot products of q and k
  for (int t = 0; t < kv_len; ++t) {
    float score = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      score += qh[i] * half_to_float(kh[t * kv_stride + i]);
    }
    score /= sqrtf(head_dim);
    atth[t] = score;
  }

  // softmax the scores to get attention weights over [0..kv_len)
  softmax(atth, atth, kv_len);

  // mix values with attention weights
  for (int i = 0; i < head_dim; ++i) {
    float vi = 0.0f;
    for (int t = 0; t < kv_len; ++t) {
      vi += atth[t] * half_to_float(vh[t * kv_stride + i]);
    }
    xout[i] = vi;
  }
}

// Compute forward pass for a single block and update the inference state accordingly.
// PRECONDITIONS: 
// - `s.x()` contains the input to the block. Output will also go here.
// - Block KV cache is hydrated.
template <typename T>
void Block::_block_cpu(
  InferenceState& s,  // inference state
  int pos,            // index of the current token in the sequence
  int kv_sink,        // number of sink tokens currently in the KV cache
  int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
  int kv_len          // number of tokens in the kv cache that we will attend over
) const {
  assert(_config);
  const Config& c = *_config;

  // attention pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm(s.xb(), s.x(), rms_att_weight(), c.dim, c.norm_eps);
      break;
    }
  }

  int q_dim = c.n_heads * c.head_dim;
  int kv_dim = c.n_kv_heads * c.head_dim;

  // qkv matmuls for this position
  matmul(s.q(), s.xb(), wq<T>(), c.dim, q_dim);
  matmul(s.k(), s.xb(), wk<T>(), c.dim, kv_dim);
  matmul(s.v(), s.xb(), wv<T>(), c.dim, kv_dim);

  // some models require clipping qkv values
  for (int i = 0; i < q_dim; ++i) {
    s.q()[i] = clip(s.q()[i], c.qkv_clip);
  }
  for (int i = 0; i < kv_dim; ++i) {
    s.k()[i] = clip(s.k()[i], c.qkv_clip);
    s.v()[i] = clip(s.v()[i], c.qkv_clip);
  }

  // RoPE relative positional encoding: complex-valued rotate q and k in each head
  rope(s.q(), q_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim);
  rope(s.k(), kv_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim);
  
  // key and value point to the kv cache
  f16_t* kb = key_cache();
  f16_t* vb = value_cache();
  // update kv cache
  for (int i = 0; i < kv_dim; ++i) {
    kb[kv_pos * kv_dim + i] = float_to_half(s.k()[i]);
    vb[kv_pos * kv_dim + i] = float_to_half(s.v()[i]);
  }

  // Sink tokens remain untouched while the rest of the KV cache is incrementally 
  // replaced in ring order, but sink i must always be positioned (max_seq_len - i)
  // away from current timestep. Hence, each forward pass, rotate sink tokens 
  // forward by 1. See https://arxiv.org/abs/2309.17453 for more.
  for (int r = 0; r < kv_sink; r++) {
    for (int i = 0; i < kv_dim; ++i) {
      s.k()[i] = half_to_float(kb[r * kv_dim + i]);
    }

    rope(s.k(), kv_dim, c.head_dim, 1, c.rope_theta, c.rotary_dim);

    for (int i = 0; i < kv_dim; i++) {
      kb[r * kv_dim + i] = float_to_half(s.k()[i]);
    }
  }

  // Multihead attention. Iterate over all heads.
  int q_per_kv_head = c.n_heads / c.n_kv_heads; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < c.n_heads; h++) {
    int kv_head_offset = (h / q_per_kv_head) * c.head_dim;
    f16_t* kh = kb + kv_head_offset;
    f16_t* vh = vb + kv_head_offset;
    attn(s.xb2(h), s.att(h), s.q(h), kh, vh, c.head_dim, c.n_kv_heads, kv_len);
  }

  // final matmul to get output of the attention, using `hb` as temp storage
  matmul(s.hb(), s.xb2(), wo<T>(), q_dim, c.dim);

  // residual connection back into x
  for (int i = 0; i < c.dim; ++i) {
    s.x()[i] += s.hb()[i];
  }
  
  // ffn pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm(s.xb(), s.x(), rms_ffn_weight(), c.dim, c.norm_eps);
      break;
    }
  }

  if (c.n_experts > 0) {
    matmul(s.moe_weights(), s.xb(), moegate<T>(), c.dim, c.n_experts);
    moe_gate(s.active_experts_weights(), s.active_experts(), s.moe_weights(), c.n_experts, c.n_experts_active);
  } else {
    s.active_experts_weights()[0] = 1.0f;
    s.active_experts()[0] = 0;
  }

  for (int k = 0; k < (c.n_experts > 0 ? c.n_experts_active : 1); ++k) {
    int expert_index = s.active_experts()[k];
    int expert_size = c.dim * c.hidden_dim;
    // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
    // Note this is a feedforward with a GLU, not a simple MLP.
    matmul(s.hb(), s.xb(), w1<T>() + expert_index * expert_size, c.dim, c.hidden_dim);
    matmul(s.hb2(), s.xb(), w3<T>() + expert_index * expert_size, c.dim, c.hidden_dim);
    switch (c.act) {
      case ActivationType::GELU: {
        for (int i = 0; i < c.hidden_dim; ++i) {
          s.hb()[i] = gelu(s.hb()[i]) * s.hb2()[i];
        }
        break;
      }
      case ActivationType::SILU: {
        for (int i = 0; i < c.hidden_dim; ++i) {
          s.hb()[i] = silu(s.hb()[i]) * s.hb2()[i];
        }
        break;
      }
    }

    matmul(s.xb2(), s.hb(), w2<T>() + expert_index * expert_size, c.hidden_dim, c.dim);

    float expert_weight = s.active_experts_weights()[k];
    // residual connection back into x
    for (int i = 0; i < c.dim; ++i) {
      s.x()[i] += s.xb2()[i] * expert_weight;
    }
  }
}

void mha_cpu(
  float* xout,  // (n_heads, head_dim)
  float* att,   // (n_heads, max_seq_len)
  f16_t* kb,    // (max_seq_len, n_kv_heads, head_dim)
  f16_t* vb,    // (max_seq_len, n_kv_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
) {
  // Multihead attention. Iterate over all heads.
  int q_per_kv_head = n_heads / n_kv_heads; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < n_heads; h++) {
    int kv_head_offset = (h / q_per_kv_head) * head_dim;
    f16_t* kh = kb + kv_head_offset;
    f16_t* vh = vb + kv_head_offset;
    attn(
      xout + head_dim * h, att + max_seq_len * h, q + head_dim * h, 
      kh, vh, head_dim, n_kv_heads, kv_len
    );
  }
}

void matmul_cpu(float* xout, float* x, float* w, int n, int d) {
  matmul(xout, x, w, n, d);
}
void matmul_cpu(float* xout, float* x, f16_t* w, int n, int d) {
  matmul(xout, x, w, n, d);
}

void ffn_cpu(
  float* xout, float* x, 
  float* w1, float* w2, float* w3, 
  int hidden_dim, int dim,
  ActivationType act
) {
  float* hb = new float[hidden_dim];
  float* hb2 = new float[hidden_dim];
  // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
  // Note this is a feedforward with a GLU, not a simple MLP.
  matmul(hb, x, w1, dim, hidden_dim);
  matmul(hb2, x, w3, dim, hidden_dim);
  switch (act) {
    case ActivationType::GELU: {
      for (int i = 0; i < hidden_dim; ++i) {
        hb[i] = gelu(hb[i]) * hb2[i];
      }
      break;
    }
    case ActivationType::SILU: {
      for (int i = 0; i < hidden_dim; ++i) {
        hb[i] = silu(hb[i]) * hb2[i];
      }
      break;
    }
  }

  matmul(xout, hb, w2, hidden_dim, dim);
  
  delete[] hb;
  delete[] hb2;
}

template void Block::_block_cpu<float>(InferenceState&, int, int, int, int) const;
template void Block::_block_cpu<f16_t>(InferenceState&, int, int, int, int) const;
template void Block::_block_cpu<bf16_t>(InferenceState&, int, int, int, int) const;

void Model::_copy_embedding(InferenceState& s, int token) {
  const Config& c = *config;
  switch (c.weight_dtype) {
    case DType::F32: {
      float* emb = static_cast<float*>(token_embedding_table);
      for (int i = 0; i < c.dim; ++i) {
        s.x()[i] = emb[token * c.dim + i];
      }
      break;
    }
    case DType::F16: {
      f16_t* emb = static_cast<f16_t*>(token_embedding_table);
      for (int i = 0; i < c.dim; i+=1) {
        s.x()[i] = half_to_float(emb[token * c.dim + i]);
      }
      break;
    }
    case DType::BF16: {
      bf16_t* emb = static_cast<bf16_t*>(token_embedding_table);
      for (int i = 0; i < c.dim; i+=1) {
        s.x()[i] = bfloat16_to_float(emb[token * c.dim + i]);
      }
      break;
    }
    default: {
      assert(false && "unsupported weight dtype");
    }
  }
}

void Model::_forward_cpu(InferenceState& s, int token, int pos, InferenceMode mode) {
  const Config& c = *config;

  // copy the token embedding into `x`
  _copy_embedding(s, token);

  // When decoding past the context length, keep the first few tokens in the KV cache
  // untouched as "attention sinks" while replacing the rest in ring order.
  // See StreamingLLM (https://arxiv.org/pdf/2309.17453) for more.
  int kv_sink = pos >= c.max_seq_len ? KV_SINKS : 0;
  int kv_pos = kv_sink + (pos - kv_sink) % (c.max_seq_len - kv_sink);
  int kv_len = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;

  // forward all layers in order
  for (auto b : blocks) {
    b->block(s, pos, kv_sink, kv_pos, kv_len);
  }

  if (mode == InferenceMode::HYDRATE_KV_CACHE) {
    // only hydrate the KV cache and don't compute output logits
    return;
  }

  // final layer norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm(s.x(), s.x(), rms_final_weight, c.dim, c.norm_eps);
      break;
    }
  }

  // classifier into logits
  switch (c.weight_dtype) {
    case DType::F32: {
      matmul(s.logits(), s.x(), static_cast<float*>(wcls), c.dim, c.vocab_size);
      break;
    }
    case DType::F16: {
      matmul(s.logits(), s.x(), static_cast<f16_t*>(wcls), c.dim, c.vocab_size);
      break;
    }
    case DType::BF16: {
      matmul(s.logits(), s.x(), static_cast<bf16_t*>(wcls), c.dim, c.vocab_size);
      break;
    }
    default: {
      assert(false && "unsupported weight dtype");
    }
  }
}