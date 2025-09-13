#include <iostream>
#include <memory>
#include <omp.h>
#include <random>
#include <thread>
#include <vector>

#include "immintrin.h"
#include "fmt/format.h"

#include "model.h"
#include "time.h"

bool floatEquals(float a, float b, float epsilon = 1e-5) {
  return std::abs(a - b) < epsilon;
}

bool arrayEquals(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-4) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    if (!floatEquals(a[i], b[i], epsilon)) {
      return false;
    }
  }
  return true;
}

void assertArrayEquals(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& message, float epsilon = 1e-4) {
  if (!arrayEquals(actual, expected, epsilon)) {
    std::cerr << "Assertion failed: " << message << std::endl;
    std::cerr << "actual: ";
    for (size_t i = 0; i < actual.size(); i++) {
      std::cerr << actual[i] << " ";
    }
    std::cerr << std::endl;
    std::cerr << "expected: ";
    for (size_t i = 0; i < expected.size(); i++) {
      std::cerr << expected[i] << " ";
    }
    std::cerr << std::endl;
    exit(1);
  }
}

void assertArrayEquals(float* actual, const std::vector<float>& expected, const std::string& message) {
  std::vector<float> actual_array;
  for (size_t i = 0; i < expected.size(); i++) {
    actual_array.push_back(actual[i]);
  }
  assertArrayEquals(actual_array, expected, message);
}

std::vector<f16_t> float_array_to_half(const std::vector<float>& data) {
#if defined(__AVX2__) && defined(__F16C__)
  std::vector<f16_t> half_data(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    half_data[i] = _cvtss_sh(data[i], 0);
  }
  return half_data;
#else
  assert(false && "Cannot convert to half due to missing F16C extensions");
  return {};
#endif
}

void test_attn() {
  constexpr int TEST_SEQ_LEN = 4;
  constexpr int TEST_DIM = 6;
  constexpr int TEST_HEAD_DIM = 3;
  constexpr int TEST_N_HEADS = 2;
  constexpr int TEST_N_KV_HEADS = 1;
  std::shared_ptr<Config> config = std::make_shared<Config>();
  config->dim = TEST_DIM;
  config->hidden_dim = TEST_DIM;
  config->head_dim = TEST_HEAD_DIM;
  config->n_heads = TEST_N_HEADS;
  config->n_kv_heads = TEST_N_KV_HEADS;
  config->vocab_size = 1;
  config->max_seq_len = TEST_SEQ_LEN;
  InferenceState s(config);
  // (n_heads, head_dim) - query vectors
  std::vector<float> q{
    0., 1e4, 0., // h=0
    0., 0., 1e4 // h=1
  };
  for (size_t i = 0; i < q.size(); i++) {
    s.q()[i] = q[i];
  }
  std::vector<f16_t> kb = float_array_to_half({
    1., 0., 0., // t=0
    0., 1., 0., // t=1
    0., 0., 1., // t=2
    -1., 0., 0. // t=3
  }); // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
  std::vector<f16_t> vb = float_array_to_half({
    1., 0., 0., // t=0
    0., 1., 0., // t=1
    0., 0., 1., // t=2
    -1., 0., 0. // t=3
  }); // (kv_len, n_kv_heads, head_dim) - buffer containing value vectors of the sequence for all KV heads

  // Multihead attention. Iterate over all heads.
  int q_per_kv_head = TEST_N_HEADS / TEST_N_KV_HEADS; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < TEST_N_HEADS; h++) {
    int kv_head_offset = (h / q_per_kv_head) * TEST_HEAD_DIM;
    f16_t* kh = kb.data() + kv_head_offset;
    f16_t* vh = vb.data() + kv_head_offset;
    attn(s.xb(h), s.att(h), s.q(h), kh, vh, TEST_HEAD_DIM, TEST_N_KV_HEADS, TEST_SEQ_LEN);
  }
  // attention scores
  // h=0
  assertArrayEquals(s.att(0), {
    0., 1., 0., 0.
  }, "att(h=0)");
  // h=1
  assertArrayEquals(s.att(1), {
    0., 0., 1., 0.
  }, "att(h=1)");
  assertArrayEquals(s.xb(), {
    0., 1., 0., // h=0
    0., 0., 1. // h=1
  }, "xout");
}

void fill_random(float* data, size_t N, unsigned long seed, float scale_factor = 1.0) {
  std::default_random_engine gen(seed);
  std::normal_distribution<float> dist(0.0, 1.0);
  for (size_t i = 0; i < N; i++) {
    data[i] = dist(gen) * scale_factor;
  }
}

void fill_random(f16_t* data, size_t N, unsigned long seed, float scale_factor = 1.0) {
#if defined(__AVX2__) && defined(__F16C__)
  std::default_random_engine gen(seed);
  std::normal_distribution<float> dist(0.0, 1.0);
  for (size_t i = 0; i < N; i++) {
    data[i] = _cvtss_sh(dist(gen) * scale_factor, 0);
  }
#else
  assert(false && "Cannot fill_random due to missing F16C extensions");
#endif
}

void test_cuda_kernels() {
  int head_dim = 128;
  int n_heads = 32;
  int dim = head_dim * n_heads;
  int hidden_dim = dim;
  int n_kv_heads = 8;
  int max_seq_len = 4096;

  // matmul
  {
    std::vector<float> w(dim * head_dim);
    fill_random(w.data(), w.size(), 0);
    std::vector<float> x(dim);
    fill_random(x.data(), x.size(), 1);
    std::vector<float> xout_cpu(head_dim);
    std::vector<float> xout_cuda(head_dim);
    matmul_cpu(xout_cpu.data(), x.data(), w.data(), dim, head_dim);
    matmul_cuda(xout_cuda.data(), x.data(), w.data(), dim, head_dim);
    assertArrayEquals(xout_cuda, xout_cpu, "matmul");
  }

  // mha
  for (int kv_len : {100, 400, 800, 2000, 4096}) {
    for (bool use_flash_attention : {true, false}) {
      std::vector<f16_t> kb(max_seq_len * n_kv_heads * head_dim);
      fill_random(kb.data(), kb.size(), 0);
      std::vector<f16_t> vb(max_seq_len * n_kv_heads * head_dim);
      fill_random(vb.data(), vb.size(), 1);
      std::vector<float> q(n_heads * head_dim);
      fill_random(q.data(), q.size(), 2);
      std::vector<float> att_cpu(n_heads * max_seq_len);
      std::vector<float> att_cuda(n_heads * max_seq_len);
      std::vector<float> xout_cpu(n_heads * head_dim);
      std::vector<float> xout_cuda(n_heads * head_dim);
      mha_cpu(
        xout_cpu.data(),
        att_cpu.data(),
        kb.data(), 
        vb.data(), 
        q.data(), 
        head_dim, kv_len, max_seq_len, n_heads, n_kv_heads
      );
      mha_cuda(
        xout_cuda.data(), 
        att_cuda.data(),
        kb.data(), 
        vb.data(), 
        q.data(), 
        head_dim, kv_len, max_seq_len, n_heads, n_kv_heads,
        use_flash_attention
      );
      if (!use_flash_attention) {
        assertArrayEquals(att_cuda, att_cpu, "mha att");
      }
      assertArrayEquals(xout_cuda, xout_cpu, fmt::format("mha xout (use_flash_attention={})", use_flash_attention));
    }
  }

  // ffn
  {
    std::vector<float> x(dim);
    fill_random(x.data(), x.size(), 0);
    std::vector<float> w1(dim * hidden_dim);
    fill_random(w1.data(), w1.size(), 1, 1.0 / sqrtf(dim)); 
    std::vector<float> w2(hidden_dim * dim);
    fill_random(w2.data(), w2.size(), 2, 1.0 / sqrtf(hidden_dim));
    std::vector<float> w3(dim * hidden_dim);
    fill_random(w3.data(), w3.size(), 3, 1.0 / sqrtf(dim));
    std::vector<float> xout_cpu(dim);
    std::vector<float> xout_cuda(dim);
    ffn_cpu(
      xout_cpu.data(), 
      x.data(), 
      w1.data(), 
      w2.data(), 
      w3.data(), 
      hidden_dim, dim, 
      ActivationType::GELU
    );
    ffn_cuda(
      xout_cuda.data(), 
      x.data(), 
      w1.data(), 
      w2.data(), 
      w3.data(), 
      hidden_dim, dim, 
      ActivationType::GELU
    );
    assertArrayEquals(xout_cuda, xout_cpu, "ffn");
  }
}

// Helper function to allocate aligned memory
float* allocateAlignedArray(size_t N) {
  // Allocate aligned memory (64-byte alignment for AVX-512)
  void* ptr = nullptr;
  if (posix_memalign(&ptr, 64, N * sizeof(float)) != 0) {
    throw std::bad_alloc();
  }
  return static_cast<float*>(ptr);
}

void mem_bench() {
  constexpr size_t N_THREADS = 32;
  constexpr size_t MB_PER_THREAD = 1024;
  constexpr size_t ELS_PER_THREAD = (MB_PER_THREAD * 1024 * 1024) / sizeof(float);
  constexpr size_t N = N_THREADS * ELS_PER_THREAD;

  std::cout << "Using " << N_THREADS << " threads" << std::endl;
  std::cout << "Allocating " << N_THREADS * MB_PER_THREAD << " MB (" << N << " floats)" << std::endl;
  float* data = allocateAlignedArray(N);

  std::cout << "Filling data..." << std::endl;
#pragma omp parallel for num_threads(N_THREADS)
  for (size_t i = 0; i < N_THREADS; i++) {
    fill_random(data + i * ELS_PER_THREAD, ELS_PER_THREAD, (unsigned long)i);
  }
  std::cout << "Running memory bandwidth test..." << std::endl;

  float totalSum = 0.0;
  uint64_t start = get_timestamp_ms();
#pragma omp parallel for simd reduction(+:totalSum) schedule(guided) aligned(data: 64) num_threads(N_THREADS)
  for (size_t i = 0; i < N; i++) {
    totalSum += data[i];
  }
    
  uint64_t end = get_timestamp_ms();
  float elapsed_s = (end - start) / 1000.0;
  float mb_per_s = N_THREADS * MB_PER_THREAD / elapsed_s;

  std::cout << "Total sum: " << totalSum << std::endl;
  std::cout << "Elapsed time: " << elapsed_s << " s" << std::endl;
  std::cout << "Memory bandwidth: " << mb_per_s << " MB/s" << std::endl;
}

// 64 is the typical cache line size
struct alignas(64) ThreadData {
  volatile uint32_t sink;
  char padding[60]; // Ensures 64-byte alignment/padding
};

void mem_bench2_thread(uint32_t* data, size_t start_idx, size_t elements_per_thread, ThreadData* thread_sink) {
  for (size_t i = start_idx; i < start_idx + elements_per_thread; i++) {
    // 32-bit load stored in volatile to prevent optimization
    thread_sink->sink = data[i];
  }
}

void mem_bench2() {
  constexpr size_t N_THREADS = 64;
  constexpr size_t MB_PER_THREAD = 2048;
  constexpr size_t ELS_PER_THREAD = (MB_PER_THREAD * 1024 * 1024) / sizeof(uint32_t);
  constexpr size_t N = N_THREADS * ELS_PER_THREAD;

  std::cout << "Using " << N_THREADS << " threads" << std::endl;
  std::cout << "Allocating " << N_THREADS * MB_PER_THREAD << " MB (" << N << " uint32_t)" << std::endl;
  uint32_t* data = new uint32_t[N];

  std::cout << "Filling data..." << std::endl;
#pragma omp parallel for num_threads(N_THREADS)
  for (size_t i = 0; i < N_THREADS; i++) {
    for (size_t j = 0; j < ELS_PER_THREAD; j++) {
      data[i * ELS_PER_THREAD + j] = i + j;
    }
  }
  std::cout << "Running memory bandwidth test..." << std::endl;

  // Allocate cache-line aligned sinks for each thread
  std::vector<ThreadData> thread_sinks(N_THREADS);

  uint64_t start = get_timestamp_ms();
  std::vector<std::thread> threads;
  
  // Launch threads
  for (size_t i = 0; i < N_THREADS; i++) {
    threads.emplace_back(mem_bench2_thread, 
      data,
      i * ELS_PER_THREAD, 
      ELS_PER_THREAD,
      &thread_sinks[i]
    );
  }
  
  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }
    
  uint64_t end = get_timestamp_ms();
  float elapsed_s = (end - start) / 1000.0;
  float mb_per_s = N_THREADS * MB_PER_THREAD / elapsed_s;

  std::cout << "Elapsed time: " << elapsed_s << " s" << std::endl;
  std::cout << "Memory bandwidth: " << mb_per_s << " MB/s" << std::endl;
}

void kernel_bench(const std::string& kernel_name) {
  int head_dim = 128;
  int n_heads = 32;
  int dim = head_dim * n_heads;
  int hidden_dim = 14336;
  int n_kv_heads = 8;
  int max_seq_len = 4096;
  int kv_len = 4096;

  if (kernel_name == "matmul") {
    std::vector<f16_t> w(dim * hidden_dim);
    fill_random(w.data(), w.size(), 0);
    std::vector<float> x(dim);
    fill_random(x.data(), x.size(), 1);
    std::vector<float> xout_cuda(hidden_dim);
    matmul_cuda<f16_t>(xout_cuda.data(), x.data(), w.data(), dim, hidden_dim);
  } else if (kernel_name == "matmul-wide") {
    int hidden_dim = 32000;

    std::vector<f16_t> w(dim * hidden_dim);
    fill_random(w.data(), w.size(), 0);
    std::vector<float> x(dim);
    fill_random(x.data(), x.size(), 1);
    std::vector<float> xout_cuda(hidden_dim);
    matmul_cuda<f16_t>(xout_cuda.data(), x.data(), w.data(), dim, hidden_dim);
  } else if (kernel_name == "mha" || kernel_name == "mha_flash_attn") {
    std::vector<f16_t> kb(max_seq_len * n_kv_heads * head_dim);
    fill_random(kb.data(), kb.size(), 0);
    std::vector<f16_t> vb(max_seq_len * n_kv_heads * head_dim);
    fill_random(vb.data(), vb.size(), 1);
    std::vector<float> q(n_heads * head_dim);
    fill_random(q.data(), q.size(), 2);
    std::vector<float> att_cuda(n_heads * max_seq_len);
    std::vector<float> xout_cuda(n_heads * head_dim);
    bool use_flash_attention = kernel_name == "mha_flash_attn";
    mha_cuda(
      xout_cuda.data(), 
      att_cuda.data(),
      kb.data(), 
      vb.data(), 
      q.data(), 
      head_dim, kv_len, max_seq_len, n_heads, n_kv_heads,
      use_flash_attention
    );
  } else if (kernel_name == "ffn") {
    std::vector<float> x(dim);
    fill_random(x.data(), x.size(), 0);
    std::vector<f16_t> w1(dim * hidden_dim);
    fill_random(w1.data(), w1.size(), 1, 1.0 / sqrtf(dim)); 
    std::vector<f16_t> w2(hidden_dim * dim);
    fill_random(w2.data(), w2.size(), 2, 1.0 / sqrtf(hidden_dim));
    std::vector<f16_t> w3(dim * hidden_dim);
    fill_random(w3.data(), w3.size(), 3, 1.0 / sqrtf(dim));
    std::vector<float> xout_cuda(dim);
    ffn_cuda<f16_t>(
      xout_cuda.data(), 
      x.data(), 
      w1.data(), 
      w2.data(), 
      w3.data(), 
      hidden_dim, dim, 
      ActivationType::GELU
    );
  } else {
    std::cerr << "Unknown kernel: " << kernel_name << std::endl;
    exit(1);
  }
}

int main(int argc, char* argv[]) {
  if (argc == 2 && std::string(argv[1]) == "-b") {
    std::cout << "Running memory benchmark" << std::endl;
    mem_bench();
  } else if (argc == 2 && std::string(argv[1]) == "-b2") {
    std::cout << "Running memory benchmark 2" << std::endl;
    mem_bench2();
  } else if (argc >= 2 && std::string(argv[1]) == "-bk") {
    if (argc != 3) {
      std::cerr << "Usage: " << argv[0] << " -bk <kernel_name>" << std::endl;
      exit(1);
    }
    std::cout << "Running kernel benchmark" << std::endl;
    for (int i = 0; i < 1; i++) {
      kernel_bench(argv[2]);
    }
  } else {
    test_attn();
    test_cuda_kernels();
  }
  std::cout << "All tests passed" << std::endl;
  return 0;
}