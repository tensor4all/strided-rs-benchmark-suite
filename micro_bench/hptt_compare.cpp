#include <hptt.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

int env_int(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  return std::stoi(value);
}

std::size_t product(const std::vector<int>& dims) {
  std::size_t total = 1;
  for (int dim : dims) {
    total *= static_cast<std::size_t>(dim);
  }
  return total;
}

std::vector<std::size_t> col_major_strides(const std::vector<int>& dims) {
  std::vector<std::size_t> strides(dims.size(), 1);
  for (std::size_t i = 1; i < dims.size(); ++i) {
    strides[i] = strides[i - 1] * static_cast<std::size_t>(dims[i - 1]);
  }
  return strides;
}

std::vector<int> permuted_dims(const std::vector<int>& dims, const std::vector<int>& perm) {
  std::vector<int> out(dims.size());
  for (std::size_t i = 0; i < dims.size(); ++i) {
    out[i] = dims[static_cast<std::size_t>(perm[i])];
  }
  return out;
}

std::size_t permuted_destination_offset(
    std::size_t src_flat,
    const std::vector<int>& dims,
    const std::vector<int>& perm) {
  const auto src_strides = col_major_strides(dims);
  const auto out_dims = permuted_dims(dims, perm);
  const auto dst_strides = col_major_strides(out_dims);
  std::vector<int> src_idx(dims.size(), 0);
  std::vector<int> dst_idx(dims.size(), 0);

  for (std::size_t d = dims.size(); d-- > 0;) {
    src_idx[d] = static_cast<int>(src_flat / src_strides[d]);
    src_flat %= src_strides[d];
  }
  for (std::size_t d = 0; d < dims.size(); ++d) {
    dst_idx[d] = src_idx[static_cast<std::size_t>(perm[d])];
  }

  std::size_t dst_flat = 0;
  for (std::size_t d = 0; d < dims.size(); ++d) {
    dst_flat += static_cast<std::size_t>(dst_idx[d]) * dst_strides[d];
  }
  return dst_flat;
}

std::vector<double> make_input(std::size_t total) {
  std::vector<double> data(total);
  for (std::size_t i = 0; i < total; ++i) {
    data[i] = static_cast<double>((i * 1315423911ULL) & 0xffff) * 0.25 + 1.0;
  }
  return data;
}

double median_ms(std::vector<double>& samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

double iqr_ms(const std::vector<double>& sorted_samples) {
  return sorted_samples[3 * sorted_samples.size() / 4] - sorted_samples[sorted_samples.size() / 4];
}

void bench(const std::string& label, std::size_t bytes, int warmup, int nruns, const std::function<void()>& f) {
  for (int i = 0; i < warmup; ++i) {
    f();
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(nruns));
  for (int i = 0; i < nruns; ++i) {
    auto t0 = Clock::now();
    f();
    auto t1 = Clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  std::sort(samples.begin(), samples.end());
  const double med = median_ms(samples);
  const double iqr = iqr_ms(samples);
  const double gbps = static_cast<double>(bytes) / (med / 1000.0) / 1.0e9;
  std::cout << "  " << label;
  for (std::size_t i = label.size(); i < 28; ++i) {
    std::cout << ' ';
  }
  std::cout << med << " ms (IQR " << iqr << ")  " << gbps << " GB/s\n";
}

void verify_case(
    const std::string& name,
    const std::vector<double>& src,
    const std::vector<double>& dst,
    const std::vector<int>& dims,
    const std::vector<int>& perm) {
  const std::size_t total = product(dims);
  const std::vector<std::size_t> samples = {
      0,
      total > 1 ? std::size_t{1} : std::size_t{0},
      total / 3,
      total / 2,
      total > 0 ? total - 1 : 0,
  };
  for (std::size_t src_flat : samples) {
    const std::size_t dst_flat = permuted_destination_offset(src_flat, dims, perm);
    if (dst[dst_flat] != src[src_flat]) {
      throw std::runtime_error("verification failed for " + name);
    }
  }
}

void run_case(
    const std::string& name,
    const std::vector<int>& dims,
    const std::vector<int>& perm,
    int threads,
    int warmup,
    int nruns) {
  const std::size_t total = product(dims);
  const std::size_t bytes = total * sizeof(double) * 2;
  auto src = make_input(total);
  std::vector<double> dst(total, 0.0);

  auto plan = hptt::create_plan(
      perm.data(),
      static_cast<int>(dims.size()),
      1.0,
      src.data(),
      dims.data(),
      nullptr,
      0.0,
      dst.data(),
      nullptr,
      hptt::ESTIMATE,
      threads);

  plan->execute();
  if (dims.size() <= 8) {
    verify_case(name, src, dst, dims, perm);
  }

  std::cout << "=== " << name << " ===\n";
  bench("hptt execute", bytes, warmup, nruns, [&]() {
    plan->execute();
  });

  bench("hptt create+execute", bytes, warmup, nruns, [&]() {
    auto local_plan = hptt::create_plan(
        perm.data(),
        static_cast<int>(dims.size()),
        1.0,
        src.data(),
        dims.data(),
        nullptr,
        0.0,
        dst.data(),
        nullptr,
        hptt::ESTIMATE,
        threads);
    local_plan->execute();
  });
  std::cout << '\n';
}

}  // namespace

int main() {
  const int threads = env_int("THREADS", 1);
  const int warmup = env_int("WARMUP", 3);
  const int nruns = env_int("NRUNS", 11);

  std::cout << "HPTT comparison benchmark\n";
  std::cout << "threads=" << threads << " warmup=" << warmup << " nruns=" << nruns << "\n";
  std::cout << "columns: label median_ms(iqr) bandwidth_GB/s\n\n";

  run_case(
      "2D 1024^2 transpose [1,0]",
      {1024, 1024},
      {1, 0},
      threads,
      warmup,
      nruns);

  run_case(
      "3D 256^3 transpose [2,0,1]",
      {256, 256, 256},
      {2, 0, 1},
      threads,
      warmup,
      nruns);

  run_case(
      "3D 256^3 transpose [1,0,2]",
      {256, 256, 256},
      {1, 0, 2},
      threads,
      warmup,
      nruns);
}
