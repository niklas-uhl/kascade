#pragma once

#include "communicator.hpp"
#include "kamping/collectives/allreduce.hpp"


inline void calculate_send_displacements_and_reset_num_packets_per_PE(
    std::vector<std::int32_t>& send_displacements,
    std::vector<std::int32_t>& num_packets_per_PE,
    kamping::Communicator<>& comm) {
  send_displacements[0] = 0;
  for (std::int32_t i = 1; i < comm.size() + 1; i++)
    send_displacements[i] = send_displacements[i - 1] + num_packets_per_PE[i - 1];
  std::fill(num_packets_per_PE.begin(), num_packets_per_PE.end(), 0);
}

inline bool is_pow_of_two(std::uint32_t p) {
  return std::lround(std::pow(2, std::lround(std::log2(p)))) == p;
}

inline bool any_PE_has_work(kamping::Communicator<>& comm, timer& timer, bool this_PE_has_work) {
  timer.switch_category("communication");
  std::vector<int> recv = comm.allreduce(kamping::send_buf((int)this_PE_has_work),
                                         kamping::op(kamping::ops::plus<>()));
  timer.switch_category("local_work");

  return recv[0] > 0;
}

inline bool any_PE_has_work(kamping::Communicator<>& comm,
                     karam::mpi::GridCommunicator& grid_comm,
                     timer& timer,
                     bool this_PE_has_work,
                     bool grid) {
  std::vector<std::int32_t> work_vec =
      allgatherv(timer, (std::int32_t)this_PE_has_work, comm, grid_comm, grid);

  std::int32_t size = comm.size();
  std::int32_t work = 0;
  for (std::int32_t i = 0; i < size; i++)
    work += work_vec[i];
  return work > 0;
}

inline uint64_t get_time() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// input > 1

inline double lambertw(double input, int iterations) {
  double lower_bound = 1;
  double upper_bound = input;

  for (int i = 0; i < iterations; i++) {
    double middle = (lower_bound + upper_bound) / 2;
    double e = 2.718281828459045235;
    double f_of_middle = middle * std::pow(e, middle);

    if (f_of_middle > input) {
      upper_bound = middle;
    } else {
      lower_bound = middle;
    }
  }

  return lower_bound;
}

inline double lambertw(double input) {
  return lambertw(input, 30);
}

// sagen wir mal, die obersten 4 bit stehen zum markieren frei

inline std::uint64_t unmask(std::uint64_t value) {
  return value & 0xfffffffffffffff;
}

// the nth most significant bit will be marked, n>= 0
inline std::uint64_t mark(std::uint64_t index, int n) {
  return index | (((std::uint64_t)0x8000000000000000) >> n);
}

inline std::uint64_t unmark(std::uint64_t index, int n) {
  return index & (0xffffffffffffffff & (~(((std::uint64_t)0x8000000000000000) >> n)));
}

inline bool is_marked(std::uint64_t index, int n) {
  return (index & (((std::uint64_t)0x8000000000000000) >> n)) != 0;
}

inline double calculate_runtime_pointer_doubling(std::uint64_t n, std::uint32_t p) {
  return 2 * std::log2(n) *
         (4.12 * std::pow(10, -7) * n / p + 1.88 * std::pow(10, -6) * p);
}

inline double calculate_runtime_ruling_set(std::uint64_t n,
                                    std::uint32_t p,
                                    std::uint32_t dist_rulers) {
  double n_rec = n / dist_rulers * std::log(dist_rulers);
  double t_rec = calculate_runtime_pointer_doubling(n_rec, p);
  return 5.81 * std::pow(10, -7) * n / p +
         2.06 * std::pow(10, -6) * p * (dist_rulers + 5) + t_rec;
}

inline double calculate_runtime_ruling_set(std::uint64_t n,
                                    std::uint32_t p,
                                    std::uint32_t dist_rulers,
                                    std::uint32_t dist_rulers_rec) {
  double n_rec = n / dist_rulers * std::log(dist_rulers);
  double t_rec = calculate_runtime_ruling_set(n_rec, p, dist_rulers_rec);
  return 5.81 * std::pow(10, -7) * n / p +
         2.06 * std::pow(10, -6) * p * (dist_rulers + 5) + t_rec;
}

inline double calculate_runtime_ruling_set(std::uint64_t n,
                                    std::uint32_t p,
                                    std::uint32_t dist_rulers,
                                    std::uint32_t dist_rulers_rec,
                                    std::uint32_t dist_rulers_rec_rec) {
  double n_rec = n / dist_rulers * std::log(dist_rulers);
  double t_rec =
      calculate_runtime_ruling_set(n_rec, p, dist_rulers_rec, dist_rulers_rec_rec);
  return 5.81 * std::pow(10, -7) * n / p +
         2.06 * std::pow(10, -6) * p * (dist_rulers + 5) + t_rec;
}
