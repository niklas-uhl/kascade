#include <algorithm>
#include <numeric>
#include <utility>

#include <kascade/distribution.hpp>
#include <kascade/pack.hpp>
#include <kascade/types.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kamping/collectives/scatter.hpp>
#include <kamping/communicator.hpp>

TEST(PackTest, works) {
  // NOLINTBEGIN(*-magic-numbers)
  kamping::Communicator<> comm;
  EXPECT_EQ(comm.size(), 4);
  std::vector<kascade::idx_t> succ_array;
  std::vector<kascade::rank_t> rank_array;
  std::vector<kascade::idx_t> active_indices;
  switch (comm.rank()) {
    case 0:
      succ_array = {0, 6, 0, 0, 0, 0, 8, 0};
      rank_array = {42, 1, 42, 42, 42, 42, 1, 42};
      active_indices = {1, 6};
      break;
    case 1:
      succ_array = {10, 0, 14, 0, 0, 0};
      rank_array = {1, 42, 1, 42, 42, 42};
      active_indices = {0, 2};
      break;
    case 2:
      succ_array = {15, 18};
      rank_array = {1, 1};
      active_indices = {0, 1};
      break;
    case 3:
      succ_array = {0, 0, 18, 0};
      rank_array = {42, 42, 0, 42};
      active_indices = {2};
      break;
    default:
      std::unreachable();
  }
  kascade::Distribution dist{succ_array.size(), comm};
  std::vector<kascade::idx_t> packed_succ_array(2, 42);
  std::vector<kascade::rank_t> packed_rank_array(2, 42);

  auto [packed_dist, unpack] = kascade::pack(succ_array, rank_array, dist, active_indices,
                                             packed_succ_array, packed_rank_array, comm);
  switch (comm.rank()) {
    case 0:
      EXPECT_THAT(packed_succ_array, ::testing::ElementsAreArray({1, 2}));
      EXPECT_THAT(packed_rank_array, ::testing::ElementsAreArray({1, 1}));
      break;
    case 1:
      EXPECT_THAT(packed_succ_array, ::testing::ElementsAreArray({3, 4}));
      EXPECT_THAT(packed_rank_array, ::testing::ElementsAreArray({1, 1}));
      break;
    case 2:
      EXPECT_THAT(packed_succ_array, ::testing::ElementsAreArray({5, 6}));
      EXPECT_THAT(packed_rank_array, ::testing::ElementsAreArray({1, 1}));
      break;
    case 3:
      EXPECT_THAT(packed_succ_array, ::testing::ElementsAreArray(
                                         {6, 42}));  // last element remains unchanged
      EXPECT_THAT(packed_rank_array, ::testing::ElementsAreArray({0, 42}));
      break;
    default:
      std::unreachable();
  }

  // simulate the result of list ranking, 6 is the root
  std::ranges::fill(packed_succ_array, 6);
  for (std::size_t i = 0; i < packed_rank_array.size(); ++i) {
    if (packed_rank_array[i] != 42) {
      packed_rank_array[i] = 6 - i - comm.rank() * 2;
    }
  }

  // no let's unpack and check if we get the indices and ranks in the right place
  unpack(packed_succ_array, packed_rank_array, packed_dist, succ_array, rank_array, dist,
         active_indices, comm);

  // all active indices should now point to the root 18
  switch (comm.rank()) {
    case 0:
      EXPECT_THAT(succ_array, ::testing::ElementsAreArray({0, 18, 0, 0, 0, 0, 18, 0}));
      EXPECT_THAT(rank_array,
                  ::testing::ElementsAreArray({42, 6, 42, 42, 42, 42, 5, 42}));
      break;
    case 1:
      EXPECT_THAT(succ_array, ::testing::ElementsAreArray({18, 0, 18, 0, 0, 0}));
      EXPECT_THAT(rank_array, ::testing::ElementsAreArray({4, 42, 3, 42, 42, 42}));
      break;
    case 2:
      EXPECT_THAT(succ_array, ::testing::ElementsAreArray({18, 18}));
      EXPECT_THAT(rank_array, ::testing::ElementsAreArray({2, 1}));
      break;
    case 3:
      EXPECT_THAT(succ_array, ::testing::ElementsAreArray({0, 0, 18, 0}));
      EXPECT_THAT(rank_array, ::testing::ElementsAreArray({42, 42, 0, 42}));
      break;
    default:
      std::unreachable();
  }

  // NOLINTEND(*-magic-numbers)
}

TEST(PackTest, sanity_check) {
  // NOLINTBEGIN(*-magic-numbers)
  kamping::Communicator<> comm;
  EXPECT_EQ(comm.size(), 4);
  // build a simple path of length 12, and scatter it
  std::vector<kascade::idx_t> succ_array;
  std::vector<kascade::rank_t> rank_array;
  if (comm.is_root()) {
    succ_array.resize(12);
    std::ranges::iota(succ_array, 1);
    succ_array.back() = succ_array.size() - 1;
    rank_array.resize(12, 1);
    rank_array.back() = 0;
  }
  succ_array = comm.scatter(kamping::send_buf(std::move(succ_array)));
  rank_array = comm.scatter(kamping::send_buf(std::move(rank_array)));

  kascade::Distribution dist{succ_array.size(), comm};
  std::vector<kascade::idx_t> packed_succ_array(succ_array.size(), 42);
  std::vector<kascade::rank_t> packed_rank_array(rank_array.size(), 42);
  kascade::pack(succ_array, rank_array, dist, dist.local_indices(comm.rank()),
                packed_succ_array, packed_rank_array, comm);
  EXPECT_THAT(packed_succ_array, ::testing::ElementsAreArray(succ_array));
  EXPECT_THAT(packed_rank_array, ::testing::ElementsAreArray(rank_array));
  // NOLINTEND(*-magic-numbers)
}
