
#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <kamping/communicator.hpp>

#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/bcast.hpp"

namespace karam {

class StandardKey {
public:
    using Id        = std::string;
    using Iteration = std::uint64_t;
    StandardKey()   = default;
    StandardKey(Id const& id, Iteration const& iteration) : _id{id}, _iteration{iteration} {}
    void append_to_id(std::string const& str) {
        _id.append("_");
        _id.append(str);
    }
    bool operator==(StandardKey const& other_key) const {
        return std::tie(_id, _iteration) == std::tie(other_key._id, other_key._iteration);
    }
    Id const& get_id() const {
        return _id;
    }
    Iteration const& get_iteration() const {
        return _iteration;
    }
    friend std::ostream& operator<<(std::ostream& out, StandardKey const& key) {
        return out << key._id << "-" << key._iteration;
    }
    std::vector<char> serialize() const;

private:
    Id        _id;
    Iteration _iteration;
};
} // namespace karam

namespace std {
template <>
struct hash<karam::StandardKey> {
    std::size_t operator()(karam::StandardKey const& key) const {
        return (
            hash<karam::StandardKey::Id>{}(key.get_id())
            ^ (std::hash<karam::StandardKey::Iteration>{}(key.get_iteration()) << 1) >> 1
        );
    }
};
} // namespace std
namespace karam {
StandardKey              deserialize(char const* buf);
std::vector<char>        serialize(std::vector<StandardKey>& keys);
std::vector<StandardKey> deserialize(std::vector<char>& serialization);
bool                     operator<(StandardKey const& lhs, StandardKey const& rhs);
//
class Timer {
    using PointInTime          = double;
    using TimeIntervalDataType = unsigned long long;
    using Key                  = StandardKey;

    struct Timeintervals {
        TimeIntervalDataType activeTime;
        TimeIntervalDataType totalTime;
        friend std::ostream& operator<<(std::ostream& out, Timeintervals const& intervals) {
            return out << "active time: " << intervals.activeTime << " total time: " << intervals.totalTime;
        }
    };
    struct TimeType {
        TimeIntervalDataType time;
        std::string          type;
    };

    struct DataType {
        std::int64_t data;
        std::string  type;
    };

    struct StartStop {
        PointInTime start;
        PointInTime stop;
    };

public:
    using TimeOutputType      = std::pair<Key, TimeType>;
    using StartStopOutputType = std::tuple<Key, PointInTime, PointInTime>;
    Timer()                   = default;
    void reset();
    void start(const typename Key::Id& key_id, const typename Key::Iteration& iteration);
    void start(const typename Key::Id& key_id);
    void stop(const typename Key::Id& key_id, const typename Key::Iteration& iteration);
    void stop(const typename Key::Id& key_id);

    template <typename OutputIterator>
    void collect(OutputIterator out) const {
        std::vector<Key> local_keys;
        if (_comm.is_root()) {
            for (auto const& [key, ignore]: keyToTime)
                local_keys.push_back(key);
        }
        std::sort(local_keys.begin(), local_keys.end());
        std::vector<char> bytestream_root_keys = serialize(local_keys);
        _comm.bcast(kamping::send_recv_buf(bytestream_root_keys));
        std::vector<Key> root_keys = deserialize(bytestream_root_keys);
        for (auto const& key: root_keys) {
            const TimeType tt{maxTime(key), "maxTime"};
            out = std::make_pair(key, tt);
        }
    }
    std::string output(std::string prefix = "") const;

private:
    bool                    is_measurement_enabled      = true;
    bool                    is_barrier_in_start_enabled = true;
    bool                    is_barrier_in_stop_enabled  = true;
    bool                    is_debug_output_enabled     = false;
    kamping::Communicator<> _comm{MPI_COMM_WORLD, 0};
    const std::uint64_t     RESOLUTION = 1'000'000;

    TimeIntervalDataType maxTime(Key const& key) const {
        auto const itKeyTime                  = keyToTime.find(key);
        bool const is_present                 = itKeyTime != keyToTime.end();
        auto const& [active_time, total_time] = is_present ? itKeyTime->second : Timeintervals{0ull, 0ull};

        std::vector<TimeIntervalDataType> input{active_time};
        return _comm.allreduce(kamping::send_buf(input), kamping::op(kamping::ops::max<>()))
            .extract_recv_buffer()
            .front();
    }
    std::unordered_map<Key, PointInTime>   keyToStart;
    std::unordered_map<Key, PointInTime>   keyToStop;
    std::unordered_map<Key, Timeintervals> keyToTime;
    const std::string                      default_phase = "default_phase";
    std::string                            active_phase  = default_phase;
};

inline std::ostream& operator<<(std::ostream& out, Timer::TimeOutputType const& elem) {
    return out << elem.first << "=" << elem.second.time;
}
inline Timer& get_timer() {
    static Timer timer;
    return timer;
}
} // namespace karam
