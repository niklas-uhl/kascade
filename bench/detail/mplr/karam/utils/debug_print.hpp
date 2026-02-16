#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <kamping/communicator.hpp>

namespace karam::utils {
// TODO wrap in macro to get variable name?
inline std::stringstream
fill_with_rank(std::string const& msg, kamping::Communicator<> const& comm = kamping::comm_world()) {
    std::stringstream sstr;
    int const         num_bits = 1 + static_cast<int>(std::log10(comm.size()));
    sstr << "[" << std::setw(num_bits) << comm.rank() << "] (" << msg << "): ";
    return sstr;
}

template <typename T, typename U>
std::stringstream& operator<<(std::stringstream& out, std::pair<T, U> const& pair) {
    out << "(";
    out << pair.first;
    out << ", ";
    out << pair.second;
    out << ")";
    return out;
}

template <typename... Args>
std::stringstream& operator<<(std::stringstream& out, std::vector<Args...> const& vector) {
    out << "[";
    bool do_print_comma = false;
    for (auto const& elem: vector) {
        if (do_print_comma) {
            out << ", ";
        }
        do_print_comma = true;
        out << elem;
    }
    out << "]";
    return out;
}

template <typename T>
std::stringstream& operator<<(std::stringstream& out, std::span<T> const& span) {
    out << "[";
    bool do_print_comma = false;
    for (auto const& elem: span) {
        if (do_print_comma) {
            out << ", ";
        }
        do_print_comma = true;
        out << elem;
    }
    out << "]";
    return out;
}

template <typename Key, typename Value>
std::stringstream& operator<<(std::stringstream& out, std::unordered_map<Key, Value> const& map) {
    if (map.empty()) {
        out << "{}";
        return out;
    }
    out << "{\n";
    bool do_print_comma = false;
    for (auto const& [key, value]: map) {
        if (do_print_comma) {
            out << ",\n";
        }
        do_print_comma = true;
        // out << key << ":" << value;
        out << "\t" << key << ": ";
        out << value;
    }
    out << "\n}";
    return out;
}



template <typename T>
void debug_print(T const& t, std::string const& msg = "", kamping::Communicator<> const& comm = kamping::comm_world()) {
    auto sstr = fill_with_rank(msg, comm);
    sstr << t;
    std::cout << sstr.str() << std::endl;
}

} // namespace karam::utils
