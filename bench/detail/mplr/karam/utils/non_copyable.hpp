#pragma once

/// @brief Base object for classes which deletes copy constructor and assignment operator and enables move.
///
/// You can inherit from this class privately.
/// While  constructors are never inherited, the derived class still has no copy constructor (assignment), because it
/// can not be default constructed, due to the missing implementation in the base class. Because we provide a (default)
/// implementation for the move constructor (assignment) in the base class, the derived class can construct default
/// implementations.
namespace karam::utils {

class NonCopyable {
protected:
    constexpr NonCopyable() = default;
    ~NonCopyable()          = default;

    /// @brief Copy constructor is deleted as buffers should only be moved.
    NonCopyable(NonCopyable const&) = delete;
    /// @brief Copy assignment operator is deleted as buffers should only be moved.
    NonCopyable& operator=(NonCopyable const&) = delete;
    /// @brief Move constructor.
    NonCopyable(NonCopyable&&) = default;
    /// @brief Move assignment operator.
    NonCopyable& operator=(NonCopyable&&) = default;
};
} // namespace karam::utils
