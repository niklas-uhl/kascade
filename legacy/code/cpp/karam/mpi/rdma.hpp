#pragma once

#include <optional>
#include <span>

#include <kamping/communicator.hpp>
#include <kamping/mpi_datatype.hpp>
#include <kamping/mpi_ops.hpp>
#include <kassert/kassert.hpp>
#include <mpi.h>

#include "karam/mpi/atomic_ops.hpp"
#include "karam/mpi/datatype.hpp"
#include "karam/utils/non_copyable.hpp"

namespace karam::mpi {

template <MPIDatatype T>
class Win;

template <MPIDatatype T>
class WindowObject : private karam::utils::NonCopyable {
public:
    WindowObject(int rank, Win<T> const& win) : _rank(rank), _win(win) {}
    int rank() const {
        return _rank;
    }

    Win<T> const& win() const {
        return _win;
    }

    bool operations_allowed() const {
        return true;
    }

private:
    int           _rank;
    Win<T> const& _win;
};

template <MPIDatatype T, typename WindowObjectBase>
class WindowLock {
public:
    friend WindowObjectBase;
    WindowLock(WindowLock&& other) : _owns_lock(other._owns_lock) {
        other._owns_lock = false;
    }

    WindowLock& operator=(WindowLock&& other) {
        WindowLock tmp(other);
        std::swap(tmp._owns_lock, this->_owns_lock);
        return this;
    }

    void lock(bool exclusive = false, bool no_conflicts = true) {
        KASSERT(_owns_lock);
        int lock_type = MPI_LOCK_SHARED;
        if (exclusive) {
            lock_type = MPI_LOCK_EXCLUSIVE;
        }
        int assert = 0;
        if (no_conflicts) {
            assert = MPI_MODE_NOCHECK;
        }
        MPI_Win_lock(lock_type, get_win().rank(), assert, get_win().win().native_win());
    }

    void unlock() {
        KASSERT(_owns_lock);
        MPI_Win_unlock(get_win().rank(), get_win().win().native_win());
        _owns_lock = false;
    }

    virtual ~WindowLock() {
        if (_owns_lock) {
            unlock();
        }
    }

private:
    WindowObjectBase& get_win() {
        return static_cast<WindowObjectBase&>(*this);
    }
    WindowLock(bool exclusive = false, bool no_conflicts = true) : _owns_lock(true) {
        lock(exclusive, no_conflicts);
    }
    bool operations_allowed() const {
        return _owns_lock;
    }
    bool _owns_lock;
};

template <MPIDatatype T, typename WindowObjectBase>
class WindowFlush {
public:
    void flush() {
        KASSERT(get_win().operations_allowed());
        MPI_Win_flush(get_win().rank(), get_win().win().native_win());
    }

    void flush_local() {
        KASSERT(get_win().operations_allowed());
        MPI_Win_flush_local(get_win().rank(), get_win().win().native_win());
    }

private:
    WindowObjectBase& get_win() {
        return static_cast<WindowObjectBase&>(*this);
    }
};

template <MPIDatatype T, typename WindowObjectBase>
class RemoteOperations {
private:
    WindowObjectBase& get_win() {
        return static_cast<WindowObjectBase&>(*this);
    }

public:
    void get(std::size_t index, T& value) {
        KASSERT(get_win().operations_allowed());
        MPI_Datatype datatype = karam::mpi::datatype<T>::get_type();
        MPI_Get(
            &value,
            1,
            datatype,
            get_win().rank(),
            static_cast<MPI_Aint>(index),
            1,
            datatype,
            get_win().win().native_win()
        );
    }

    void get(std::size_t index, MPIBuffer<T> auto& values) {
        KASSERT(get_win().operations_allowed());
        MPI_Datatype datatype = karam::mpi::datatype<T>::get_type();
        MPI_Get(
            std::ranges::data(values),
            static_cast<int>(std::ranges::size(values)),
            datatype,
            get_win().rank(),
            static_cast<int>(index),
            static_cast<int>(std::ranges::size(values)),
            datatype,
            get_win().win().native_win()
        );
    }

    void put(std::size_t index, T const& value) {
        KASSERT(get_win().operations_allowed());
        MPI_Datatype datatype = karam::mpi::datatype<T>::get_type();
        MPI_Put(
            &value,
            1,
            datatype,
            get_win().rank(),
            static_cast<int>(index),
            1,
            datatype,
            get_win().win().native_win()
        );
    }

    void put(std::size_t index, MPIBuffer<T> auto const& values) {
        KASSERT(get_win().operations_allowed());
        MPI_Datatype datatype = karam::mpi::datatype<T>::get_type();
        MPI_Put(
            std::ranges::data(values),
            static_cast<int>(std::ranges::size(values)),
            datatype,
            get_win().rank(),
            static_cast<int>(index),
            static_cast<int>(std::ranges::size(values)),
            datatype,
            get_win().win().native_win()
        );
    }

    template <ops::AtomicOp Op>
    void fetch_and_op(std::size_t index, T const& value, T& old_value, Op) {
        KASSERT(get_win().operations_allowed());
        MPI_Op       op       = Op::mpi_op();
        MPI_Datatype datatype = karam::mpi::datatype<T>::get_type();
        MPI_Fetch_and_op(
            &value,                      // origin_addr
            &old_value,                  // result_addr
            datatype,                    // datatype
            get_win().rank(),            // target_rank
            static_cast<int>(index),     // target_disp
            op,                          // op
            get_win().win().native_win() // win
        );
    }

    template <ops::AtomicOp Op>
    void accumulate(std::size_t index, T const& value, Op) {
        KASSERT(get_win().operations_allowed());
        MPI_Op       op       = Op::mpi_op();
        MPI_Datatype datatype = karam::mpi::datatype<T>::get_type();
        MPI_Accumulate(
            &value,                      // origin_addr
            1,                           // origin_count
            datatype,                    // origin_datatype
            get_win().rank(),            // target_rank
            static_cast<int>(index),     // target_disp
            1,                           // target_count
            datatype,                    // target_datatype
            op,                          // op
            get_win().win().native_win() // win
        );
    }

    template <ops::AtomicOp Op>
    void accumulate(std::size_t index, MPIBuffer<T> auto const& values, Op) {
        KASSERT(get_win().operations_allowed());
        MPI_Op       op       = Op::mpi_op();
        MPI_Datatype datatype = karam::mpi::datatype<T>::get_type();
        MPI_Accumulate(
            std::ranges::data(values),                   // origin_addr
            static_cast<int>(std::ranges::size(values)), // origin_count
            datatype,                                    // origin_datatype
            get_win().rank(),                            // target_rank
            static_cast<int>(index),                     // target_disp
            static_cast<int>(std::ranges::size(values)), // target_count
            datatype,                                    // target_datatype
            op,                                          // op
            get_win().win().native_win()                 // win
        );
    }

    template <ops::AtomicOp Op>
    void get_accumulate(std::size_t index, MPIBuffer<T> auto const& values, MPIBuffer<T> auto& old_values, Op) {
        KASSERT(get_win().operations_allowed());
        KASSERT(values.size() == old_values.size());
        MPI_Op       op       = Op::mpi_op();
        MPI_Datatype datatype = karam::mpi::datatype<T>::get_type();
        MPI_Get_Accumulate(
            std::ranges::data(values),                       // origin_addr
            static_cast<int>(std::ranges::size(values)),     // origin_count
            datatype,                                        // origin_datatype
            std::ranges::data(old_values),                   // result_addr
            static_cast<int>(std::ranges::size(old_values)), // result_count
            datatype,                                        // result_datatype
            get_win().rank(),                                // target_rank
            static_cast<int>(index),                         // target_disp
            static_cast<int>(std::ranges::size(values)),     // target_count
            datatype,                                        // target_datatype
            op,                                              // op
            get_win().win().native_win()                     // win
        );
    }
};

template <typename T, typename WindowObjectBase>
class LocalOperations {
private:
    WindowObjectBase& get_win() {
        return static_cast<WindowObjectBase&>(*this);
    }
    WindowObjectBase const& get_win() const {
        return static_cast<WindowObjectBase const&>(*this);
    }

public:
    T& operator[](std::size_t index) {
        KASSERT(get_win().operations_allowed());
        return get_win().win()._local_data[index];
    }

    T const& operator[](std::size_t index) const {
        KASSERT(get_win().operations_allowed());
        return get_win().win()._local_data[index];
    }

    std::size_t size() const {
        return get_win().win()._local_data.size();
    }
    typename std::span<T>::iterator begin() const {
        KASSERT(get_win().operations_allowed());
        return get_win().win()._local_data.begin();
    }
    typename std::span<T>::iterator end() const {
        KASSERT(get_win().operations_allowed());
        return get_win().win()._local_data.end();
    }
};

template <MPIDatatype T>
class RemotePassiveTargetLock : public WindowObject<T>,
                                public WindowLock<T, RemotePassiveTargetLock<T>>,
                                public WindowFlush<T, RemotePassiveTargetLock<T>>,
                                public RemoteOperations<T, RemotePassiveTargetLock<T>> {
    friend class WindowLock<T, RemotePassiveTargetLock<T>>;
    friend class RemoteOperations<T, RemotePassiveTargetLock<T>>;
    friend class WindowFlush<T, RemotePassiveTargetLock<T>>;
    friend class Win<T>;

private:
    bool operations_allowed() const {
        return WindowLock<T, RemotePassiveTargetLock<T>>::operations_allowed();
    }
    RemotePassiveTargetLock(int rank, Win<T> const& win, bool exclusive = false, bool no_conflicts = true)
        : WindowObject<T>(rank, win),
          WindowLock<T, RemotePassiveTargetLock<T>>(exclusive, no_conflicts) {}
};

template <MPIDatatype T>
class LocalPassiveTargetLock : public WindowObject<T>,
                               public WindowLock<T, LocalPassiveTargetLock<T>>,
                               public LocalOperations<T, LocalPassiveTargetLock<T>> {
    friend class WindowLock<T, LocalPassiveTargetLock<T>>;
    friend class LocalOperations<T, LocalPassiveTargetLock<T>>;
    friend class Win<T>;

private:
    bool operations_allowed() const {
        return WindowLock<T, LocalPassiveTargetLock<T>>::operations_allowed();
    }
    LocalPassiveTargetLock(Win<T> const& win, bool exclusive = false, bool no_conflicts = true)
        : WindowObject<T>(win.comm().rank_signed(), win),
          WindowLock<T, LocalPassiveTargetLock<T>>(exclusive, no_conflicts) {}
};

template <MPIDatatype T>
class LockAll;

template <MPIDatatype T>
class LockAllRemote : public WindowObject<T>,
                      public WindowFlush<T, LockAllRemote<T>>,
                      public RemoteOperations<T, LockAllRemote<T>> {
    friend class RemoteOperations<T, LockAllRemote<T>>;
    friend class WindowFlush<T, LockAllRemote<T>>;
    friend class Win<T>;
    friend class LockAll<T>;

private:
    bool operations_allowed() const {
        return _lock_all.owns_lock();
    }

    LockAllRemote(int rank, LockAll<T> const& lock_all) : WindowObject<T>(rank, lock_all._win), _lock_all(lock_all) {}
    LockAll<T> const& _lock_all;
};

template <MPIDatatype T>
class LockAllLocal : public WindowObject<T>, public LocalOperations<T, LockAllLocal<T>> {
    friend class LocalOperations<T, LockAllLocal<T>>;
    friend class Win<T>;
    friend class LockAll<T>;

private:
    bool operations_allowed() const {
        return _lock_all.owns_lock();
    }

    LockAllLocal(LockAll<T> const& lock_all)
        : WindowObject<T>(lock_all._win.comm().rank_signed(), lock_all._win),
          _lock_all(lock_all) {}
    LockAll<T> const& _lock_all;
};

template <MPIDatatype T>
class LockAll : private karam::utils::NonCopyable {
public:
    friend class LockAllRemote<T>;
    friend class LockAllLocal<T>;

    LockAll(Win<T>& win, bool no_conflicts = true) : _win(win), _owns_lock(true) {
        lock_all(no_conflicts);
    }

    void lock_all(bool no_conflicts = true) {
        KASSERT(_owns_lock);
        int assert = 0;
        if (no_conflicts) {
            assert = MPI_MODE_NOCHECK;
        }
        MPI_Win_lock_all(assert, _win.native_win());
    }

    void unlock_all() {
        KASSERT(_owns_lock);
        MPI_Win_unlock_all(_win.native_win());
        _owns_lock = false;
    }

    LockAll(LockAll&& other) : _win(other._win), _owns_lock(other._owns_lock) {
        other._owns_lock = false;
    }

    LockAll& operator=(LockAll&& other) {
        auto tmp(std::move(other));
        std::swap(tmp._win, this->_win);
        std::swap(tmp._owns_lock, this->_owns_lock);
        return *this;
    }

    void lock() {
        MPI_Win_lock_all(MPI_MODE_NOCHECK, _win.native_win());
        _owns_lock = true;
    }

    void unlock() {
        KASSERT(_owns_lock);
        MPI_Win_unlock_all(_win.native_win());
        _owns_lock = false;
    }

    ~LockAll() {
        if (_owns_lock) {
            unlock_all();
        }
    }

    LockAllRemote<T> on(int rank) {
        return LockAllRemote<T>(rank, *this);
    }

    LockAllLocal<T> local() {
        return LockAllLocal<T>(*this);
    }

private:
    bool owns_lock() const {
        return _owns_lock;
    }

    Win<T>& _win;
    bool    _owns_lock;
};
template <MPIDatatype T>
class AccessEpoch;

template <MPIDatatype T>
class EpochRemote : public WindowObject<T>,
                    public WindowFlush<T, EpochRemote<T>>,
                    public RemoteOperations<T, EpochRemote<T>> {
    friend class RemoteOperations<T, EpochRemote<T>>;
    friend class WindowFlush<T, EpochRemote<T>>;
    friend class AccessEpoch<T>;

private:
    EpochRemote(int rank, AccessEpoch<T> const& epoch) : WindowObject<T>(rank, epoch._win) {}
};

template <MPIDatatype T>
class EpochLocal : public WindowObject<T>, public LocalOperations<T, EpochLocal<T>> {
    friend class LocalOperations<T, EpochLocal<T>>;
    friend class AccessEpoch<T>;

private:
    EpochLocal(AccessEpoch<T> const& epoch) : WindowObject<T>(epoch._win.comm().rank_signed(), epoch._win) {}
};

template <MPIDatatype T>
class AccessEpoch : utils::NonCopyable {
    friend class EpochRemote<T>;
    friend class EpochLocal<T>;

public:
    AccessEpoch(AccessEpoch&& other) : _win(other._win) {}

    AccessEpoch& operator=(AccessEpoch&& other) {
        this->_win = std::move(other._win);
        return *this;
    }

    AccessEpoch(Win<T>& win) : _win(win) {
        fence();
    }

    AccessEpoch<T> fence() {
        KASSERT(_win.native_win() != MPI_WIN_NULL);
        MPI_Win_fence(0, _win.native_win());
        return std::move(*this);
    }

    EpochRemote<T> on(int rank) {
        return EpochRemote<T>(rank, *this);
    }

    EpochLocal<T> local() {
        return EpochLocal<T>(*this);
    }

private:
    Win<T>& _win;
};

template <MPIDatatype T>
class Win : private karam::utils::NonCopyable {
public:
    friend class LocalOperations<T, LocalPassiveTargetLock<T>>;
    friend class LocalOperations<T, LockAllLocal<T>>;
    friend class LocalOperations<T, EpochLocal<T>>;
    Win(std::size_t                    size,
        bool                           same_size_on_all_rank = false,
        kamping::Communicator<> const& comm                  = kamping::comm_world())
        : _comm(&comm),
          _win(MPI_WIN_NULL),
          _local_data() {
        auto     disp          = sizeof(T);
        auto     size_in_bytes = disp * size;
        T*       base_ptr;
        MPI_Info info;
        MPI_Info_create(&info);
        if (same_size_on_all_rank) {
            MPI_Info_set(info, "same_size", "true");
        }
        MPI_Info_set(info, "same_disp_unit", "true");
        MPI_Win_allocate(
            static_cast<MPI_Aint>(size_in_bytes),
            static_cast<int>(disp),
            info,
            comm.mpi_communicator(),
            &base_ptr,
            &_win
        );
        MPI_Info_free(&info);
        _local_data = std::span{base_ptr, size};
    }

    virtual ~Win() {
        if (_win != MPI_WIN_NULL) {
            MPI_Win_free(&_win);
        }
    }

    Win(Win<T>&& other) : _comm(other._comm), _win(std::move(other._win)), _local_data(std::move(other._local_data)) {
        other._win = MPI_WIN_NULL;
    }

    Win<T>& operator=(Win<T>&& other) {
        Win tmp(std::move(other));
        std::swap(tmp._comm, this->_comm);
        std::swap(tmp._win, this->_win);
        std::swap(tmp._local_data, this->_local_data);
        return *this;
    }

    MPI_Win& native_win() {
        return _win;
    }

    MPI_Win const& native_win() const {
        return _win;
    }

    kamping::Communicator<> const& comm() const {
        return *_comm;
    }

    RemotePassiveTargetLock<T> lock(int rank, bool exclusive = false, bool no_conflicts = true) {
        KASSERT(_win != MPI_WIN_NULL, "Trying to access an invalid window.");
        return RemotePassiveTargetLock<T>(rank, *this, exclusive, no_conflicts);
    }

    LocalPassiveTargetLock<T> lock_local(bool exclusive = false, bool no_conflicts = true) {
        KASSERT(_win != MPI_WIN_NULL, "Trying to access an invalid window.");
        return LocalPassiveTargetLock(*this, exclusive, no_conflicts);
    }

    LockAll<T> lock_all(bool no_conflicts = true) {
        return LockAll(*this, no_conflicts);
    }

    AccessEpoch<T> fence() {
        return AccessEpoch<T>(*this);
    }

    std::span<T> local_data() {
        return _local_data;
    }

private:
    kamping::Communicator<> const* _comm;
    MPI_Win                        _win;
    std::span<T>                   _local_data;
};

} // namespace karam::mpi
