// Authors: abagusetty@github and alvarovm@github , Argonne UChicago LLC.
#pragma once

#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include <type_traits>
#include <sycl/sycl.hpp>

// SYCL_DEVICE can be defined at configuration time via CMake (default: gpu)
// Valid values: cpu, gpu, accelerator, all
#ifndef SYCL_DEVICE
#define SYCL_DEVICE all
#endif

// Helper macros to properly expand SYCL_DEVICE before token pasting
#define SYCL_DEVICE_TYPE_CONCAT(x) sycl::info::device_type::x
#define SYCL_DEVICE_TYPE(x) SYCL_DEVICE_TYPE_CONCAT(x)
#define SYCL_DEVICE_TYPE_VALUE SYCL_DEVICE_TYPE(SYCL_DEVICE)

#if defined(_MSC_VER)
#define __global__ __forceinline
#elif defined(__clang__) || defined(__GNUC__)
#define __global__ inline __attribute__((always_inline))
#else
#define __global__ inline
#endif

namespace syclex = sycl::ext::oneapi;

template <typename T1, typename T2> static inline T1 atomicAdd(T1* addr, const T2 val) {
    sycl::atomic_ref<
        T1,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>
        atom(*addr);
    return atom.fetch_add(static_cast<T1>(val));
}
template <typename T1, typename T2> static inline T1 atomicMax(T1* addr, const T2 val) {
    sycl::atomic_ref<
        T1,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>
        atom(*addr);
    return atom.fetch_max(static_cast<T1>(val));
}
template <typename T>
static inline typename std::enable_if<std::is_integral<T>::value, T>::type atomicOr(
    T* addr, const T val
) {
    sycl::atomic_ref<
        T,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>
        atom(addr[0]);
    return atom.fetch_or(val);
}

template <class T> using sycl_device_global = sycl::ext::oneapi::experimental::device_global<T>;

inline auto asyncHandler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (sycl::exception const& e) {
            std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                      << e.what() << std::endl
                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                      << std::endl;
        }
    }
};

class device_ext : public sycl::device {
  public:
    device_ext() : sycl::device() {}
    ~device_ext() { std::lock_guard<std::mutex> lock(m_mutex); }
    device_ext(const sycl::device& base) : sycl::device(base) {}

  private:
    mutable std::mutex m_mutex;
};

using thread_id_t = std::thread::id;

static inline thread_id_t get_tid() { return std::this_thread::get_id(); }

class dev_mgr {
  public:
    int current_device() {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto tid = get_tid();
        auto it = _thread2dev_map.find(tid);
        if (it != _thread2dev_map.end()) {
            check_id(it->second);
            return it->second;
        }
        // Insert default device if not present
        _thread2dev_map[tid] = DEFAULT_DEVICE_ID;
        return DEFAULT_DEVICE_ID;
    }
    sycl::queue* current_queue() { return _queues[current_device()]; }
    sycl::queue* select_queue(int id) {
        std::lock_guard<std::mutex> lock(m_mutex);
        check_id(id);
        return _queues[id];
    }
    void select_device(int id) {
        std::lock_guard<std::mutex> lock(m_mutex);
        check_id(id);
        _thread2dev_map[get_tid()] = id;
    }
    int device_count() { return _queues.size(); }

    /// Returns the instance of device manager singleton.
    static dev_mgr& instance() {
        static dev_mgr d_m{};
        return d_m;
    }
    dev_mgr(const dev_mgr&) = delete;
    dev_mgr& operator=(const dev_mgr&) = delete;
    dev_mgr(dev_mgr&&) = delete;
    dev_mgr& operator=(dev_mgr&&) = delete;

  private:
    mutable std::mutex m_mutex;

    dev_mgr() {
        auto devices = sycl::device::get_devices(SYCL_DEVICE_TYPE_VALUE);
        if (devices.empty()) {
            throw std::runtime_error("No SYCL GPU devices found.");
        }

        for (const auto& dev : devices) {
            auto* q = new sycl::queue(
                dev, asyncHandler, sycl::property_list{sycl::property::queue::in_order{}}
            );
            _queues.push_back(q);
        }
    }

    void check_id(int id) const {
        if (id >= _queues.size()) {
            throw std::runtime_error("Invalid device id");
        }
    }

    std::vector<sycl::queue*> _queues;

    /// DEFAULT_DEVICE_ID is used, if current_device() can not find current
    /// thread id in _thread2dev_map, which means default device should be used
    /// for the current thread.
    const int DEFAULT_DEVICE_ID = 0;
    /// thread-id to device-id map.
    std::map<thread_id_t, int> _thread2dev_map;
};

/// Util function to get the current device (in int).
static inline void syclGetDevice(int* id) { *id = dev_mgr::instance().current_device(); }

/// Util function to get the current queue
static inline sycl::queue* sycl_get_queue() { return dev_mgr::instance().current_queue(); }
/// Util function to get queue from device`id`
static inline sycl::queue* sycl_get_queue_nth(int device_id) {
    return dev_mgr::instance().select_queue(device_id);
}

/// Util function to set a device by id. (to _thread2dev_map)
static inline void syclSetDevice(int id) { dev_mgr::instance().select_device(id); }

/// Util function to get number of GPU devices (default: explicit scaling)
static inline void syclGetDeviceCount(int* id) { *id = dev_mgr::instance().device_count(); }
