#pragma once
#include <stdint.h>
#include <cstddef>

// HAL (Hardware Abstraction Layer) for RTOS primitives
namespace hal {

    using TaskFn = void(*)(void*);

    struct Task { void* p = nullptr; };
    struct Event { void* p = nullptr; };

    void schedulerStart(); // Only needed for FreeRTOS

    Task  taskCreate(TaskFn fn, void* arg, const char* name, uint32_t stackBytes, uint32_t prio);
    void  taskYield();
    void  sleepMs(uint32_t ms);

    Event eventCreate(bool initial = false);
    void  eventWait(Event e, uint32_t timeoutMs);      // 0xFFFFFFFF -> forever
    void  eventGiveFromISR(Event e);                   // ISR-safe

    // Blocking queue interface
    template<typename T, size_t Capacity>
    class Queue {
    public:
        Queue();
        ~Queue();
        
        // Returns true if pushed, false if full
        bool tryPush(const T& item);
        
        // Blocks until item available or timeout
        // Returns true and copies item to out, or false on timeout
        bool pop(T& out, uint32_t timeoutMs);
        
        size_t size() const;
        
    private:
        void* impl;
    };

}

// Platform-specific template implementations
#ifdef ARDUINO_ARCH_MBED
#include <mbed.h>
#include <rtos.h>

namespace hal {
    template<typename T, size_t Capacity>
    struct QueueImpl {
        rtos::Mail<T, Capacity> mail;
    };

    template<typename T, size_t Capacity>
    Queue<T, Capacity>::Queue() {
        impl = new QueueImpl<T, Capacity>();
    }

    template<typename T, size_t Capacity>
    Queue<T, Capacity>::~Queue() {
        delete static_cast<QueueImpl<T, Capacity>*>(impl);
    }

    template<typename T, size_t Capacity>
    bool Queue<T, Capacity>::tryPush(const T& item) {
        auto* q = static_cast<QueueImpl<T, Capacity>*>(impl);
        T* slot = q->mail.try_alloc();
        if (!slot) {
            return false;  // Queue full
        }
        *slot = item;
        q->mail.put(slot);
        return true;
    }

    template<typename T, size_t Capacity>
    bool Queue<T, Capacity>::pop(T& out, uint32_t timeoutMs) {
        auto* q = static_cast<QueueImpl<T, Capacity>*>(impl);
        T* ptr = nullptr;
        // Use modern Mbed OS 6.x try_get_for API with proper duration types
        if (timeoutMs == 0xFFFFFFFF) {
            ptr = q->mail.try_get_for(rtos::Kernel::wait_for_u32_forever);
        } else {
            ptr = q->mail.try_get_for(rtos::Kernel::Clock::duration_u32(timeoutMs));
        }
        if (ptr != nullptr) {
            out = *ptr;
            q->mail.free(ptr);
            return true;
        }
        return false;
    }

    template<typename T, size_t Capacity>
    size_t Queue<T, Capacity>::size() const {
        // Mail doesn't have a count method, return 0
        return 0;
    }
}
#endif
