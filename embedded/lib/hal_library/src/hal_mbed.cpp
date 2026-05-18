#ifdef ARDUINO_ARCH_MBED
// Include Arduino.h first to avoid macro conflicts
#include <Arduino.h>
// Undefine Arduino macros that conflict with C++ standard library
#undef abs
#undef min
#undef max

#include "hal.hpp"
#include <mbed.h>

namespace hal {
    struct ThreadWrap {
        rtos::Thread* th;
        TaskFn fn;
        void* arg;
    };
    static void tramp(ThreadWrap* w){
        w->fn(w->arg);
        delete w->th;
        delete w;
    }

    void schedulerStart(){ /* Not needed for mbed */ }

    Task taskCreate(TaskFn fn, void* arg, const char* name, uint32_t stackBytes, uint32_t prio){
        auto *w = new ThreadWrap{nullptr, fn, arg};

        // Map FreeRTOS-style priorities (0-5) to mbed osPriority
        osPriority mbedPrio;
        switch(prio) {
            case 0: mbedPrio = osPriorityIdle; break;
            case 1: mbedPrio = osPriorityLow; break;
            case 2: mbedPrio = osPriorityNormal; break;
            case 3: mbedPrio = osPriorityAboveNormal; break;
            case 4: mbedPrio = osPriorityHigh; break;
            case 5: mbedPrio = osPriorityRealtime; break;
            default: mbedPrio = osPriorityNormal; break;
        }

        // mbed Thread constructor: Thread(osPriority priority, uint32_t stack_size, unsigned char *stack_mem, const char *name)
        w->th = new rtos::Thread(mbedPrio, stackBytes, nullptr, name);
        w->th->start(mbed::callback(tramp, w));
        return Task{w};
    }

    void taskYield() { rtos::ThisThread::yield(); }
    void sleepMs(uint32_t ms){ rtos::ThisThread::sleep_for(std::chrono::milliseconds(ms)); }

    Event eventCreate(bool initial){
        auto *s = new rtos::Semaphore(initial?1:0, 1);
        return Event{s};
    }

    void eventWait(hal::Event e, uint32_t timeoutMs) {
        if(!e.p) return;
        auto* sem = static_cast<rtos::Semaphore*>(e.p);
        if (timeoutMs == 0xFFFFFFFF) {
            sem->acquire();
        } else {
            sem->try_acquire_for(std::chrono::milliseconds(timeoutMs));
        }
    }

    void eventGiveFromISR(Event e){
        static_cast<rtos::Semaphore*>(e.p)->release();
    }
}
#endif
