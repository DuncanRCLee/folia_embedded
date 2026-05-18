#ifdef ARDUINO_ARCH_RENESAS
#include <Arduino.h>
#undef abs
#undef min
#undef max

#include "hal.hpp"
#include <Arduino_FreeRTOS.h>

namespace hal {

    void schedulerStart(){ vTaskStartScheduler(); }

    Task taskCreate(TaskFn fn, void* arg, const char* name, uint32_t stackBytes, uint32_t prio){
        TaskHandle_t h=nullptr;
        BaseType_t result = xTaskCreate((TaskFunction_t)fn, name, stackBytes/sizeof(StackType_t), arg, prio, &h);
        if(result != pdPASS) {
            h = nullptr;
        }
        return Task{h};
    }

    void taskYield(){ taskYIELD(); }
    void sleepMs(uint32_t ms){ vTaskDelay(pdMS_TO_TICKS(ms)); }

    Event eventCreate(bool initial){
        // Use counting semaphore (max=1) instead of binary for proper event semantics
        SemaphoreHandle_t s = xSemaphoreCreateCounting(1, initial ? 1 : 0);
        return Event{s};
    }

    void eventWait(Event e, uint32_t timeoutMs){
        if(e.p) {
            xSemaphoreTake((SemaphoreHandle_t)e.p, timeoutMs==0xFFFFFFFF ? portMAX_DELAY : pdMS_TO_TICKS(timeoutMs));
        }
    }

    void eventGiveFromISR(Event e){
        if(e.p) {
            BaseType_t hpw = pdFALSE;
            xSemaphoreGiveFromISR((SemaphoreHandle_t)e.p, &hpw);
            portYIELD_FROM_ISR(hpw);
        }
    }

} // ns
#endif
