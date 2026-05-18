#pragma once

#include "Arduino.h"
#include "pinDefinitions.h"
#include <iq_module_communication.hpp>

namespace vertiq {

// Motor states
enum class MotorState : uint8_t {
    DISABLED = 0,
    ENABLED = 1,
    ERROR = 2
};

// Motor parameters
constexpr uint32_t MOTOR_BAUD_RATE = 115200;
constexpr int MAX_STATE = 5;  // Number of states (0-4)

// Barrel cam parameters
constexpr float CAM_LEAD_MM = 11.0f;           // Stroke per revolution (mm)
constexpr float LINEAR_STEP_MM = 2.5f;         // Linear movement per state (mm)
constexpr float ROTATION_PER_STATE = (LINEAR_STEP_MM / CAM_LEAD_MM) * 2.0f * PI;  // Rotation per state (rad)
constexpr float TRAJ_DURATION = 0.2f;          // Time to move between states (seconds)

class Motor {
    public:
        Motor(PinName rxPin, PinName txPin);
        ~Motor() = default;

        // Setup and basic control
        int8_t setup();
        int8_t loop();
        int8_t enable();
        int8_t disable();

        // State control
        int8_t moveToState(int state);
        int8_t moveToStateVerified(int state, float tolerance_rad = 0.1f);
        int getCurrentState() const { return currentState_; }
        float getZeroAngle() const { return zeroAngle_; }

        // PID Control
        int8_t setAngleKp(float kp);
        int8_t setAngleKi(float ki);
        int8_t setAngleKd(float kd);
        float getAngleKp();
        float getAngleKi();
        float getAngleKd();

        // Status and feedback
        float getVelocity();
        float getPosition();
        float getDisplacement();
        float getVoltage();
        float getCurrent();
        float getPower();
        MotorState getState() const { return motorState_; }
        uint8_t getStateAsU8() const { return static_cast<uint8_t>(currentState_); }

        // Command interface
        void printStatus();
        int8_t commandFromChar(char c);

    private:
        UART serial_;
        IqSerial iq_;

        // Vertiq API clients
        MultiTurnAngleControlClient mult_;
        BrushlessDriveClient drive_;
        PowerMonitorClient pwr_;

        // State
        int currentState_;
        float zeroAngle_;
        MotorState motorState_;
};

// Command queue for thread-safe motor control
class MotorCommandQueue {
    public:
        MotorCommandQueue(Motor& motor) : motor_(motor) {}

        // Queue commands (safe to call from network/comm layer)
        int8_t queueStateCommand(int state);
        int8_t queueStatusCommand();
        int8_t queuePidCommand(char gainType, float value);

        // Process queued commands (call from main loop)
        int8_t processCommands();

    private:
        Motor& motor_;

        // Command queue (volatile for thread safety)
        volatile int cmdState_ = -1;
        volatile bool cmdStatePending_ = false;
        volatile bool cmdStatusPending_ = false;
        volatile char cmdPidType_ = 0;
        volatile float cmdPidValue_ = 0.0f;
        volatile bool cmdPidPending_ = false;

        // Move tracking
        bool movePending_ = false;
        uint32_t lastMoveTime_ = 0;
    };

// ============================================================================
// Virtual motor – mirrors MotorCommandQueue interface for testing AUTO mode
// without real hardware.  Logs state transitions to Serial and TCP.
// ============================================================================
class VirtualMotorCommandQueue {
public:
    VirtualMotorCommandQueue() = default;

    /// Queue a state change command (thread-safe by design – same as real queue)
    int8_t queueStateCommand(int state);

    /// Process queued commands; call from main loop (or after queueStateCommand)
    int8_t processCommands();

    /// Read back current simulated state (-1 = uninitialised)
    int getCurrentState() const { return currentState_; }

    /// True if a command has been queued but not yet processed
    bool hasPendingCommand() const { return pendingStateCmdSet_; }

private:
    volatile int  pendingState_{-1};
    volatile bool pendingStateCmdSet_{false};
    int currentState_{-1};
};

} // namespace vertiq
