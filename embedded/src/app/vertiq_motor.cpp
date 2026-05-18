#include "vertiq_motor.hpp"
#include "indicator.hpp"
#include "network.hpp"
#include <ctype.h>

namespace vertiq {

Motor::Motor(PinName rxPin, PinName txPin)
    : serial_(rxPin, txPin, NC, NC)
    , iq_(serial_)
    , mult_(0)
    , drive_(0)
    , pwr_(0)
    , currentState_(0)
    , zeroAngle_(0)
    , motorState_(MotorState::DISABLED)
{
}

int8_t Motor::setup() {
    serial_.begin(MOTOR_BAUD_RATE);
    delay(100);

    // Set current position as zero reference
    float startAng;
    iq_.get(drive_.obs_angle_, startAng);
    iq_.set(mult_.zero_angle_, startAng);
    iq_.get(mult_.zero_angle_, zeroAngle_);

    // Configure PID for position holding
    iq_.set(mult_.angle_Kp_, 4.0f);
    iq_.set(mult_.angle_Ki_, 0.1f);
    iq_.set(mult_.angle_Kd_, 0.1f);

    enable();
    currentState_ = 0;

    network::println("Motor ready (coasting)");
    return 0;
}

int8_t Motor::loop() {
    return 0;
}

int8_t Motor::enable() {
    if (motorState_ == MotorState::ERROR) {
        return -1;
    }
    motorState_ = MotorState::ENABLED;
    return 0;
}

int8_t Motor::disable() {
    motorState_ = MotorState::DISABLED;
    return 0;
}

int8_t Motor::moveToState(int state) {
    if (motorState_ != MotorState::ENABLED || state < 0 || state >= MAX_STATE) {
        return -1;
    }

    float target = ROTATION_PER_STATE * state;
    iq_.set(mult_.ctrl_angle_, target);

    currentState_ = state;

    char buf[64];
    snprintf(buf, sizeof(buf), "-> State %d (%.1fmm)", state, state * LINEAR_STEP_MM);
    network::println(buf);

    return 0;
}

int8_t Motor::moveToStateVerified(int state, float tolerance_rad) {
    if (motorState_ != MotorState::ENABLED || state < 0 || state >= MAX_STATE) {
        return -1;
    }

    float target = ROTATION_PER_STATE * state;

    iq_.set(mult_.ctrl_angle_, target);
    currentState_ = state;

    delay(500);  // Give PID controller time to settle

    float actualDisplacement;
    iq_.get(mult_.obs_angular_displacement_, actualDisplacement);
    float error = abs(actualDisplacement - target);

    if (error > tolerance_rad) {
        delay(300);
        iq_.get(mult_.obs_angular_displacement_, actualDisplacement);
        error = abs(actualDisplacement - target);
    }

    return (error <= tolerance_rad) ? 0 : -1;
}

int8_t Motor::setAngleKp(float kp) {
    if (motorState_ != MotorState::ENABLED) return -1;

    float current;
    iq_.set(mult_.angle_Kp_, kp);
    iq_.get(mult_.angle_Kp_, current);

    if (abs(current - kp) < 0.001f) {
        char buf[48];
        snprintf(buf, sizeof(buf), "Angle Kp set to: %.3f", current);
        network::println(buf);
        return 0;
    }
    return -1;
}

int8_t Motor::setAngleKi(float ki) {
    if (motorState_ != MotorState::ENABLED) return -1;

    float current;
    iq_.set(mult_.angle_Ki_, ki);
    iq_.get(mult_.angle_Ki_, current);

    if (abs(current - ki) < 0.001f) {
        char buf[48];
        snprintf(buf, sizeof(buf), "Angle Ki set to: %.3f", current);
        network::println(buf);
        return 0;
    }
    return -1;
}

int8_t Motor::setAngleKd(float kd) {
    if (motorState_ != MotorState::ENABLED) return -1;

    float current;
    iq_.set(mult_.angle_Kd_, kd);
    iq_.get(mult_.angle_Kd_, current);

    if (abs(current - kd) < 0.001f) {
        char buf[48];
        snprintf(buf, sizeof(buf), "Angle Kd set to: %.3f", current);
        network::println(buf);
        return 0;
    }
    return -1;
}

float Motor::getAngleKp() {
    float kp;
    iq_.get(mult_.angle_Kp_, kp);
    return kp;
}

float Motor::getAngleKi() {
    float ki;
    iq_.get(mult_.angle_Ki_, ki);
    return ki;
}

float Motor::getAngleKd() {
    float kd;
    iq_.get(mult_.angle_Kd_, kd);
    return kd;
}

float Motor::getVelocity() {
    float v;
    iq_.get(drive_.obs_velocity_, v);
    return v;
}

float Motor::getPosition() {
    float p;
    iq_.get(drive_.obs_angle_, p);
    return p;
}

float Motor::getDisplacement() {
    float d;
    iq_.get(mult_.obs_angular_displacement_, d);
    return d;
}

float Motor::getVoltage() {
    float v;
    iq_.get(pwr_.volts_, v);
    return v;
}

float Motor::getCurrent() {
    float a;
    iq_.get(pwr_.amps_, a);
    return a;
}

float Motor::getPower() {
    float w;
    iq_.get(pwr_.watts_, w);
    return w;
}

int8_t Motor::commandFromChar(char c) {
    char uc = toupper((unsigned char)c);
    int target = -1;
    switch (uc) {
        case 'A': target = 0; break;
        case 'S': target = 1; break;
        case 'D': target = 2; break;
        case 'F': target = 3; break;
        case 'G': target = 4; break;
        default: return -1;
    }
    return moveToState(target);
}

void Motor::printStatus() {
    char buf[384];
    int len = 0;

    float angularPos = getPosition();
    float displacement = getDisplacement();
    float linearPos = (displacement / (2.0f * PI)) * CAM_LEAD_MM;

    len += snprintf(buf + len, sizeof(buf) - len, "\n=== Motor Status ===\n");
    len += snprintf(buf + len, sizeof(buf) - len, "State: %d/%d (%.1fmm)\n",
                    currentState_, MAX_STATE - 1, currentState_ * LINEAR_STEP_MM);
    len += snprintf(buf + len, sizeof(buf) - len, "Position: %.4f rad\n", angularPos);
    len += snprintf(buf + len, sizeof(buf) - len, "Displacement: %.4f rad\n", displacement);
    len += snprintf(buf + len, sizeof(buf) - len, "Linear: %.2f mm\n", linearPos);
    len += snprintf(buf + len, sizeof(buf) - len, "Velocity: %.4f rad/s\n", getVelocity());
    len += snprintf(buf + len, sizeof(buf) - len, "Voltage: %.2f V\n", getVoltage());
    len += snprintf(buf + len, sizeof(buf) - len, "Current: %.3f A\n", getCurrent());
    len += snprintf(buf + len, sizeof(buf) - len, "Power: %.2f W\n", getPower());
    len += snprintf(buf + len, sizeof(buf) - len, "PID: Kp=%.3f Ki=%.3f Kd=%.3f\n",
                    getAngleKp(), getAngleKi(), getAngleKd());

    network::print(buf);
}

// MotorCommandQueue implementation

int8_t MotorCommandQueue::queueStateCommand(int state) {
    cmdState_ = state;
    cmdStatePending_ = true;
    return 0;
}

int8_t MotorCommandQueue::queueStatusCommand() {
    cmdStatusPending_ = true;
    return 0;
}

int8_t MotorCommandQueue::queuePidCommand(char gainType, float value) {
    cmdPidType_ = gainType;
    cmdPidValue_ = value;
    cmdPidPending_ = true;
    return 0;
}

int8_t MotorCommandQueue::processCommands() {
    // Process status request
    if (cmdStatusPending_) {
        cmdStatusPending_ = false;
        motor_.printStatus();
    }

    // Process PID updates
    if (cmdPidPending_) {
        char type = cmdPidType_;
        float value = cmdPidValue_;
        cmdPidPending_ = false;

        int8_t result = -1;
        switch (type) {
            case 'P': result = motor_.setAngleKp(value); break;
            case 'I': result = motor_.setAngleKi(value); break;
            case 'D': result = motor_.setAngleKd(value); break;
        }

        (void)result;  // Ignore result, error already logged at motor level
    }

    // Process state command
    if (cmdStatePending_) {
        int state = cmdState_;
        cmdStatePending_ = false;

        if (motor_.moveToState(state) == 0) {
            indicator::setMotorMoving(true);
            movePending_ = true;
            lastMoveTime_ = millis();
        }
    }

    // Check if move completed
    if (movePending_ && (millis() - lastMoveTime_ > (TRAJ_DURATION * 1000 + 200))) {
        movePending_ = false;
        indicator::setMotorMoving(false);
    }

    return 0;
}

// ============================================================================
// VirtualMotorCommandQueue
// ============================================================================

int8_t VirtualMotorCommandQueue::queueStateCommand(int state) {
    if (state < 0 || state >= MAX_STATE) {
        return -1;
    }
    pendingState_ = state;
    pendingStateCmdSet_ = true;
    return 0;
}

int8_t VirtualMotorCommandQueue::processCommands() {
    if (!pendingStateCmdSet_) {
        return 0;
    }

    int state = pendingState_;
    pendingStateCmdSet_ = false;

    int prev = currentState_;
    currentState_ = state;
    (void)prev;  // state change logged at call site

    return 0;
}

} // namespace vertiq
