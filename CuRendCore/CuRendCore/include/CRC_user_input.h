#pragma once

#include "CRC_config.h"
#include "CRC_funcs.cuh"
#include "CRC_event.h"
#include "CRC_container.h"

#include <unordered_map>
#include <utility>
#include <chrono>
#include <ctime>

struct CRC_API CRCInputState
{
    bool isPressed = false;
    bool isReleased = false;
    bool isHeld = false;
    bool isDBL = false;
};

enum class CRC_API CRC_KEY : std::size_t
{
    A = 0, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    RETURN, ESCAPE, SPACE, TAB, BACKSPACE,
    R_ALT, L_ALT, L_SHIFT, R_SHIFT, L_CTRL, R_CTRL, L_WIN, R_WIN,
    UP, DOWN, LEFT, RIGHT,
    INSERT, DEL, HOME, END, PAGE_UP, PAGE_DOWN, CAPS_LOCK,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13,
    ALPHA_0, ALPHA_1, ALPHA_2, ALPHA_3, ALPHA_4, ALPHA_5, ALPHA_6, ALPHA_7, ALPHA_8, ALPHA_9,
    NUMPAD_0, NUMPAD_1, NUMPAD_2, NUMPAD_3, NUMPAD_4, NUMPAD_5, NUMPAD_6, NUMPAD_7, NUMPAD_8, NUMPAD_9, NUMPAD_ENTER,
    OEM_1, OEM_PLUS, OEM_COMMA, OEM_MINUS, OEM_PERIOD, OEM_2, OEM_3, OEM_4, OEM_5, OEM_6, OEM_7, OEM_102,
    COUNT
};

enum class CRC_API CRC_MOUSE : std::size_t
{
    LEFT = 0, RIGHT, MIDDLE, WHEEL, MOVE, COUNT
};

class CRC_API CRCUserInputAttr : public ICRCContainable
{
private:
    double dblTapTime_ = 300.0;

    CRCInputState keyState_[static_cast<std::size_t>(CRC_KEY::COUNT)];
    CRC_KEY previousKey_ = CRC_KEY::COUNT;
    std::chrono::high_resolution_clock::time_point keyTime_ = std::chrono::high_resolution_clock::now();

    CRCInputState mouseState_[static_cast<std::size_t>(CRC_MOUSE::COUNT)];
    CRC_MOUSE previousMouse_ = CRC_MOUSE::COUNT;
    std::chrono::high_resolution_clock::time_point mouseTime_ = std::chrono::high_resolution_clock::now();

    int mouseWheelDelta_ = 0;
    float mousePosX_ = 0.0f;
    float mousePosY_ = 0.0f;

public:
    CRCUserInputAttr();
    ~CRCUserInputAttr() override = default;

    void SetDBLTapTime(double time) { dblTapTime_ = time; };

    const CRCInputState& GetKeyState(CRC_KEY key) const { return keyState_[static_cast<std::size_t>(key)]; };
    const CRCInputState& GetMouseState(CRC_MOUSE btn) const { return mouseState_[static_cast<std::size_t>(btn)]; };

    const int& GetMouseWheelDelta() const { return mouseWheelDelta_; };
    const float& GetMousePosX() const { return mousePosX_; };
    const float& GetMousePosY() const { return mousePosY_; };

    friend class CRCUserInputEvent;
};

class CRC_API CRCUserInputEvent : public ICRCWinMsgEvent
{
private:
    const int idAttr_ = CRC::ID_INVALID;

    std::unordered_map<std::pair<WPARAM, bool>, CRC_KEY, CRC::PairHash, CRC::PairEqual> keyMap_;
    std::unordered_map<WPARAM, std::pair<CRC_MOUSE, CRCInputState>> mouseMap_;

public:
    CRCUserInputEvent(int& idAttr);
    ~CRCUserInputEvent() override = default;

    virtual void OnUpdate(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam) override;
    virtual void OnKeyDown(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam) override;
    virtual void OnKeyUp(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam) override;
    virtual void OnMouse(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam) override;
};
