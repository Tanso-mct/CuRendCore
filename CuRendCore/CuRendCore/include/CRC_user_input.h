#pragma once

#include "CRC_config.h"
#include "CRC_event_listener.h"
#include "CRC_container.h"

#include <array>

struct CRC_API CRCInputState
{
    bool isPressed = false;
    bool isReleased = false;
    bool isHeld = false;
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

enum class CRC_API CRC_MOUSE_BTN : std::size_t
{
    LEFT = 0, RIGHT, MIDDLE, X1, X2, COUNT
};

class CRC_API CRCUserInputAttr : public ICRCContainable
{
private:
    CRCInputState keyState_[static_cast<std::size_t>(CRC_KEY::COUNT)];
    CRCInputState mouseState_[static_cast<std::size_t>(CRC_MOUSE_BTN::COUNT)];

public:
    CRCUserInputAttr();
    ~CRCUserInputAttr() override = default;

    CRCInputState GetKeyState(CRC_KEY key) const { return keyState_[static_cast<std::size_t>(key)]; };
    CRCInputState GetMouseState(CRC_MOUSE_BTN btn) const { return mouseState_[static_cast<std::size_t>(btn)]; };

    friend class CRCUserInputListener;
};

class CRC_API CRCUserInputListener : public ICRCWinMsgListener
{
public:
    CRCUserInputListener() = default;
    ~CRCUserInputListener() override = default;

    virtual void OnUpdate(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam) override;
    virtual void OnKeyDown(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam) override;
    virtual void OnKeyUp(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam) override;
    virtual void OnMouse(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam) override;
};
