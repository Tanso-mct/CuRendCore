#pragma once

#include "CRC_config.h"

#include <array>

struct CRC_API CRCInputState
{
    bool isPressed = false;
    bool isReleased = false;
    bool isHeld = false;
};

template <std::size_t N>
class CRC_API CRCInputStateSet
{
private:
    CRCInputState states_[N];

public:
    CRCInputStateSet()
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            states_[i] = CRCInputState();
        }
    }

    ~CRCInputStateSet() = default;

    // Delete copy constructor and operator=.
    CRCInputStateSet(const CRCInputStateSet&) = delete;
    CRCInputStateSet& operator=(const CRCInputStateSet&) = delete;

    // Delete move constructor and operator=.
    CRCInputStateSet(CRCInputStateSet&&) = delete;
    CRCInputStateSet& operator=(CRCInputStateSet&&) = delete;

    CRCInputState& operator[](std::size_t index)
    {
        return states_[index];
    }

    const CRCInputState& operator[](std::size_t index) const
    {
        return states_[index];
    }
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
