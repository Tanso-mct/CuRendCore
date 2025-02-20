#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_user_input.h"

CRCUserInputAttr::CRCUserInputAttr()
{
    for (std::size_t i = 0; i < static_cast<std::size_t>(CRC_KEY::COUNT); ++i)
    {
        keyState_[i].isPressed = false;
        keyState_[i].isReleased = false;
        keyState_[i].isHeld = false;
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(CRC_MOUSE::COUNT); ++i)
    {
        mouseState_[i].isPressed = false;
        mouseState_[i].isReleased = false;
        mouseState_[i].isHeld = false;
    }
}

CRCUserInputEvent::CRCUserInputEvent(int& idAttr)
: idAttr_(idAttr)
{
    // Initialize key map.
    keyMap_[{'A', false}] = CRC_KEY::A;
    keyMap_[{'B', false}] = CRC_KEY::B;
    keyMap_[{'C', false}] = CRC_KEY::C;
    keyMap_[{'D', false}] = CRC_KEY::D;
    keyMap_[{'E', false}] = CRC_KEY::E;
    keyMap_[{'F', false}] = CRC_KEY::F;
    keyMap_[{'G', false}] = CRC_KEY::G;
    keyMap_[{'H', false}] = CRC_KEY::H;
    keyMap_[{'I', false}] = CRC_KEY::I;
    keyMap_[{'J', false}] = CRC_KEY::J;
    keyMap_[{'K', false}] = CRC_KEY::K;
    keyMap_[{'L', false}] = CRC_KEY::L;
    keyMap_[{'M', false}] = CRC_KEY::M;
    keyMap_[{'N', false}] = CRC_KEY::N;
    keyMap_[{'O', false}] = CRC_KEY::O;
    keyMap_[{'P', false}] = CRC_KEY::P;
    keyMap_[{'Q', false}] = CRC_KEY::Q;
    keyMap_[{'R', false}] = CRC_KEY::R;
    keyMap_[{'S', false}] = CRC_KEY::S;
    keyMap_[{'T', false}] = CRC_KEY::T;
    keyMap_[{'U', false}] = CRC_KEY::U;
    keyMap_[{'V', false}] = CRC_KEY::V;
    keyMap_[{'W', false}] = CRC_KEY::W;
    keyMap_[{'X', false}] = CRC_KEY::X;
    keyMap_[{'Y', false}] = CRC_KEY::Y;
    keyMap_[{'Z', false}] = CRC_KEY::Z;

    keyMap_[{VK_RETURN, false}] = CRC_KEY::RETURN;
    keyMap_[{VK_ESCAPE, false}] = CRC_KEY::ESCAPE;
    keyMap_[{VK_SPACE, false}] = CRC_KEY::SPACE;
    keyMap_[{VK_TAB, false}] = CRC_KEY::TAB;
    keyMap_[{VK_BACK, false}] = CRC_KEY::BACKSPACE;

    keyMap_[{VK_MENU, true}] = CRC_KEY::R_ALT;
    keyMap_[{VK_MENU, false}] = CRC_KEY::L_ALT;
    keyMap_[{VK_SHIFT, true}] = CRC_KEY::R_SHIFT;
    keyMap_[{VK_SHIFT, false}] = CRC_KEY::L_SHIFT;
    keyMap_[{VK_CONTROL, true}] = CRC_KEY::R_CTRL;
    keyMap_[{VK_CONTROL, false}] = CRC_KEY::L_CTRL;
    keyMap_[{VK_LWIN, false}] = CRC_KEY::L_WIN;
    keyMap_[{VK_RWIN, false}] = CRC_KEY::R_WIN;

    keyMap_[{VK_UP, false}] = CRC_KEY::UP;
    keyMap_[{VK_DOWN, false}] = CRC_KEY::DOWN;
    keyMap_[{VK_LEFT, false}] = CRC_KEY::LEFT;
    keyMap_[{VK_RIGHT, false}] = CRC_KEY::RIGHT;

    keyMap_[{VK_INSERT, false}] = CRC_KEY::INSERT;
    keyMap_[{VK_DELETE, false}] = CRC_KEY::DEL;
    keyMap_[{VK_HOME, false}] = CRC_KEY::HOME;
    keyMap_[{VK_END, false}] = CRC_KEY::END;
    keyMap_[{VK_PRIOR, false}] = CRC_KEY::PAGE_UP;
    keyMap_[{VK_NEXT, false}] = CRC_KEY::PAGE_DOWN;
    keyMap_[{VK_CAPITAL, false}] = CRC_KEY::CAPS_LOCK;

    keyMap_[{VK_F1, false}] = CRC_KEY::F1;
    keyMap_[{VK_F2, false}] = CRC_KEY::F2;
    keyMap_[{VK_F3, false}] = CRC_KEY::F3;
    keyMap_[{VK_F4, false}] = CRC_KEY::F4;
    keyMap_[{VK_F5, false}] = CRC_KEY::F5;
    keyMap_[{VK_F6, false}] = CRC_KEY::F6;
    keyMap_[{VK_F7, false}] = CRC_KEY::F7;
    keyMap_[{VK_F8, false}] = CRC_KEY::F8;
    keyMap_[{VK_F9, false}] = CRC_KEY::F9;
    keyMap_[{VK_F10, false}] = CRC_KEY::F10;
    keyMap_[{VK_F11, false}] = CRC_KEY::F11;
    keyMap_[{VK_F12, false}] = CRC_KEY::F12;
    keyMap_[{VK_F13, false}] = CRC_KEY::F13;

    keyMap_[{'0', false}] = CRC_KEY::ALPHA_0;
    keyMap_[{'1', false}] = CRC_KEY::ALPHA_1;
    keyMap_[{'2', false}] = CRC_KEY::ALPHA_2;
    keyMap_[{'3', false}] = CRC_KEY::ALPHA_3;
    keyMap_[{'4', false}] = CRC_KEY::ALPHA_4;
    keyMap_[{'5', false}] = CRC_KEY::ALPHA_5;
    keyMap_[{'6', false}] = CRC_KEY::ALPHA_6;
    keyMap_[{'7', false}] = CRC_KEY::ALPHA_7;
    keyMap_[{'8', false}] = CRC_KEY::ALPHA_8;
    keyMap_[{'9', false}] = CRC_KEY::ALPHA_9;

    keyMap_[{VK_NUMPAD0, false}] = CRC_KEY::NUMPAD_0;
    keyMap_[{VK_NUMPAD1, false}] = CRC_KEY::NUMPAD_1;
    keyMap_[{VK_NUMPAD2, false}] = CRC_KEY::NUMPAD_2;
    keyMap_[{VK_NUMPAD3, false}] = CRC_KEY::NUMPAD_3;
    keyMap_[{VK_NUMPAD4, false}] = CRC_KEY::NUMPAD_4;
    keyMap_[{VK_NUMPAD5, false}] = CRC_KEY::NUMPAD_5;
    keyMap_[{VK_NUMPAD6, false}] = CRC_KEY::NUMPAD_6;
    keyMap_[{VK_NUMPAD7, false}] = CRC_KEY::NUMPAD_7;
    keyMap_[{VK_NUMPAD8, false}] = CRC_KEY::NUMPAD_8;
    keyMap_[{VK_NUMPAD9, false}] = CRC_KEY::NUMPAD_9;

    keyMap_[{VK_OEM_1, false}] = CRC_KEY::OEM_1;
    keyMap_[{VK_OEM_PLUS, false}] = CRC_KEY::OEM_PLUS;
    keyMap_[{VK_OEM_COMMA, false}] = CRC_KEY::OEM_COMMA;
    keyMap_[{VK_OEM_MINUS, false}] = CRC_KEY::OEM_MINUS;
    keyMap_[{VK_OEM_PERIOD, false}] = CRC_KEY::OEM_PERIOD;
    keyMap_[{VK_OEM_2, false}] = CRC_KEY::OEM_2;
    keyMap_[{VK_OEM_3, false}] = CRC_KEY::OEM_3;
    keyMap_[{VK_OEM_4, false}] = CRC_KEY::OEM_4;
    keyMap_[{VK_OEM_5, false}] = CRC_KEY::OEM_5;
    keyMap_[{VK_OEM_6, false}] = CRC_KEY::OEM_6;
    keyMap_[{VK_OEM_7, false}] = CRC_KEY::OEM_7;
    keyMap_[{VK_OEM_102, false}] = CRC_KEY::OEM_102;

    // Initialize mouse map.
    CRCInputState pressState;
    pressState.isPressed = true;

    CRCInputState releaseState;
    releaseState.isReleased = true;
    
    mouseMap_[WM_LBUTTONDOWN] = std::make_pair(CRC_MOUSE::LEFT, pressState);
    mouseMap_[WM_LBUTTONUP] = std::make_pair(CRC_MOUSE::LEFT, releaseState);

    mouseMap_[WM_RBUTTONDOWN] = std::make_pair(CRC_MOUSE::RIGHT, pressState);
    mouseMap_[WM_RBUTTONUP] = std::make_pair(CRC_MOUSE::RIGHT, releaseState);

    mouseMap_[WM_MBUTTONDOWN] = std::make_pair(CRC_MOUSE::MIDDLE, pressState);
    mouseMap_[WM_MBUTTONUP] = std::make_pair(CRC_MOUSE::MIDDLE, releaseState);

    mouseMap_[WM_MOUSEWHEEL] = std::make_pair(CRC_MOUSE::WHEEL, CRCInputState());
    mouseMap_[WM_MOUSEMOVE] = std::make_pair(CRC_MOUSE::MOVE, CRCInputState());
}

void CRCUserInputEvent::OnUpdate(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam)
{
    CRCUserInputAttr* input = CRC::As<CRCUserInputAttr>(container->Get(idAttr_).get());
    if (!input) return;

    for (int i = 0; i < static_cast<int>(CRC_KEY::COUNT); i++)
    {
        if (input->keyState_[i].isPressed)
        {
            input->keyState_[i].isHeld = true;
            input->keyState_[i].isPressed = false;
        }
        else if (input->keyState_[i].isReleased)
        {
            input->keyState_[i].isHeld = false;
            input->keyState_[i].isReleased = false;

            if (input->previousKey_ == static_cast<CRC_KEY>(i))
            {
                auto currentTime = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>
                (
                    currentTime - input->keyTime_
                ).count();

                if (elapsed < input->dblTapTime_)
                {
                    input->keyState_[i].isDBL = true;
                }
            }

            input->previousKey_ = static_cast<CRC_KEY>(i);
            input->keyTime_ = std::chrono::high_resolution_clock::now();
        }
        else if (input->keyState_[i].isDBL)
        {
            input->keyState_[i].isDBL = false;
        }
    }

    for (int i = 0; i < static_cast<int>(CRC_MOUSE::COUNT); i++)
    {
        if (input->mouseState_[i].isPressed)
        {
            input->mouseState_[i].isHeld = true;
            input->mouseState_[i].isPressed = false;
        }
        else if (input->mouseState_[i].isReleased)
        {
            input->mouseState_[i].isHeld = false;
            input->mouseState_[i].isReleased = false;

            if (input->previousMouse_ == static_cast<CRC_MOUSE>(i))
            {
                auto currentTime = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>
                (
                    currentTime - input->mouseTime_
                ).count();

                if (elapsed < input->dblTapTime_)
                {
                    input->mouseState_[i].isDBL = true;
                }
            }

            input->previousMouse_ = static_cast<CRC_MOUSE>(i);
            input->mouseTime_ = std::chrono::high_resolution_clock::now();
        }
        else if (input->mouseState_[i].isDBL)
        {
            input->mouseState_[i].isDBL = false;
        }
    }

    input->mouseWheelDelta_ = 0;
}

void CRCUserInputEvent::OnKeyDown(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam)
{
    CRCUserInputAttr* input = CRC::As<CRCUserInputAttr>(container->Get(idAttr_).get());
    if (!input) return;

    if (keyMap_.find({wParam, (lParam & (1 << 24)) != 0}) == keyMap_.end()) return;

    if (input->keyState_[static_cast<std::size_t>(keyMap_[{wParam, (lParam & (1 << 24)) != 0}])].isHeld) return;
    input->keyState_[static_cast<std::size_t>(keyMap_[{wParam, (lParam & (1 << 24)) != 0}])].isPressed = true;
}

void CRCUserInputEvent::OnKeyUp(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam)
{
    CRCUserInputAttr* input = CRC::As<CRCUserInputAttr>(container->Get(idAttr_).get());
    if (!input) return;

    if (keyMap_.find({wParam, (lParam & (1 << 24)) != 0}) == keyMap_.end()) return;
    input->keyState_[static_cast<std::size_t>(keyMap_[{wParam, (lParam & (1 << 24)) != 0}])].isReleased = true;
}

void CRCUserInputEvent::OnMouse(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam)
{
    CRCUserInputAttr* input = CRC::As<CRCUserInputAttr>(container->Get(idAttr_).get());
    if (!input) return;

    if (mouseMap_.find(msg) == mouseMap_.end()) return;
    input->mouseState_[static_cast<std::size_t>(mouseMap_[msg].first)] = mouseMap_[msg].second;

    if (msg == WM_MOUSEWHEEL) input->mouseWheelDelta_ = GET_WHEEL_DELTA_WPARAM(wParam);
    else if (msg == WM_MOUSEMOVE)
    {
        input->mousePosX_ = LOWORD(lParam);
        input->mousePosY_ = HIWORD(lParam);
    }
}
