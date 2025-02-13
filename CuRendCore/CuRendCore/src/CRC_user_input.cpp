#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_user_input.h"

CRCUserInputAttr::CRCUserInputAttr()
{
    for (std::size_t i = 0; i < static_cast<std::size_t>(CRC_KEY::COUNT); ++i)
    {
        keyState_[i].isPressed = false;
        keyState_[i].isReleased = false;
        keyState_[i].isHeld = false;
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(CRC_MOUSE_BTN::COUNT); ++i)
    {
        mouseState_[i].isPressed = false;
        mouseState_[i].isReleased = false;
        mouseState_[i].isHeld = false;
    }
}

void CRCUserInputListener::OnUpdate(ICRCContainable *attr, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "User Input Update" << std::endl;
}

void CRCUserInputListener::OnKeyDown(ICRCContainable *attr, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "User Input Key Down" << std::endl;
}

void CRCUserInputListener::OnKeyUp(ICRCContainable *attr, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "User Input Key Up" << std::endl;
}

void CRCUserInputListener::OnMouse(ICRCContainable *attr, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "User Input Mouse" << std::endl;
}
