#include "scene.h"

#include <iostream>

void MainSceneListener::InputHandleExample()
{
    CRCUserInputAttr* ptrInput = CRC::PtrAs<CRCUserInputAttr>(input.get());
    if (!ptrInput) return;

    if (ptrInput->GetKeyState(CRC_KEY::W).isPressed)
    {
        std::cout << "Main Scene Press W" << std::endl;
    }

    if (ptrInput->GetKeyState(CRC_KEY::W).isHeld)
    {
        std::cout << "Main Scene Hold W" << std::endl;
    }

    if (ptrInput->GetKeyState(CRC_KEY::W).isReleased)
    {
        std::cout << "Main Scene Release W" << std::endl;
    }

    if (ptrInput->GetKeyState(CRC_KEY::D).isDBL)
    {
        std::cout << "Main Scene Double Press D" << std::endl;
    }

    if (ptrInput->GetMouseState(CRC_MOUSE::LEFT).isPressed)
    {
        std::cout << "Main Scene Down Mouse Left Button" << std::endl;
    }

    if (ptrInput->GetMouseState(CRC_MOUSE::LEFT).isHeld)
    {
        std::cout << "Main Scene Hold Mouse Left Button" << std::endl;
    }

    if (ptrInput->GetMouseState(CRC_MOUSE::LEFT).isReleased)
    {
        std::cout << "Main Scene Release Mouse Left Button" << std::endl;
    }

    if (ptrInput->GetMouseState(CRC_MOUSE::RIGHT).isDBL)
    {
        std::cout << "Main Scene Double Press Mouse Right Button" << std::endl;
    }

    if (ptrInput->GetMouseWheelDelta() != 0)
    {
        std::cout << "Main Scene Mouse Wheel Delta: " << ptrInput->GetMouseWheelDelta() << std::endl;
    }

    if (ptrInput->GetKeyState(CRC_KEY::RETURN).isPressed)
    {
        std::cout << "Main Scene Mouse Pos: " << 
        ptrInput->GetMousePosX() << ", " << ptrInput->GetMousePosY() << std::endl;
    }
}

void MainSceneListener::OnUpdate(ICRCContainable *attr, UINT msg, WPARAM wParam, LPARAM lParam)
{   
    InputHandleExample();
}

void MainSceneListener::OnSize(ICRCContainable *attr, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Scene Size" << std::endl;
}

void MainSceneListener::OnDestroy(ICRCContainable *attr, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Scene End" << std::endl;
}
