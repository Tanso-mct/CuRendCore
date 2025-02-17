#include "scene.h"

#include "slot_id.h"

#include <iostream>

void MainSceneListener::InputHandleExample()
{
    CRCUserInputAttr* input = CRC::GetContainablePtr<CRCUserInputAttr>
    (
        SlotID::USER_INPUT_CONTAINER, SlotID::USER_INPUT
    );
    if (!input) return;

    if (input->GetKeyState(CRC_KEY::W).isPressed)
    {
        std::cout << "Main Scene Press W" << std::endl;
    }

    if (input->GetKeyState(CRC_KEY::W).isHeld)
    {
        std::cout << "Main Scene Hold W" << std::endl;
    }

    if (input->GetKeyState(CRC_KEY::W).isReleased)
    {
        std::cout << "Main Scene Release W" << std::endl;
    }

    if (input->GetKeyState(CRC_KEY::D).isDBL)
    {
        std::cout << "Main Scene Double Press D" << std::endl;
    }

    if (input->GetMouseState(CRC_MOUSE::LEFT).isPressed)
    {
        std::cout << "Main Scene Down Mouse Left Button" << std::endl;
    }

    if (input->GetMouseState(CRC_MOUSE::LEFT).isHeld)
    {
        std::cout << "Main Scene Hold Mouse Left Button" << std::endl;
    }

    if (input->GetMouseState(CRC_MOUSE::LEFT).isReleased)
    {
        std::cout << "Main Scene Release Mouse Left Button" << std::endl;
    }

    if (input->GetMouseState(CRC_MOUSE::RIGHT).isDBL)
    {
        std::cout << "Main Scene Double Press Mouse Right Button" << std::endl;
    }

    if (input->GetMouseWheelDelta() != 0)
    {
        std::cout << "Main Scene Mouse Wheel Delta: " << input->GetMouseWheelDelta() << std::endl;
    }

    if (input->GetKeyState(CRC_KEY::RETURN).isPressed)
    {
        std::cout << "Main Scene Mouse Pos: " << 
        input->GetMousePosX() << ", " << input->GetMousePosY() << std::endl;
    }
}

void MainSceneListener::OnUpdate(UINT msg, WPARAM wParam, LPARAM lParam)
{   
    InputHandleExample();
}

void MainSceneListener::OnSize(UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Scene Size" << std::endl;
}

void MainSceneListener::OnDestroy(UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Scene End" << std::endl;
}
