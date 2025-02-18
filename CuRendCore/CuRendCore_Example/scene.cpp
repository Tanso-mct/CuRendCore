#include "scene.h"

#include <iostream>

MainSceneEvent::MainSceneEvent(int &idMainWindow, int &idMainScene, int &idUserInput)
: idMainWindow_(idMainWindow), idMainScene_(idMainScene), idUserInput_(idUserInput)
{
}

MainSceneEvent::~MainSceneEvent()
{
}

void MainSceneEvent::InputHandleExample(std::unique_ptr<ICRCContainable>& inputAttr)
{
    CRCUserInputAttr* input = CRC::PtrAs<CRCUserInputAttr>(inputAttr.get());
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

void MainSceneEvent::OnUpdate(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam)
{   
    InputHandleExample(container->Get(idUserInput_));
}

void MainSceneEvent::OnSize(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Scene Size" << std::endl;
}

void MainSceneEvent::OnDestroy(std::unique_ptr<ICRCContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam)
{
    std::cout << "Main Scene End" << std::endl;
}
