#include "ExampleScene.h"

CRC_SCENE_STATE ExampleSceneMani::Start()
{
    return CRC_SCENE_STATE_EXECUTING;
}

CRC_SCENE_STATE ExampleSceneMani::Update()
{
    if (GetInput()->IsKeyDown(CRC_KEY_MSG_ESCAPE))
    {
        OutputDebugStringA("Example Scene Escape\n");
    }

    if (GetInput()->GetKeyText() != "")
    {
        OutputDebugStringA(GetInput()->GetKeyText().c_str());
        OutputDebugStringA("\n");
    }

    if (GetInput()->IsKeyDouble(CRC_KEY_MSG_A))
    {
        OutputDebugStringA("Double A\n");
    }

    if (GetInput()->IsMouse(CRC_MOUSE_MSG_LBTN))
    {
        OutputDebugStringA("LBTN\n");
    }

    if (GetInput()->IsMouseDouble(CRC_MOUSE_MSG_LBTN))
    {
        OutputDebugStringA("Double LBTN\n");
    }

    int mouseWheel = GetInput()->GetMouseWheelDelta();
    if (mouseWheel != 0)
    {
        OutputDebugStringA("Mouse Wheel: ");
        OutputDebugStringA(std::to_string(mouseWheel).c_str());
        OutputDebugStringA("\n");
    }
    
    return CRC_SCENE_STATE_EXECUTING;
}

CRC_SCENE_STATE ExampleSceneMani::Destroy()
{
    return CRC_SCENE_STATE_EXECUTING;
}

CRC_SCENE_STATE ExampleSceneMani::ReStart()
{
    return CRC_SCENE_STATE_EXECUTING;
}
