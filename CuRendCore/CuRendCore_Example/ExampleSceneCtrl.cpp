#include "ExampleSceneCtrl.h"

ExampleSceneCtrl::ExampleSceneCtrl()
{
    OutputDebugStringA("ExampleSceneCtrl::ExampleSceneCtrl()\n");
}

ExampleSceneCtrl::~ExampleSceneCtrl()
{
    OutputDebugStringA("ExampleSceneCtrl::~ExampleSceneCtrl()\n");
}

void ExampleSceneCtrl::Init()
{
    OutputDebugStringA("ExampleSceneCtrl::Init()\n");

    // Get the resource factory instance
    CRC::ResourceFactory* rf = CRC::CuRendCore::GetInstance()->resourceFc;

    // Create resources
    CRC::RESOURCEATTR rattr;
    rattr.path = "Resource/Object/Cube.obj";
    slotCubeObj = rf->CreateResource(rattr);

    rattr.path = "Resource/Object/Floor.obj";
    slotFloorObj = rf->CreateResource(rattr);

    rattr.path = "Resource/Material/Cube/cube_baseColor.png";
    slotCubeBasePng = rf->CreateResource(rattr);

    rattr.path = "Resource/Material/Cube/cube_normal.png";
    slotCubeNormalPng = rf->CreateResource(rattr);

    rattr.path = "Resource/Material/Floor/floor_baseColor.png";
    slotFloorBasePng = rf->CreateResource(rattr);

    rattr.path = "Resource/Material/Floor/floor_normal.png";
    slotFloorNormalPng = rf->CreateResource(rattr);

    // Add resources to the scene controller
    AddResource(slotCubeObj);
    AddResource(slotFloorObj);
    AddResource(slotCubeBasePng);
    AddResource(slotCubeNormalPng);
    AddResource(slotFloorBasePng);
    AddResource(slotFloorNormalPng);

    // Load resources
    LoadResources();
}

void ExampleSceneCtrl::Update()
{
    // OutputDebugStringA("ExampleSceneCtrl::Update()\n");

    if (GetInput()->IsKeyDown(CRC_KEY_MSG_ESCAPE))
    {
        CRC::WindowFactory* wf = CRC::CuRendCore::GetInstance()->windowFc;
        wf->SetSceneCtrl(slotWnd, exampleScene2Ctrl.lock());

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
}

void ExampleSceneCtrl::ReStart()
{
    // OutputDebugStringA("ExampleSceneCtrl::ReStart()\n");
}

void ExampleSceneCtrl::End()
{
    // OutputDebugStringA("ExampleSceneCtrl::End()\n");

    // Unload resources
    UnLoadResources();

    NeedDestroy(false);
    NeedInit();
}

ExampleScene2Ctrl::ExampleScene2Ctrl()
{
    OutputDebugStringA("ExampleScene2Ctrl::ExampleScene2Ctrl()\n");
}

ExampleScene2Ctrl::~ExampleScene2Ctrl()
{
    OutputDebugStringA("ExampleScene2Ctrl::~ExampleScene2Ctrl()\n");
}

void ExampleScene2Ctrl::Init()
{
    OutputDebugStringA("ExampleScene2Ctrl::Init()\n");

    // Get the resource factory instance
    CRC::ResourceFactory* rf = CRC::CuRendCore::GetInstance()->resourceFc;

    // Create resources
    CRC::RESOURCEATTR rattr;
    rattr.path = "Resource/Image/rect_background.png";
    slotRectBackgroud = rf->CreateResource(rattr);

    rattr.path = "Resource/Image/rect_red.png";
    slotRectRed = rf->CreateResource(rattr);

    rattr.path = "Resource/Image/rect_blue.png";
    slotRectBlue = rf->CreateResource(rattr);

    // Add resources to the scene controller
    AddResource(slotRectBackgroud);
    AddResource(slotRectRed);
    AddResource(slotRectBlue);

    // Load resources
    LoadResources();
}

void ExampleScene2Ctrl::Update()
{
    // OutputDebugStringA("ExampleScene2Ctrl::Update()\n");

    if (GetInput()->IsKeyDown(CRC_KEY_MSG_ESCAPE))
    {
        OutputDebugStringA("Example Scene2 Escape\n");
    }
}

void ExampleScene2Ctrl::ReStart()
{
    // OutputDebugStringA("ExampleScene2Ctrl::ReStart()\n");
}

void ExampleScene2Ctrl::End()
{
    // OutputDebugStringA("ExampleScene2Ctrl::End()\n");

    // Unload resources
    UnLoadResources();

    NeedDestroy(true);
}
