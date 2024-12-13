#include "ExampleScene.h"

CRC_SCENE_STATE ExampleSceneMani::Start()
{
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

    rattr.path = "Resource/Image/rect_background.png";
    slotRectBackgroud = rf->CreateResource(rattr);

    rattr.path = "Resource/Image/rect_red.png";
    slotRectRed = rf->CreateResource(rattr);

    rattr.path = "Resource/Image/rect_blue.png";
    slotRectBlue = rf->CreateResource(rattr);

    // Add resources to the scene controller
    GetScene()->AddResource(slotCubeObj);
    GetScene()->AddResource(slotFloorObj);
    GetScene()->AddResource(slotCubeBasePng);
    GetScene()->AddResource(slotCubeNormalPng);
    GetScene()->AddResource(slotFloorBasePng);
    GetScene()->AddResource(slotFloorNormalPng);

    GetScene()->AddResource(slotRectBackgroud);
    GetScene()->AddResource(slotRectRed);
    GetScene()->AddResource(slotRectBlue);

    // Load resources
    GetScene()->LoadResources();

    // Create main group
    CRC::GROUPATTR gattr;
    gattr.name = "Main";
    gattr.active = FALSE;
    slotMainGroup = GetScene()->CreateGroup(gattr);

    // Create components
    CRC::OBJECT_ATTR oattr;
    oattr.name = "Cube";
    oattr.ctrl = nullptr; // Set the default controller
    oattr.from = CRC_OBJECT_FROM_OBJ;
    oattr.slotRc = slotCubeObj;
    oattr.slotBaseTex = slotCubeBasePng;
    oattr.slotNormalTex = slotCubeNormalPng;
    slotCubeComp = GetScene()->CreateComponent(slotMainGroup, oattr);

    oattr.name = "Floor";
    oattr.ctrl = nullptr; // Set the default controller
    oattr.from = CRC_OBJECT_FROM_OBJ;
    oattr.slotRc = slotFloorObj;
    oattr.slotBaseTex = slotFloorBasePng;
    oattr.slotNormalTex = slotFloorNormalPng;
    slotFloorComp = GetScene()->CreateComponent(slotMainGroup, oattr);

    // Create UI group
    gattr.name = "UI";
    gattr.active = TRUE;
    slotUIGroup = GetScene()->CreateGroup(gattr);

    // Create UI components
    CRC::UIATTR uattr;
    uattr.name = "RectBackground";
    uattr.ctrl = nullptr; // Set the default controller
    uattr.slotResource = slotRectBackgroud;
    slotRectBackgroudComp = GetScene()->CreateComponent(slotUIGroup, uattr);

    uattr.name = "RectRed";
    uattr.ctrl = nullptr; // Set the default controller
    uattr.slotResource = slotRectRed;
    slotRectRedComp = GetScene()->CreateComponent(slotUIGroup, uattr);

    uattr.name = "RectBlue";
    uattr.ctrl = nullptr; // Set the default controller
    uattr.slotResource = slotRectBlue;
    slotRectBlueComp = GetScene()->CreateComponent(slotUIGroup, uattr);

    return CRC_SCENE_STATE_EXECUTING;
}

CRC_SCENE_STATE ExampleSceneMani::Update()
{
    if (GetInput()->IsKeyDown(CRC_KEY_MSG_ESCAPE))
    {
        OutputDebugStringA("Example Scene Escape\n");

        return CRC_SCENE_STATE_CLOSE;
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
        // OutputDebugStringA("LBTN\n");
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

CRC_SCENE_STATE ExampleSceneMani::End()
{
    // Unload resources
    GetScene()->UnLoadResources();

    // Get the resource factory instance
    CRC::ResourceFactory* rf = CRC::CuRendCore::GetInstance()->resourceFc;

    // Destroy resources
    rf->DestroyResource(slotCubeObj);
    rf->DestroyResource(slotFloorObj);
    rf->DestroyResource(slotCubeBasePng);
    rf->DestroyResource(slotCubeNormalPng);
    rf->DestroyResource(slotFloorBasePng);
    rf->DestroyResource(slotFloorNormalPng);

    rf->DestroyResource(slotRectBackgroud);
    rf->DestroyResource(slotRectRed);
    rf->DestroyResource(slotRectBlue);

    return CRC_SCENE_STATE_DESTROY;
}

CRC_SCENE_STATE ExampleSceneMani::ReStart()
{
    return CRC_SCENE_STATE_EXECUTING;
}
