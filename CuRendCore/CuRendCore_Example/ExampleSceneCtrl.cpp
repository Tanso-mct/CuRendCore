#include "ExampleSceneCtrl.h"

ExampleSceneCtrl::ExampleSceneCtrl()
{
    // OutputDebugStringA("ExampleSceneCtrl::ExampleSceneCtrl()\n");
}

ExampleSceneCtrl::~ExampleSceneCtrl()
{
    // OutputDebugStringA("ExampleSceneCtrl::~ExampleSceneCtrl()\n");
}

void ExampleSceneCtrl::Init()
{
    // OutputDebugStringA("ExampleSceneCtrl::Init()\n");

    // Get the resource factory instance
    CRC::ResourceFactory* rf = CRC::ResourceFactory::GetInstance();

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
}

void ExampleSceneCtrl::ReStart()
{
    // OutputDebugStringA("ExampleSceneCtrl::ReStart()\n");
}

void ExampleSceneCtrl::End()
{
    // OutputDebugStringA("ExampleSceneCtrl::End()\n");

    // Remove resources from the scene controller
    RemoveResource(slotCubeNormalPng);

    // Unload resources
    UnLoadResources();

    NeedDestroy(true);
}
