#pragma once


#include "CuRendCore.h"

class ExampleSceneCtrl : public CRC::SceneController
{
private:
    CRC_SLOT slotCubeObj;
    CRC_SLOT slotFloorObj;
    CRC_SLOT slotCubeBasePng;
    CRC_SLOT slotCubeNormalPng;
    CRC_SLOT slotFloorBasePng;
    CRC_SLOT slotFloorNormalPng;

    CRC_SLOT slotWnd;
    std::weak_ptr<CRC::SceneController> exampleScene2Ctrl;
    
public:
    ExampleSceneCtrl();
    ~ExampleSceneCtrl() override;

    void SetSlotWnd(CRC_SLOT slotWnd) { this->slotWnd = slotWnd; }
    void SetExampleScene2Ctrl(std::weak_ptr<CRC::SceneController> exampleScene2Ctrl) { this->exampleScene2Ctrl = exampleScene2Ctrl; }

    void Init() override;
    void Update() override;
    void ReStart() override;
    void End() override;
};

class ExampleScene2Ctrl : public CRC::SceneController
{
private:
    CRC_SLOT slotRectBackgroud;
    CRC_SLOT slotRectRed;
    CRC_SLOT slotRectBlue;
    
public:
    ExampleScene2Ctrl();
    ~ExampleScene2Ctrl() override;

    void Init() override;
    void Update() override;
    void ReStart() override;
    void End() override;
};