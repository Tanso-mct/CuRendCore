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
    
public:
    ExampleSceneCtrl();
    ~ExampleSceneCtrl() override;

    void Init() override;
    void Update() override;
    void ReStart() override;
    void End() override;
};