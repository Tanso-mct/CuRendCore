#pragma once

#include "CuRendCore.h"

class ExampleSceneMani : public CRC::SceneMani
{
private:
    CRC_SLOT slotCubeObj;
    CRC_SLOT slotFloorObj;
    CRC_SLOT slotCubeBasePng;
    CRC_SLOT slotCubeNormalPng;
    CRC_SLOT slotFloorBasePng;
    CRC_SLOT slotFloorNormalPng;

    CRC_SLOT slotRectBackgroud;
    CRC_SLOT slotRectRed;
    CRC_SLOT slotRectBlue;
protected:
    CRC_SCENE_STATE Start() override;
    CRC_SCENE_STATE Update() override;
    CRC_SCENE_STATE End() override;

    CRC_SCENE_STATE ReStart() override;

public:
    ExampleSceneMani() {CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};
    ~ExampleSceneMani() override {CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};
};