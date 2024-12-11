#pragma once

#include "CuRendCore.h"

class ExampleSceneMani : public CRC::SceneMani
{
protected:
    CRC_SCENE_STATE Start() override;
    CRC_SCENE_STATE Update() override;
    CRC_SCENE_STATE Destroy() override;

    CRC_SCENE_STATE ReStart() override;

public:
    ExampleSceneMani() {CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};
    ~ExampleSceneMani() override {CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};
};