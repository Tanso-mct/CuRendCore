#pragma once

#include "CuRendCore.h"

class MainSceneListener : public ICRCWinMsgListener
{
public:
    MainSceneListener() = default;
    ~MainSceneListener() override = default;

    void OnUpdate(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnDestroy(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam) override;
};