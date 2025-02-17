#pragma once

#include "CuRendCore.h"

class MainSceneListener : public ICRCWinMsgEvent
{
public:
    MainSceneListener() = default;
    ~MainSceneListener() override = default;

    void InputHandleExample();

    void OnUpdate(UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnSize(UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnDestroy(UINT msg, WPARAM wParam, LPARAM lParam) override;
};