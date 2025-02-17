#pragma once

#include "CuRendCore.h"

class MainWindowListener : public ICRCWinMsgEvent
{
public:
    MainWindowListener() = default;
    ~MainWindowListener() override = default;

    void OnUpdate(UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnSize(UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnDestroy(UINT msg, WPARAM wParam, LPARAM lParam) override;
};