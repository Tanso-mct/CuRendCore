#pragma once

#include "CuRendCore.h"

class MainWindowListener : public ICRCWinMsgListener
{
public:
    MainWindowListener() = default;
    ~MainWindowListener() override = default;

    void OnUpdate(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnMinimize(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnMaximize(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnRestored(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnDestroy(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam) override;
};