#pragma once

#include "CuRendCore.h"

class MainSceneListener : public ICRCWinMsgListener
{
private:
    std::unique_ptr<ICRCContainable>& input;

public:
    MainSceneListener(std::unique_ptr<ICRCContainable>& input) : input(input) {}
    ~MainSceneListener() override = default;

    void OnUpdate(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnSize(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnDestroy(ICRCContainable* attr, UINT msg, WPARAM wParam, LPARAM lParam) override;
};