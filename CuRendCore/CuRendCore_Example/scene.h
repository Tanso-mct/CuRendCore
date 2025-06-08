#pragma once

#include "CuRendCore/include/CuRendCore.h"

class MainSceneEvent : public WACore::IWPEvent
{
private:
    const int idMainWindow_ = CRC::ID_INVALID;
    const int idMainScene_ = CRC::ID_INVALID;
    const int idUserInput_ = CRC::ID_INVALID;

public:
    MainSceneEvent(int& idMainWindow, int& idMainScene, int& idUserInput);
    ~MainSceneEvent() override;

    void InputHandleExample(std::unique_ptr<CRCUserInputAttr>& input);

    void OnUpdate(std::unique_ptr<WACore::IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnSize(std::unique_ptr<WACore::IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnDestroy(std::unique_ptr<WACore::IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam) override;
};