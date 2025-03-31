#pragma once

#include "CuRendCore/include/CuRendCore.h"

class MainWindowEvent : public WACore::IWPEvent
{
private:
    const int idMainWindow_ = CRC::ID_INVALID;
    const int idMainScene_ = CRC::ID_INVALID;
    const int idUserInput_ = CRC::ID_INVALID;

public:
    MainWindowEvent(int& idMainWindow, int& idMainScene, int& idUserInput);
    ~MainWindowEvent() override;

    void OnUpdate(std::unique_ptr<WACore::IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnSize(std::unique_ptr<WACore::IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam) override;
    void OnDestroy(std::unique_ptr<WACore::IContainer>& container, UINT msg, WPARAM wParam, LPARAM lParam) override;
};