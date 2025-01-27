#pragma once

#include "CRC_interface.h"

#include <Windows.h>
#include <memory>
#include <vector>

class CRCWindowData : public CRCData
{
public:
    virtual ~CRCWindowData() = default;

    HWND hWnd_ = nullptr;
};

class CRCWindowAttr
{
public:
    WNDCLASSEX wcex_ = 
    {
        sizeof(WNDCLASSEX),
        CS_HREDRAW | CS_VREDRAW,
        nullptr,
        0,
        0,
        nullptr,
        LoadIcon(nullptr, IDI_APPLICATION),
        LoadCursor(nullptr, IDC_ARROW),
        (HBRUSH)(COLOR_WINDOW + 1),
        nullptr,
        L"Window",
        LoadIcon(nullptr, IDI_APPLICATION)
    };

    LPCWSTR name_ = L"Window";
    int initialPosX_ = CW_USEDEFAULT;
    int initialPosY_ = CW_USEDEFAULT;
    unsigned int width_ = 800;
    unsigned int height_ = 600;
    DWORD style_ = WS_OVERLAPPEDWINDOW;
    HWND hWndParent_ = NULL;
    HINSTANCE hInstance = nullptr;
};

class CRCWindowContainer : public CRCContainer
{
private:
    std::vector<std::unique_ptr<CRCWindowData>> data_;

public:
    virtual ~CRCWindowContainer() = default;

    virtual int Add(std::unique_ptr<CRCData>& data) override;
    virtual HRESULT Remove(int id) override;

    virtual std::unique_ptr<CRCData>& Get(int id) override;
    virtual int GetSize() override;

    virtual void Clear() override;
};