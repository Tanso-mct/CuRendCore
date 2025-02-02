#pragma once

#include "CRC_config.h"
#include "CRC_interface.h"

#include <Windows.h>
#include <memory>
#include <vector>

struct CRC_API CRCWindowAttr
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

class CRC_API CRCWindowData : public CRCData
{
public:
    virtual ~CRCWindowData() override = default;

    HWND hWnd_ = nullptr;
    std::unique_ptr<CRCWindowAttr> src_ = nullptr;
};

class CRC_API CRCWindowContainer : public CRCContainer
{
private:
    std::vector<std::unique_ptr<CRCWindowData>> data_;

public:
    CRCWindowContainer() = default;
    virtual ~CRCWindowContainer() override = default;

    // Delete copy constructor and operator=.
    CRCWindowContainer(const CRCWindowContainer&) = delete;
    CRCWindowContainer& operator=(const CRCWindowContainer&) = delete;

    int Add(std::unique_ptr<CRCData>& data);
    HRESULT Remove(int id) override;

    CRCData* Get(int id);
    int GetSize();

    void Clear();
};