#pragma once

#include "CRC_interface.h"

#include <Windows.h>
#include <memory>
#include <vector>
#include <mutex>

struct CRCWindowAttr;

class CRCWindowData : public CRCData
{
public:
    virtual ~CRCWindowData() = default;

    HWND hWnd_ = nullptr;
    int idScene_ = CRC::INVALID_ID;

    // If this flag is true, a window is created when a creative command is sent to the Window thread.
    bool needCreateFlag_ = true;

    // Used when creating a window. After creation, nullPtr.
    std::unique_ptr<CRCWindowAttr> src_ = nullptr;
};

struct CRCWindowAttr
{
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

    bool needCreateFlag_ = true;
};

class CRCWindowContainer : public CRCContainer
{
private:
    std::vector<std::unique_ptr<CRCWindowData>> data_;

public:
    virtual ~CRCWindowContainer() = default;
    std::mutex mtx;

    virtual int Add(std::unique_ptr<CRCData>& data) override;

    virtual std::unique_ptr<CRCData> Take(int id) override;
    virtual HRESULT Set(int id, std::unique_ptr<CRCData>& data) override;

    virtual UINT GetSize() override;

    virtual void Clear(int id) override;
    virtual void Clear() override;
};