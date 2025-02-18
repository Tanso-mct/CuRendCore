#pragma once

#include "CRC_config.h"
#include "CRC_container.h"

#include <Windows.h>
#include <memory>
#include <vector>

#include <d3d11.h>
#include <wrl/client.h>

struct CRC_API CRCWindowSrc
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

class CRC_API CRCWindowAttr : public ICRCContainable
{
public:
    virtual ~CRCWindowAttr() override = default;
    std::unique_ptr<CRCWindowSrc> src_ = nullptr;

    HWND hWnd_ = nullptr;

    Microsoft::WRL::ComPtr<ID3D11Device> device_ = nullptr;
    Microsoft::WRL::ComPtr<IDXGISwapChain> swapChain_ = nullptr;
};