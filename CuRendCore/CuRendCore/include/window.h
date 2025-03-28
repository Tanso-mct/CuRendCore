#pragma once

#include "CuRendCore/include/config.h"
#include "CuRendCore/include/container.h"
#include "CuRendCore/include/factory.h"

#include <Windows.h>
#include <memory>
#include <vector>

#include <d3d11.h>
#include <wrl/client.h>

class CRC_API CRC_WINDOW_DESC : public IDESC
{
public:
    ~CRC_WINDOW_DESC() override = default;

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

class CRC_API CRCWindowFactory : public ICRCFactory
{
public:
    virtual ~CRCWindowFactory() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCWindowAttr : public ICRCContainable
{
public:
    virtual ~CRCWindowAttr() override = default;

    HWND hWnd_ = nullptr;

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device_ = nullptr;
    Microsoft::WRL::ComPtr<IDXGISwapChain> swapChain_ = nullptr;
};