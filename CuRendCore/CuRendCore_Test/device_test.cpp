#include "pch.h"
#include "CuRendCore/include/CuRendCore.h"

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_PAINT:
        // Returning DefWindowProc with WM_PAINT will stop the updating, so WM_PAINT is returned 0.
        break;

    default:
        return DefWindowProc(hWnd, msg, wParam, lParam);
    }

    return 0;
}

TEST(CuRendCore, CreateAndShowWindow) 
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"CreateAndShowWindow";
    desc.wcex_.lpfnWndProc = WindowProc;
    desc.name_ = L"CreateAndShowWindow";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<ICRCContainable> windowAttr = windowFactory.Create(desc);

    // Show window.
    HRESULT hr = CRC::ShowWindowCRC(CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_);

    EXPECT_EQ(hr, S_OK);
}

TEST(CuRendCore, CreateDeviceAndSwapChain) 
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"CreateDeviceAndSwapChain";
    desc.wcex_.lpfnWndProc = WindowProc;
    desc.name_ = L"CreateDeviceAndSwapChain";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<ICRCContainable> windowAttr = windowFactory.Create(desc);

    // Show window.
    HRESULT hr = CRC::ShowWindowCRC(CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_);

    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> swapChain;

    CRC::CreateDeviceAndSwapChain(CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_, device, swapChain);

    EXPECT_NE(device.Get(), nullptr);
    EXPECT_NE(swapChain.Get(), nullptr);
}