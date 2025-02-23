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

TEST(CuRendCore, CreateD3D11DeviceAndSwapChain) 
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"CreateD3D11DeviceAndSwapChain";
    desc.wcex_.lpfnWndProc = WindowProc;
    desc.name_ = L"CreateD3D11DeviceAndSwapChain";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<ICRCContainable> windowAttr = windowFactory.Create(desc);

    // Show window.
    HRESULT hr = CRC::ShowWindowCRC(CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_);

    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> swapChain;

    CRC::CreateD3D11DeviceAndSwapChain(CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_, device, swapChain);

    EXPECT_NE(device.Get(), nullptr);
    EXPECT_NE(swapChain.Get(), nullptr);
}

TEST(CuRendCor, CreateCRCSwapChain)
{
    // Create window attributes.
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"CreateCRCSwapChain";
        desc.wcex_.lpfnWndProc = WindowProc;
        desc.name_ = L"CreateCRCSwapChain";
        desc.hInstance = GetModuleHandle(NULL);

        CRCWindowFactory windowFactory;
        windowAttr = windowFactory.Create(desc);
    }
    // Show window.
    HRESULT hr = CRC::ShowWindowCRC(CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_);

    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> swapChain;

    CRC::CreateD3D11DeviceAndSwapChain(CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_, device, swapChain);

    // Create CRC swap chain.
    std::unique_ptr<ICRCContainable> swapChainAttr;
    {
        CRC_SWAP_CHAIN_DESC desc(swapChain);
        desc.BufferCount() = 2;
        desc.BufferUsage() = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.RefreshRate().Numerator = 60;
        desc.RefreshRate().Denominator = 1;
        desc.SwapEffect() = DXGI_SWAP_EFFECT_DISCARD;

        CRCSwapChainFactoryL0_0 swapChainFactory;
        swapChainAttr = swapChainFactory.Create(desc);
    }

    EXPECT_NE(swapChainAttr.get(), nullptr);
}

TEST(CuRendCore, GetSwapChainBuffer)
{
    // Create window attributes.
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"GetSwapChainBuffer";
        desc.wcex_.lpfnWndProc = WindowProc;
        desc.name_ = L"GetSwapChainBuffer";
        desc.hInstance = GetModuleHandle(NULL);

        CRCWindowFactory windowFactory;
        windowAttr = windowFactory.Create(desc);
    }
    // Show window.
    HRESULT hr = CRC::ShowWindowCRC(CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_);

    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> swapChain;

    CRC::CreateD3D11DeviceAndSwapChain(CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_, device, swapChain);

    // Create CRC swap chain.
    std::unique_ptr<ICRCContainable> crcSwapChain;
    {
        CRC_SWAP_CHAIN_DESC desc(swapChain);
        desc.BufferCount() = 2;
        desc.BufferUsage() = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.RefreshRate().Numerator = 60;
        desc.RefreshRate().Denominator = 1;
        desc.SwapEffect() = DXGI_SWAP_EFFECT_DISCARD;

        CRCSwapChainFactoryL0_0 swapChainFactory;
        crcSwapChain = swapChainFactory.Create(desc);
    }

    ICRCTexture2D* backBuffer = nullptr;
    CRC::As<CRCSwapChain>(crcSwapChain.get())->GetBuffer(0, backBuffer);

    EXPECT_NE(backBuffer, nullptr);
}

// TEST(CuRendCore, PresentSwapChain)
// {
//     // Create window attributes.
//     std::unique_ptr<ICRCContainable> windowAttr;
//     {
//         CRC_WINDOW_DESC desc = {};
//         desc.wcex_.lpszClassName = L"PresentSwapChain";
//         desc.wcex_.lpfnWndProc = WindowProc;
//         desc.name_ = L"PresentSwapChain";
//         desc.hInstance = GetModuleHandle(NULL);

//         CRCWindowFactory windowFactory;
//         windowAttr = windowFactory.Create(desc);
//     }
//     // Show window.
//     HRESULT hr = CRC::ShowWindowCRC(CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_);

//     Microsoft::WRL::ComPtr<ID3D11Device> device;
//     Microsoft::WRL::ComPtr<IDXGISwapChain> swapChain;

//     CRC::CreateD3D11DeviceAndSwapChain(CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_, device, swapChain);

//     // Create CRC swap chain.
//     std::unique_ptr<ICRCContainable> crcSwapChain;
//     {
//         CRC_SWAP_CHAIN_DESC desc(swapChain);
//         desc.BufferCount() = 2;
//         desc.BufferUsage() = DXGI_USAGE_RENDER_TARGET_OUTPUT;
//         desc.RefreshRate().Numerator = 60;
//         desc.RefreshRate().Denominator = 1;
//         desc.SwapEffect() = DXGI_SWAP_EFFECT_DISCARD;

//         CRCSwapChainFactoryL0_0 swapChainFactory;
//         crcSwapChain = swapChainFactory.Create(desc);
//     }

//     ICRCTexture2D* backBuffer = nullptr;
//     CRC::As<CRCSwapChain>(crcSwapChain.get())->GetBuffer(0, backBuffer);

//     CRC::As<CRCSwapChain>(crcSwapChain.get())->Present(0, 0);

//     EXPECT_NE(backBuffer, nullptr);
// }