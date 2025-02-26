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
    CRC_SWAP_CHAIN_DESC swapChainDesc(swapChain);
    {
        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 2;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, device, swapChain);
    }

    EXPECT_NE(device.Get(), nullptr);
    EXPECT_NE(swapChain.Get(), nullptr);
}

TEST(CuRendCore, CreateCRCDevice)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"CreateCRCDevice";
    desc.wcex_.lpfnWndProc = WindowProc;
    desc.name_ = L"CreateCRCDevice";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<ICRCContainable> windowAttr = windowFactory.Create(desc);

    // Show window.
    HRESULT hr = CRC::ShowWindowCRC(CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_);

    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> swapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(swapChain);
    {
        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 2;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, device, swapChain);
    }

    std::unique_ptr<ICRCContainable> crcDevice;
    {
        CRC_DEVICE_DESC desc(device);

        CRCDeviceFactoryL0_0 deviceFactory;
        crcDevice = deviceFactory.Create(desc);
    }

    EXPECT_NE(crcDevice.get(), nullptr);
}

TEST(CuRendCore, CreateCRCSwapChain)
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
    CRC_SWAP_CHAIN_DESC swapChainDesc(swapChain);
    {
        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 2;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, device, swapChain);
    }

    // Create CRC swap chain.
    std::unique_ptr<ICRCContainable> swapChainAttr;
    {
        CRCSwapChainFactoryL0_0 swapChainFactory;
        swapChainAttr = swapChainFactory.Create(swapChainDesc);
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
    CRC_SWAP_CHAIN_DESC swapChainDesc(swapChain);
    {
        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 2;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, device, swapChain);
    }

    // Create CRC swap chain.
    std::unique_ptr<ICRCContainable> crcSwapChain;
    {
        CRCSwapChainFactoryL0_0 swapChainFactory;
        crcSwapChain = swapChainFactory.Create(swapChainDesc);
    }

    ICRCTexture2D* backBuffer = nullptr;
    CRC::As<CRCSwapChain>(crcSwapChain.get())->GetBuffer(0, backBuffer);

    EXPECT_NE(backBuffer, nullptr);
}

TEST(CuRendCore, PresentSwapChain)
{
    // Create window attributes.
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"PresentSwapChain";
        desc.wcex_.lpfnWndProc = WindowProc;
        desc.name_ = L"PresentSwapChain";
        desc.hInstance = GetModuleHandle(NULL);

        CRCWindowFactory windowFactory;
        windowAttr = windowFactory.Create(desc);
    }
    // Show window.
    HRESULT hr = CRC::ShowWindowCRC(CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_);

    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> swapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(swapChain);
    {
        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 2;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, device, swapChain);
    }

    // Create CRC swap chain.
    std::unique_ptr<ICRCContainable> crcSwapChain;
    {
        CRCSwapChainFactoryL0_0 swapChainFactory;
        crcSwapChain = swapChainFactory.Create(swapChainDesc);
    }

    ICRCTexture2D* backBuffer = nullptr;
    CRC::As<CRCSwapChain>(crcSwapChain.get())->GetBuffer(0, backBuffer);

    hr = CRC::As<CRCSwapChain>(crcSwapChain.get())->Present(0, 0);

    EXPECT_EQ(hr, S_OK);
}

TEST(CuRendCore, ResizeSwapChain)
{
    // Create window attributes.
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"ResizeSwapChain";
        desc.wcex_.lpfnWndProc = WindowProc;
        desc.name_ = L"ResizeSwapChain";
        desc.hInstance = GetModuleHandle(NULL);

        CRCWindowFactory windowFactory;
        windowAttr = windowFactory.Create(desc);
    }
    // Show window.
    HRESULT hr = CRC::ShowWindowCRC(CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_);

    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> swapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(swapChain);
    {
        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 2;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = CRC::As<CRCWindowAttr>(windowAttr.get())->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, device, swapChain);
    }

    // Create CRC swap chain.
    std::unique_ptr<ICRCContainable> crcSwapChain;
    {
        CRCSwapChainFactoryL0_0 swapChainFactory;
        crcSwapChain = swapChainFactory.Create(swapChainDesc);
    }

    ICRCTexture2D* backBuffer = nullptr;
    CRC::As<CRCSwapChain>(crcSwapChain.get())->GetBuffer(0, backBuffer);

    hr = CRC::As<CRCSwapChain>(crcSwapChain.get())->ResizeBuffers(2, 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, 0);

    EXPECT_EQ(hr, S_OK);
}