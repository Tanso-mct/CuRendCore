#include "pch.h"
#include "CuRendCore/include/CuRendCore.h"


#include <thread>
#include <chrono>

static LRESULT CALLBACK WindowProc_DeviceTest(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
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

TEST(CuRendCore_device_test, CreateAndShowWindow) 
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"CreateAndShowWindow";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"CreateAndShowWindow";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        EXPECT_EQ(hr, S_OK);
    }
}

TEST(CuRendCore_device_test, CreateD3D11DeviceAndSwapChain) 
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"CreateD3D11DeviceAndSwapChain";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"CreateD3D11DeviceAndSwapChain";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        ASSERT_EQ(hr, S_OK);

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
        dxgiDesc.OutputWindow = window()->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, d3d11Device, d3d11SwapChain);

        EXPECT_NE(d3d11Device.Get(), nullptr);
        EXPECT_NE(d3d11SwapChain.Get(), nullptr);
    }
}

TEST(CuRendCore_device_test, CreateCRCDevice)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"CreateCRCDevice";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"CreateCRCDevice";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        ASSERT_EQ(hr, S_OK);

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
        dxgiDesc.OutputWindow = window()->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, d3d11Device, d3d11SwapChain);
    }

    std::unique_ptr<WACore::IContainable> crcDevice;
    {
        CRC_DEVICE_DESC desc(d3d11Device);

        CRCDeviceFactoryL0_0 deviceFactory;
        crcDevice = deviceFactory.Create(desc);
    }

    EXPECT_NE(crcDevice.get(), nullptr);
}

TEST(CuRendCore_device_test, CreateCRCID3D11Device)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"CreateCRCID3D11Device";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"CreateCRCID3D11Device";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        ASSERT_EQ(hr, S_OK);

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
        dxgiDesc.OutputWindow = window()->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, d3d11Device, d3d11SwapChain);
    }

    std::unique_ptr<WACore::IContainable> crcDevice;
    {
        CRC_DEVICE_DESC desc(d3d11Device);
        desc.renderMode_ = CRC_RENDER_MODE::D3D11;

        CRCDeviceFactoryL0_0 deviceFactory;
        crcDevice = deviceFactory.Create(desc);
    }

    EXPECT_NE(crcDevice.get(), nullptr);
}

TEST(CuRendCore_device_test, CreateCRCSwapChain)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"CreateCRCSwapChain";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"CreateCRCSwapChain";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        ASSERT_EQ(hr, S_OK);

        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 3;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = window()->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, d3d11Device, d3d11SwapChain);
    }

    d3d11SwapChain->GetDesc(&swapChainDesc.GetDxgiDesc());

    // Create CRC swap chain.
    std::unique_ptr<WACore::IContainable> swapChainAttr;
    {
        CRCSwapChainFactoryL0_0 swapChainFactory;
        swapChainAttr = swapChainFactory.Create(swapChainDesc);
    }

    EXPECT_NE(swapChainAttr.get(), nullptr);
}

TEST(CuRendCore_device_test, CreateCRCID3D11SwapChain)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"CreateCRCID3D11SwapChain";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"CreateCRCID3D11SwapChain";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        ASSERT_EQ(hr, S_OK);

        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 3;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = window()->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, d3d11Device, d3d11SwapChain);
    }

    d3d11SwapChain->GetDesc(&swapChainDesc.GetDxgiDesc());

    // Create CRC swap chain.
    std::unique_ptr<WACore::IContainable> swapChainAttr;
    {
        CRCIDXGISwapChainFactoryL0_0 swapChainFactory;
        swapChainAttr = swapChainFactory.Create(swapChainDesc);
    }

    EXPECT_NE(swapChainAttr.get(), nullptr);
}

TEST(CuRendCore_device_test, CreateCRCDeviceAndSwapChain)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"CreateCRCDeviceAndSwapChain";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"CreateCRCDeviceAndSwapChain";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        ASSERT_EQ(hr, S_OK);

        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 3;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = window()->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, d3d11Device, d3d11SwapChain);
    }

    // Create CRC d3d11Device and swap chain.
    std::unique_ptr<WACore::IContainable> crcDevice;
    std::unique_ptr<WACore::IContainable> crcSwapChain;
    CRC_DEVICE_DESC deviceDesc(d3d11Device);
    {
        CRCDeviceFactoryL0_0 deviceFactory;
        CRCSwapChainFactoryL0_0 swapChainFactory;
        CRC::CreateCRCDeviceAndSwapChain
        (
            deviceDesc, swapChainDesc, deviceFactory, swapChainFactory, crcDevice, crcSwapChain
        );
    }

    EXPECT_NE(crcDevice.get(), nullptr);
    EXPECT_NE(crcSwapChain.get(), nullptr);
}

TEST(CuRendCore_device_test, CreateCRCID3D11DeviceAndSwapChain)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"CreateCRCID3D11DeviceAndSwapChain";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"CreateCRCID3D11DeviceAndSwapChain";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        ASSERT_EQ(hr, S_OK);

        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 3;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = window()->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, d3d11Device, d3d11SwapChain);
    }

    // Create CRC d3d11Device and swap chain.
    std::unique_ptr<WACore::IContainable> crcDevice;
    std::unique_ptr<WACore::IContainable> crcSwapChain;
    {
        CRC_DEVICE_DESC deviceDesc(d3d11Device);
        deviceDesc.renderMode_ = CRC_RENDER_MODE::D3D11;

        CRCDeviceFactoryL0_0 deviceFactory;
        CRCIDXGISwapChainFactoryL0_0 swapChainFactory;
        CRC::CreateCRCDeviceAndSwapChain
        (
            deviceDesc, swapChainDesc, deviceFactory, swapChainFactory, crcDevice, crcSwapChain
        );
    }

    EXPECT_NE(crcDevice.get(), nullptr);
    EXPECT_NE(crcSwapChain.get(), nullptr);
}

TEST(CuRendCore_device_test, GetSwapChainBuffer)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"GetSwapChainBuffer";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"GetSwapChainBuffer";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        ASSERT_EQ(hr, S_OK);

        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 3;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = window()->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, d3d11Device, d3d11SwapChain);
    }

    d3d11SwapChain->GetDesc(&swapChainDesc.GetDxgiDesc());

    // Create CRC swap chain.
    std::unique_ptr<WACore::IContainable> crcSwapChain;
    {
        CRCSwapChainFactoryL0_0 swapChainFactory;
        crcSwapChain = swapChainFactory.Create(swapChainDesc);
    }

    ICRCTexture2D* backBuffer = nullptr;
    {
        WACore::RevertCast<CRCSwapChain, WACore::IContainable> swapChain(crcSwapChain);
        swapChain()->GetBuffer(0, backBuffer);
    }

    EXPECT_NE(backBuffer, nullptr);
}

TEST(CuRendCore_device_test, GetD3D11SwapChainBuffer)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"GetD3D11SwapChainBuffer";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"GetD3D11SwapChainBuffer";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        ASSERT_EQ(hr, S_OK);

        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 3;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = window()->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, d3d11Device, d3d11SwapChain);
    }

    d3d11SwapChain->GetDesc(&swapChainDesc.GetDxgiDesc());

    // Create CRC swap chain.
    std::unique_ptr<WACore::IContainable> crcSwapChain;
    {
        CRCIDXGISwapChainFactoryL0_0 swapChainFactory;
        crcSwapChain = swapChainFactory.Create(swapChainDesc);
    }

    ICRCTexture2D* backBuffer = nullptr;
    {
        WACore::RevertCast<CRCIDXGISwapChain, WACore::IContainable> swapChain(crcSwapChain);
        swapChain()->GetBuffer(0, backBuffer);
    }

    EXPECT_NE(backBuffer, nullptr);
}

TEST(CuRendCore_device_test, PresentSwapChain2)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"PresentSwapChain2";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"PresentSwapChain2";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        ASSERT_EQ(hr, S_OK);

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
        dxgiDesc.OutputWindow = window()->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, d3d11Device, d3d11SwapChain);
    }

    d3d11SwapChain->GetDesc(&swapChainDesc.GetDxgiDesc());

    // Create CRC swap chain.
    std::unique_ptr<WACore::IContainable> crcSwapChain;
    {
        CRCSwapChainFactoryL0_0 swapChainFactory;
        crcSwapChain = swapChainFactory.Create(swapChainDesc);
    }

    ICRCTexture2D* backBuffer = nullptr;
    {
        WACore::RevertCast<CRCSwapChain, WACore::IContainable> swapChain(crcSwapChain);
        swapChain()->GetBuffer(0, backBuffer);

        HRESULT hr = swapChain()->Present(0, 0);
        EXPECT_EQ(hr, S_OK);
    }
}

TEST(CuRendCore_device_test, PresentSwapChain3)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"PresentSwapChain3";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"PresentSwapChain3";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        ASSERT_EQ(hr, S_OK);

        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 3;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = window()->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, d3d11Device, d3d11SwapChain);
    }

    d3d11SwapChain->GetDesc(&swapChainDesc.GetDxgiDesc());

    // Create CRC swap chain.
    std::unique_ptr<WACore::IContainable> crcSwapChain;
    {
        CRCSwapChainFactoryL0_0 swapChainFactory;
        crcSwapChain = swapChainFactory.Create(swapChainDesc);
    }

    ICRCTexture2D* backBuffer = nullptr;
    {
        WACore::RevertCast<CRCSwapChain, WACore::IContainable> swapChain(crcSwapChain);
        swapChain()->GetBuffer(0, backBuffer);

        HRESULT hr = swapChain()->Present(0, 0);
        EXPECT_EQ(hr, S_OK);

        hr = swapChain()->Present(0, 0);
        EXPECT_EQ(hr, S_OK);
    }
}

TEST(CuRendCore_device_test, PresentD3D11SwapChain2)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"PresentD3D11SwapChain2";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"PresentD3D11SwapChain2";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        ASSERT_EQ(hr, S_OK);

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
        dxgiDesc.OutputWindow = window()->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, d3d11Device, d3d11SwapChain);
    }

    d3d11SwapChain->GetDesc(&swapChainDesc.GetDxgiDesc());

    // Create CRC swap chain.
    std::unique_ptr<WACore::IContainable> crcSwapChain;
    {
        CRCIDXGISwapChainFactoryL0_0 swapChainFactory;
        crcSwapChain = swapChainFactory.Create(swapChainDesc);
    }

    ICRCTexture2D* backBuffer = nullptr;
    {
        WACore::RevertCast<CRCIDXGISwapChain, WACore::IContainable> swapChain(crcSwapChain);
        swapChain()->GetBuffer(0, backBuffer);

        HRESULT hr = swapChain()->Present(0, 0);
        EXPECT_EQ(hr, S_OK);
    }
}

TEST(CuRendCore_device_test, PresentD3D11SwapChain3)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"PresentD3D11SwapChain3";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"PresentD3D11SwapChain3";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        ASSERT_EQ(hr, S_OK);

        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 3;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = window()->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, d3d11Device, d3d11SwapChain);
    }

    d3d11SwapChain->GetDesc(&swapChainDesc.GetDxgiDesc());

    // Create CRC swap chain.
    std::unique_ptr<WACore::IContainable> crcSwapChain;
    {
        CRCIDXGISwapChainFactoryL0_0 swapChainFactory;
        crcSwapChain = swapChainFactory.Create(swapChainDesc);
    }

    ICRCTexture2D* backBuffer = nullptr;
    {
        WACore::RevertCast<CRCIDXGISwapChain, WACore::IContainable> swapChain(crcSwapChain);
        swapChain()->GetBuffer(0, backBuffer);

        HRESULT hr = swapChain()->Present(0, 0);
        EXPECT_EQ(hr, S_OK);
    }
}

TEST(CuRendCore_device_test, ResizeSwapChain)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"ResizeSwapChain";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"ResizeSwapChain";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        ASSERT_EQ(hr, S_OK);

        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 3;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = window()->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, d3d11Device, d3d11SwapChain);
    }

    d3d11SwapChain->GetDesc(&swapChainDesc.GetDxgiDesc());

    // Create CRC swap chain.
    std::unique_ptr<WACore::IContainable> crcSwapChain;
    {
        CRCSwapChainFactoryL0_0 swapChainFactory;
        crcSwapChain = swapChainFactory.Create(swapChainDesc);
    }

    ICRCTexture2D* backBuffer = nullptr;
    {
        WACore::RevertCast<CRCSwapChain, WACore::IContainable> swapChain(crcSwapChain);
        swapChain()->GetBuffer(0, backBuffer);

        HRESULT hr = swapChain()->ResizeBuffers(3, 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
        EXPECT_EQ(hr, S_OK);
    }
}

TEST(CuRendCore_device_test, ResizeD3D11SwapChain)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    CRC_WINDOW_DESC desc = {};
    desc.wcex_.lpszClassName = L"ResizeD3D11SwapChain";
    desc.wcex_.lpfnWndProc = WindowProc_DeviceTest;
    desc.name_ = L"ResizeD3D11SwapChain";
    desc.hInstance = GetModuleHandle(NULL);
    std::unique_ptr<WACore::IContainable> windowAttr = windowFactory.Create(desc);

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        WACore::RevertCast<CRCWindowAttr, WACore::IContainable> window(windowAttr);
        ASSERT_NE(window(), nullptr);

        // Show window.
        HRESULT hr = CRC::ShowWindowCRC(window()->hWnd_);
        ASSERT_EQ(hr, S_OK);

        DXGI_SWAP_CHAIN_DESC& dxgiDesc = swapChainDesc.GetDxgiDesc();
        ZeroMemory(&dxgiDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
        dxgiDesc.BufferCount = 3;
        dxgiDesc.BufferDesc.Width = 0;
        dxgiDesc.BufferDesc.Height = 0;
        dxgiDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        dxgiDesc.BufferDesc.RefreshRate.Numerator = 60;
        dxgiDesc.BufferDesc.RefreshRate.Denominator = 1;
        dxgiDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        dxgiDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        dxgiDesc.OutputWindow = window()->hWnd_;
        dxgiDesc.SampleDesc.Count = 1;
        dxgiDesc.SampleDesc.Quality = 0;
        dxgiDesc.Windowed = TRUE;
        dxgiDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

        CRC::CreateD3D11DeviceAndSwapChain(swapChainDesc, d3d11Device, d3d11SwapChain);
    }

    d3d11SwapChain->GetDesc(&swapChainDesc.GetDxgiDesc());

    // Create CRC swap chain.
    std::unique_ptr<WACore::IContainable> crcSwapChain;
    {
        CRCIDXGISwapChainFactoryL0_0 swapChainFactory;
        crcSwapChain = swapChainFactory.Create(swapChainDesc);
    }

    ICRCTexture2D* backBuffer = nullptr;
    {
        WACore::RevertCast<CRCIDXGISwapChain, WACore::IContainable> swapChain(crcSwapChain);
        swapChain()->GetBuffer(0, backBuffer);

        WACore::As<CRCID3D11Texture2D>(backBuffer)->Get()->Release();

        HRESULT hr = swapChain()->ResizeBuffers(3, 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, 0); 
        EXPECT_EQ(hr, S_OK);
    }
}