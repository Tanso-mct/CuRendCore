#include "pch.h"
#include "CuRendCore/include/CuRendCore.h"


static LRESULT CALLBACK WindowProc_ResourceTest(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
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

TEST(CuRendCore_resource_test, CreateBuffer)
{
    Microsoft::WRL::ComPtr<ID3D11Device> device;

    CRCBufferFactoryL0_0 factory;
    CRC_BUFFER_DESC desc(device);

    D3D11_BUFFER_DESC& bufferDesc = desc.desc_;
    bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
    bufferDesc.ByteWidth = 1024;
    bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bufferDesc.MiscFlags = 0;
    bufferDesc.StructureByteStride = 0;

    std::unique_ptr<WACore::IContainable> buffer = factory.Create(desc);

    EXPECT_NE(buffer.get(), nullptr);
}

TEST(CuRendCore_resource_test, CreateID3D11Buffer)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    std::unique_ptr<WACore::IContainable> windowAttr;
    {
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"CreateID3D11Buffer";
        desc.wcex_.lpfnWndProc = WindowProc_ResourceTest;
        desc.name_ = L"CreateID3D11Buffer";
        desc.hInstance = GetModuleHandle(NULL);
        windowAttr = windowFactory.Create(desc);
    }

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

        ASSERT_NE(d3d11Device.Get(), nullptr);
        ASSERT_NE(d3d11SwapChain.Get(), nullptr);
    }

    std::unique_ptr<WACore::IContainable> buffer;
    {
        CRCID3D11BufferFactoryL0_0 factory;
        CRC_BUFFER_DESC desc(d3d11Device);

        D3D11_BUFFER_DESC& bufferDesc = desc.desc_;
        bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
        bufferDesc.ByteWidth = 1024;
        bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        bufferDesc.MiscFlags = 0;
        bufferDesc.StructureByteStride = 0;

        buffer = factory.Create(desc);
    }

    EXPECT_NE(buffer.get(), nullptr);
}

TEST(CuRendCore_resource_test, CreateTexture2D)
{
    Microsoft::WRL::ComPtr<ID3D11Device> device;

    CRCTexture2DFactoryL0_0 factory;
    CRC_TEXTURE2D_DESC desc(device);

    D3D11_TEXTURE2D_DESC& textureDesc = desc.desc_;
    textureDesc.Width = 1920;
    textureDesc.Height = 1080;
    textureDesc.MipLevels = 1;
    textureDesc.ArraySize = 1;
    textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Usage = D3D11_USAGE_DEFAULT;
    textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    std::unique_ptr<WACore::IContainable> texture = factory.Create(desc);

    EXPECT_NE(texture.get(), nullptr);
}

TEST(CuRendCore_resource_test, CreateID3D11Texture2D)
{
    // Create window factory.
    CRCWindowFactory windowFactory;

    // Create window attributes.
    std::unique_ptr<WACore::IContainable> windowAttr;
    {
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"CreateID3D11Texture2D";
        desc.wcex_.lpfnWndProc = WindowProc_ResourceTest;
        desc.name_ = L"CreateID3D11Texture2D";
        desc.hInstance = GetModuleHandle(NULL);
        windowAttr = windowFactory.Create(desc);
    }

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

        ASSERT_NE(d3d11Device.Get(), nullptr);
        ASSERT_NE(d3d11SwapChain.Get(), nullptr);
    }

    std::unique_ptr<WACore::IContainable> texture;
    {
        CRCID3D11Texture2DFactoryL0_0 factory;
        CRC_TEXTURE2D_DESC desc(d3d11Device);

        D3D11_TEXTURE2D_DESC& textureDesc = desc.desc_;
        textureDesc.Width = 1920;
        textureDesc.Height = 1080;
        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.Usage = D3D11_USAGE_DEFAULT;
        textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        texture = factory.Create(desc);
    }

    EXPECT_NE(texture.get(), nullptr);
}
