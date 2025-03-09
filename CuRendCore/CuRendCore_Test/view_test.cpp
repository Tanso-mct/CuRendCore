#include "pch.h"
#include "CuRendCore/include/CuRendCore.h"

static LRESULT CALLBACK WindowProc_ViewTest(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
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

TEST(CuRendCore_view_test, CreateShaderResourceView)
{
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        // Create window factory.
        CRCWindowFactory windowFactory;

        // Create window attributes.
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"CreateShaderResourceView";
        desc.wcex_.lpfnWndProc = WindowProc_ViewTest;
        desc.name_ = L"CreateShaderResourceView";
        desc.hInstance = GetModuleHandle(NULL);
        windowAttr = windowFactory.Create(desc);
    }

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        CRCTransCastUnique<CRCWindowAttr, ICRCContainable> window(windowAttr);
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

    std::unique_ptr<ICRCContainable> texture;
    {
        CRCTexture2DFactoryL0_0 factory;
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
        texture = factory.Create(desc);

        ASSERT_NE(texture.get(), nullptr);
    }

    std::unique_ptr<ICRCContainable> shaderResourceView;
    {
        CRCShaderResourceViewFactoryL0_0 factory;
        CRC_SHADER_RESOURCE_VIEW_DESC desc(d3d11Device, texture);

        D3D11_SHADER_RESOURCE_VIEW_DESC& srvDesc = desc.desc_;
        srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MostDetailedMip = 0;
        srvDesc.Texture2D.MipLevels = 1;

        shaderResourceView = factory.Create(desc);
    }

    EXPECT_NE(shaderResourceView.get(), nullptr);
}

TEST(CuRendCore_view_test, CreateD3D11ShaderResourceView)
{
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        // Create window factory.
        CRCWindowFactory windowFactory;

        // Create window attributes.
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"CreateD3D11ShaderResourceView";
        desc.wcex_.lpfnWndProc = WindowProc_ViewTest;
        desc.name_ = L"CreateD3D11ShaderResourceView";
        desc.hInstance = GetModuleHandle(NULL);
        windowAttr = windowFactory.Create(desc);
    }

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        CRCTransCastUnique<CRCWindowAttr, ICRCContainable> window(windowAttr);
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

    std::unique_ptr<ICRCContainable> texture;
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

        texture = factory.Create(desc);

        ASSERT_NE(texture.get(), nullptr);
    }

    std::unique_ptr<ICRCContainable> shaderResourceView;
    {
        CRCID3D11ShaderResourceViewFactoryL0_0 factory;
        CRC_SHADER_RESOURCE_VIEW_DESC desc(d3d11Device, texture);

        D3D11_SHADER_RESOURCE_VIEW_DESC& srvDesc = desc.desc_;
        srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MostDetailedMip = 0;
        srvDesc.Texture2D.MipLevels = 1;

        shaderResourceView = factory.Create(desc);
    }

    EXPECT_NE(shaderResourceView.get(), nullptr);
}

TEST(CuRendCore_view_test, CreateRenderTargetView)
{
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        // Create window factory.
        CRCWindowFactory windowFactory;

        // Create window attributes.
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"CreateRenderTargetView";
        desc.wcex_.lpfnWndProc = WindowProc_ViewTest;
        desc.name_ = L"CreateRenderTargetView";
        desc.hInstance = GetModuleHandle(NULL);
        windowAttr = windowFactory.Create(desc);
    }

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        CRCTransCastUnique<CRCWindowAttr, ICRCContainable> window(windowAttr);
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

    std::unique_ptr<ICRCContainable> texture;
    {
        CRCTexture2DFactoryL0_0 factory;
        CRC_TEXTURE2D_DESC desc(d3d11Device);

        D3D11_TEXTURE2D_DESC& textureDesc = desc.desc_;
        textureDesc.Width = 1920;
        textureDesc.Height = 1080;
        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.Usage = D3D11_USAGE_DEFAULT;
        textureDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
        texture = factory.Create(desc);

        ASSERT_NE(texture.get(), nullptr);
    }

    std::unique_ptr<ICRCContainable> renderTargetView;
    {
        CRCRenderTargetViewFactoryL0_0 factory;
        CRC_RENDER_TARGET_VIEW_DESC desc(d3d11Device, texture);

        D3D11_RENDER_TARGET_VIEW_DESC& rtvDesc = desc.desc_;
        rtvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        rtvDesc.Texture2D.MipSlice = 0;

        renderTargetView = factory.Create(desc);
    }

    EXPECT_NE(renderTargetView.get(), nullptr);
}

TEST(CuRendCore_view_test, CreateD3D11RenderTargetView)
{
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        // Create window factory.
        CRCWindowFactory windowFactory;

        // Create window attributes.
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"CreateD3D11RenderTargetView";
        desc.wcex_.lpfnWndProc = WindowProc_ViewTest;
        desc.name_ = L"CreateD3D11RenderTargetView";
        desc.hInstance = GetModuleHandle(NULL);
        windowAttr = windowFactory.Create(desc);
    }

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        CRCTransCastUnique<CRCWindowAttr, ICRCContainable> window(windowAttr);
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

    std::unique_ptr<ICRCContainable> texture;
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
        textureDesc.BindFlags = D3D11_BIND_RENDER_TARGET;

        texture = factory.Create(desc);

        ASSERT_NE(texture.get(), nullptr);
    }

    std::unique_ptr<ICRCContainable> renderTargetView;
    {
        CRCID3D11RenderTargetViewFactoryL0_0 factory;
        CRC_RENDER_TARGET_VIEW_DESC desc(d3d11Device, texture);

        D3D11_RENDER_TARGET_VIEW_DESC& rtvDesc = desc.desc_;
        rtvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        rtvDesc.Texture2D.MipSlice = 0;

        renderTargetView = factory.Create(desc);
    }

    EXPECT_NE(renderTargetView.get(), nullptr);
}

TEST(CuRendCore_view_test, CreateDepthStencilView)
{
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        // Create window factory.
        CRCWindowFactory windowFactory;

        // Create window attributes.
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"CreateDepthStencilView";
        desc.wcex_.lpfnWndProc = WindowProc_ViewTest;
        desc.name_ = L"CreateDepthStencilView";
        desc.hInstance = GetModuleHandle(NULL);
        windowAttr = windowFactory.Create(desc);
    }

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        CRCTransCastUnique<CRCWindowAttr, ICRCContainable> window(windowAttr);
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

    std::unique_ptr<ICRCContainable> texture;
    {
        CRCTexture2DFactoryL0_0 factory;
        CRC_TEXTURE2D_DESC desc(d3d11Device);

        D3D11_TEXTURE2D_DESC& textureDesc = desc.desc_;
        textureDesc.Width = 1920;
        textureDesc.Height = 1080;
        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.Usage = D3D11_USAGE_DEFAULT;
        textureDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        texture = factory.Create(desc);

        ASSERT_NE(texture.get(), nullptr);
    }

    std::unique_ptr<ICRCContainable> depthStencilView;
    {
        CRCDepthStencilViewFactoryL0_0 factory;
        CRC_DEPTH_STENCIL_VIEW_DESC desc(d3d11Device, texture);

        D3D11_DEPTH_STENCIL_VIEW_DESC& dsvDesc = desc.desc_;
        dsvDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
        dsvDesc.Texture2D.MipSlice = 0;

        depthStencilView = factory.Create(desc);
    }

    EXPECT_NE(depthStencilView.get(), nullptr);
}

TEST(CuRendCore_view_test, CreateD3D11DepthStencilView)
{
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        // Create window factory.
        CRCWindowFactory windowFactory;

        // Create window attributes.
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"CreateD3D11DepthStencilView";
        desc.wcex_.lpfnWndProc = WindowProc_ViewTest;
        desc.name_ = L"CreateD3D11DepthStencilView";
        desc.hInstance = GetModuleHandle(NULL);
        windowAttr = windowFactory.Create(desc);
    }

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11Device, d3d11SwapChain);
    {
        CRCTransCastUnique<CRCWindowAttr, ICRCContainable> window(windowAttr);
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

    std::unique_ptr<ICRCContainable> texture;
    {
        CRCID3D11Texture2DFactoryL0_0 factory;
        CRC_TEXTURE2D_DESC desc(d3d11Device);

        D3D11_TEXTURE2D_DESC& textureDesc = desc.desc_;
        textureDesc.Width = 1920;
        textureDesc.Height = 1080;
        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.Usage = D3D11_USAGE_DEFAULT;
        textureDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;

        texture = factory.Create(desc);

        ASSERT_NE(texture.get(), nullptr);
    }

    std::unique_ptr<ICRCContainable> depthStencilView;
    {
        CRCID3D11DepthStencilViewFactoryL0_0 factory;
        CRC_DEPTH_STENCIL_VIEW_DESC desc(d3d11Device, texture);

        D3D11_DEPTH_STENCIL_VIEW_DESC& dsvDesc = desc.desc_;
        dsvDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
        dsvDesc.Texture2D.MipSlice = 0;

        depthStencilView = factory.Create(desc);
    }

    EXPECT_NE(depthStencilView.get(), nullptr);
}