#include "pch.h"
#include "CuRendCore/include/CuRendCore.h"

static LRESULT CALLBACK WindowProc_DeviceContextTest(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
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

TEST(CuRendCore_device_context, GetImmediateContext)
{
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        // Create window factory.
        CRCWindowFactory windowFactory;

        // Create window attributes.
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"GetImmediateContext";
        desc.wcex_.lpfnWndProc = WindowProc_DeviceContextTest;
        desc.name_ = L"GetImmediateContext";
        desc.hInstance = GetModuleHandle(NULL);
        windowAttr = windowFactory.Create(desc);
    }

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11SwapChain);
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
    }

    std::unique_ptr<ICRCContainable> crcDevice;
    {
        CRC_DEVICE_DESC desc(d3d11Device);

        CRCDeviceFactoryL0_0 deviceFactory;
        crcDevice = deviceFactory.Create(desc);

        ASSERT_NE(crcDevice.get(), nullptr);
    }

    {
        CRCTransCastUnique<ICRCDevice, ICRCContainable> device(crcDevice);
        ASSERT_NE(device(), nullptr);

        std::unique_ptr<ICRCDeviceContext>& immediateContext = device()->GetImmediateContext();
        ASSERT_NE(immediateContext, nullptr);
    }
}

TEST(CuRendCore_device_context, GetD3D11ImmediateContext)
{
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        // Create window factory.
        CRCWindowFactory windowFactory;

        // Create window attributes.
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"GetD3D11ImmediateContext";
        desc.wcex_.lpfnWndProc = WindowProc_DeviceContextTest;
        desc.name_ = L"GetD3D11ImmediateContext";
        desc.hInstance = GetModuleHandle(NULL);
        windowAttr = windowFactory.Create(desc);
    }

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11SwapChain);
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
    }

    std::unique_ptr<ICRCContainable> crcDevice;
    {
        CRC_DEVICE_DESC desc(d3d11Device);
        desc.renderMode_ = CRC_RENDER_MODE::D3D11;

        CRCDeviceFactoryL0_0 deviceFactory;
        crcDevice = deviceFactory.Create(desc);

        ASSERT_NE(crcDevice.get(), nullptr);
    }

    {
        CRCTransCastUnique<ICRCDevice, ICRCContainable> device(crcDevice);
        ASSERT_NE(device(), nullptr);

        std::unique_ptr<ICRCDeviceContext>& immediateContext = device()->GetImmediateContext();
        ASSERT_NE(immediateContext, nullptr);
    }
}

TEST(CuRendCore_device_context, ImmediateContextMapAndUnmap)
{
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        // Create window factory.
        CRCWindowFactory windowFactory;

        // Create window attributes.
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"ImmediateContextMapAndUnmap";
        desc.wcex_.lpfnWndProc = WindowProc_DeviceContextTest;
        desc.name_ = L"ImmediateContextMapAndUnmap";
        desc.hInstance = GetModuleHandle(NULL);
        windowAttr = windowFactory.Create(desc);
    }

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11SwapChain);
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
    }

    std::unique_ptr<ICRCContainable> crcDevice;
    {
        CRC_DEVICE_DESC desc(d3d11Device);

        CRCDeviceFactoryL0_0 deviceFactory;
        crcDevice = deviceFactory.Create(desc);

        ASSERT_NE(crcDevice.get(), nullptr);
    }

    std::unique_ptr<ICRCContainable> buffer;
    {
        CRCBufferFactoryL0_0 factory;
        CRC_BUFFER_DESC desc(d3d11Device);

        D3D11_BUFFER_DESC& bufferDesc = desc.desc_;
        bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
        bufferDesc.ByteWidth = 1024;
        bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        bufferDesc.MiscFlags = 0;
        bufferDesc.StructureByteStride = 0;

        buffer = factory.Create(desc);

        ASSERT_NE(buffer.get(), nullptr);
    }

    {
        CRCTransCastUnique<ICRCDevice, ICRCContainable> device(crcDevice);
        ASSERT_NE(device(), nullptr);

        std::unique_ptr<ICRCDeviceContext>& immediateContext = device()->GetImmediateContext();
        ASSERT_NE(immediateContext, nullptr);

        D3D11_MAPPED_SUBRESOURCE mappedResource = { 0 };
        immediateContext->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
        ASSERT_NE(mappedResource.pData, nullptr);

        // Cast mappedResource.pData or otherwise rewrite the data.

        immediateContext->Unmap(buffer, 0);
    }
}

TEST(CuRendCore_device_context, D3D11ImmediateContextMapAndUnmap)
{
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        // Create window factory.
        CRCWindowFactory windowFactory;

        // Create window attributes.
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"D3D11ImmediateContextMapAndUnmap";
        desc.wcex_.lpfnWndProc = WindowProc_DeviceContextTest;
        desc.name_ = L"D3D11ImmediateContextMapAndUnmap";
        desc.hInstance = GetModuleHandle(NULL);
        windowAttr = windowFactory.Create(desc);
    }

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11SwapChain);
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
    }

    std::unique_ptr<ICRCContainable> crcDevice;
    {
        CRC_DEVICE_DESC desc(d3d11Device);
        desc.renderMode_ = CRC_RENDER_MODE::D3D11;

        CRCDeviceFactoryL0_0 deviceFactory;
        crcDevice = deviceFactory.Create(desc);

        ASSERT_NE(crcDevice.get(), nullptr);
    }

    std::unique_ptr<ICRCContainable> buffer;
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

        ASSERT_NE(buffer.get(), nullptr);
    }

    {
        CRCTransCastUnique<ICRCDevice, ICRCContainable> device(crcDevice);
        ASSERT_NE(device(), nullptr);

        std::unique_ptr<ICRCDeviceContext>& immediateContext = device()->GetImmediateContext();
        ASSERT_NE(immediateContext, nullptr);

        D3D11_MAPPED_SUBRESOURCE mappedResource = { 0 };
        immediateContext->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
        ASSERT_NE(mappedResource.pData, nullptr);

        // Cast mappedResource.pData or otherwise rewrite the data.

        immediateContext->Unmap(buffer, 0);
    }
}

TEST(CuRendCore_device_context, ImmediateContextUpdateSubresource)
{
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        // Create window factory.
        CRCWindowFactory windowFactory;

        // Create window attributes.
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"ImmediateContextUpdateSubresource";
        desc.wcex_.lpfnWndProc = WindowProc_DeviceContextTest;
        desc.name_ = L"ImmediateContextUpdateSubresource";
        desc.hInstance = GetModuleHandle(NULL);
        windowAttr = windowFactory.Create(desc);
    }

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11SwapChain);
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
    }

    std::unique_ptr<ICRCContainable> crcDevice;
    {
        CRC_DEVICE_DESC desc(d3d11Device);

        CRCDeviceFactoryL0_0 deviceFactory;
        crcDevice = deviceFactory.Create(desc);

        ASSERT_NE(crcDevice.get(), nullptr);
    }

    std::unique_ptr<ICRCContainable> buffer;
    {
        CRCBufferFactoryL0_0 factory;
        CRC_BUFFER_DESC desc(d3d11Device);

        D3D11_BUFFER_DESC& bufferDesc = desc.desc_;
        bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
        bufferDesc.ByteWidth = 1024;
        bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        bufferDesc.MiscFlags = 0;
        bufferDesc.StructureByteStride = 0;

        buffer = factory.Create(desc);

        ASSERT_NE(buffer.get(), nullptr);
    }

    {
        CRCTransCastUnique<ICRCDevice, ICRCContainable> device(crcDevice);
        ASSERT_NE(device(), nullptr);

        std::unique_ptr<ICRCDeviceContext>& immediateContext = device()->GetImmediateContext();
        ASSERT_NE(immediateContext, nullptr);

        void* data = nullptr;
        UINT dataSize = 0;
        immediateContext->UpdateSubresource(buffer, data, dataSize);
    }
}

TEST(CuRendCore_device_context, D3D11ImmediateContextUpdateSubresource)
{
    std::unique_ptr<ICRCContainable> windowAttr;
    {
        // Create window factory.
        CRCWindowFactory windowFactory;

        // Create window attributes.
        CRC_WINDOW_DESC desc = {};
        desc.wcex_.lpszClassName = L"D3D11ImmediateContextUpdateSubresource";
        desc.wcex_.lpfnWndProc = WindowProc_DeviceContextTest;
        desc.name_ = L"D3D11ImmediateContextUpdateSubresource";
        desc.hInstance = GetModuleHandle(NULL);
        windowAttr = windowFactory.Create(desc);
    }

    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> d3d11SwapChain;
    CRC_SWAP_CHAIN_DESC swapChainDesc(d3d11SwapChain);
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
    }

    std::unique_ptr<ICRCContainable> crcDevice;
    {
        CRC_DEVICE_DESC desc(d3d11Device);
        desc.renderMode_ = CRC_RENDER_MODE::D3D11;

        CRCDeviceFactoryL0_0 deviceFactory;
        crcDevice = deviceFactory.Create(desc);

        ASSERT_NE(crcDevice.get(), nullptr);
    }

    std::unique_ptr<ICRCContainable> buffer;
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

        ASSERT_NE(buffer.get(), nullptr);
    }

    {
        CRCTransCastUnique<ICRCDevice, ICRCContainable> device(crcDevice);
        ASSERT_NE(device(), nullptr);

        std::unique_ptr<ICRCDeviceContext>& immediateContext = device()->GetImmediateContext();
        ASSERT_NE(immediateContext, nullptr);

        void* data = nullptr;
        UINT dataSize = 0;
        immediateContext->UpdateSubresource(buffer, data, dataSize);
    }
}

