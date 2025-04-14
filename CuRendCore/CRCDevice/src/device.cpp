#include "CRCDevice/include/device.h"
#include "CRCDevice/include/console.h"

CRC::Device::Device
(
    std::unique_ptr<IFactory> bufferFactory, 
    std::unique_ptr<IFactory> texture1dFactory, 
    std::unique_ptr<IFactory> texture2dFactory, 
    std::unique_ptr<IFactory> texture3dFactory, 
    std::unique_ptr<IFactory> shaderResourceViewFactory, 
    std::unique_ptr<IFactory> renderTargetViewFactory, 
    std::unique_ptr<IFactory> depthStencilViewFactory, 
    std::unique_ptr<IFactory> vertexShaderFactory, 
    std::unique_ptr<IFactory> pixelShaderFactory, 
    std::unique_ptr<IFactory> blendStateFactory, 
    std::unique_ptr<IFactory> depthStencilStateFactory, 
    std::unique_ptr<IFactory> rasterizerStateFactory, 
    std::unique_ptr<IFactory> samplerStateFactory
): 
    bufferFactory_(std::move(bufferFactory)), 
    texture1dFactory_(std::move(texture1dFactory)), 
    texture2dFactory_(std::move(texture2dFactory)), 
    texture3dFactory_(std::move(texture3dFactory)), 
    shaderResourceViewFactory_(std::move(shaderResourceViewFactory)), 
    renderTargetViewFactory_(std::move(renderTargetViewFactory)), 
    depthStencilViewFactory_(std::move(depthStencilViewFactory)), 
    vertexShaderFactory_(std::move(vertexShaderFactory)), 
    pixelShaderFactory_(std::move(pixelShaderFactory)), 
    blendStateFactory_(std::move(blendStateFactory)), 
    depthStencilStateFactory_(std::move(depthStencilStateFactory)), 
    rasterizerStateFactory_(std::move(rasterizerStateFactory)), 
    samplerStateFactory_(std::move(samplerStateFactory))
{}

HRESULT CRC::Device::CreateBuffer(CRC::IDesc &desc, std::unique_ptr<CRC::IResource> &buffer)
{
    // Create buffer using the factory.
    std::unique_ptr<CRC::IProduct> product = bufferFactory_->Create(desc);
    if (!product)
    {
        CRCDevice::CoutWrn({"Failed to create buffer."});
        return E_FAIL;
    }

    // Cast the product to IResource.
    std::unique_ptr<CRC::IResource> resource = WACore::UniqueAs<CRC::IResource>(product);
    if (!resource)
    {
        CRCDevice::CoutWrn({"Failed to cast product to IResource.", "Buffer factory may be out of order."});
        return E_FAIL;
    }

    buffer = std::move(resource);
    return S_OK;
}

HRESULT CRC::Device::CreateTexture1D(CRC::IDesc &desc, std::unique_ptr<CRC::IResource> &texture1d)
{
    // Create texture1d using the factory.
    std::unique_ptr<CRC::IProduct> product = texture1dFactory_->Create(desc);
    if (!product)
    {
        CRCDevice::CoutWrn({"Failed to create texture1d."});
        return E_FAIL;
    }

    // Cast the product to IResource.
    std::unique_ptr<CRC::IResource> resource = WACore::UniqueAs<CRC::IResource>(product);
    if (!resource)
    {
        CRCDevice::CoutWrn({"Failed to cast product to IResource.", "Texture1D factory may be out of order."});
        return E_FAIL;
    }

    texture1d = std::move(resource);
    return S_OK;
}

HRESULT CRC::Device::CreateTexture2D(CRC::IDesc &desc, std::unique_ptr<CRC::IResource> &texture2d)
{
    // Create texture2d using the factory.
    std::unique_ptr<CRC::IProduct> product = texture2dFactory_->Create(desc);
    if (!product)
    {
        CRCDevice::CoutWrn({"Failed to create texture2d."});
        return E_FAIL;
    }

    // Cast the product to IResource.
    std::unique_ptr<CRC::IResource> resource = WACore::UniqueAs<CRC::IResource>(product);
    if (!resource)
    {
        CRCDevice::CoutWrn({"Failed to cast product to IResource.", "Texture2D factory may be out of order."});
        return E_FAIL;
    }

    texture2d = std::move(resource);
    return S_OK;
}

HRESULT CRC::Device::CreateTexture3D(CRC::IDesc &desc, std::unique_ptr<CRC::IResource> &texture3d)
{
    // Create texture3d using the factory.
    std::unique_ptr<CRC::IProduct> product = texture3dFactory_->Create(desc);
    if (!product)
    {
        CRCDevice::CoutWrn({"Failed to create texture3d."});
        return E_FAIL;
    }

    // Cast the product to IResource.
    std::unique_ptr<CRC::IResource> resource = WACore::UniqueAs<CRC::IResource>(product);
    if (!resource)
    {
        CRCDevice::CoutWrn({"Failed to cast product to IResource.", "Texture3D factory may be out of order."});
        return E_FAIL;
    }

    texture3d = std::move(resource);
    return S_OK;
}

HRESULT CRC::Device::CreateShaderResourceView(CRC::IDesc &desc, std::unique_ptr<CRC::IView> &srv)
{
    // Create shader resource view using the factory.
    std::unique_ptr<CRC::IProduct> product = shaderResourceViewFactory_->Create(desc);
    if (!product)
    {
        CRCDevice::CoutWrn({"Failed to create shader resource view."});
        return E_FAIL;
    }

    // Cast the product to IView.
    std::unique_ptr<CRC::IView> view = WACore::UniqueAs<CRC::IView>(product);
    if (!view)
    {
        CRCDevice::CoutWrn({"Failed to cast product to IView.", "Shader resource view factory may be out of order."});
        return E_FAIL;
    }

    srv = std::move(view);
    return S_OK;
}

HRESULT CRC::Device::CreateRenderTargetView(CRC::IDesc &desc, std::unique_ptr<CRC::IView> &rtv)
{
    // Create render target view using the factory.
    std::unique_ptr<CRC::IProduct> product = renderTargetViewFactory_->Create(desc);
    if (!product)
    {
        CRCDevice::CoutWrn({"Failed to create render target view."});
        return E_FAIL;
    }

    // Cast the product to IView.
    std::unique_ptr<CRC::IView> view = WACore::UniqueAs<CRC::IView>(product);
    if (!view)
    {
        CRCDevice::CoutWrn({"Failed to cast product to IView.", "Render target view factory may be out of order."});
        return E_FAIL;
    }

    rtv = std::move(view);
    return S_OK;
}

HRESULT CRC::Device::CreateDepthStencilView(CRC::IDesc &desc, std::unique_ptr<CRC::IView> &dsv)
{
    // Create depth stencil view using the factory.
    std::unique_ptr<CRC::IProduct> product = depthStencilViewFactory_->Create(desc);
    if (!product)
    {
        CRCDevice::CoutWrn({"Failed to create depth stencil view."});
        return E_FAIL;
    }

    // Cast the product to IView.
    std::unique_ptr<CRC::IView> view = WACore::UniqueAs<CRC::IView>(product);
    if (!view)
    {
        CRCDevice::CoutWrn({"Failed to cast product to IView.", "Depth stencil view factory may be out of order."});
        return E_FAIL;
    }

    dsv = std::move(view);
    return S_OK;
}

HRESULT CRC::Device::CreateVertexShader(CRC::IDesc &desc, std::unique_ptr<CRC::IShader> &vertexShader)
{
    // Create vertex shader using the factory.
    std::unique_ptr<CRC::IProduct> product = vertexShaderFactory_->Create(desc);
    if (!product)
    {
        CRCDevice::CoutWrn({"Failed to create vertex shader."});
        return E_FAIL;
    }

    // Cast the product to IShader.
    std::unique_ptr<CRC::IShader> shader = WACore::UniqueAs<CRC::IShader>(product);
    if (!shader)
    {
        CRCDevice::CoutWrn({"Failed to cast product to IShader.", "Vertex shader factory may be out of order."});
        return E_FAIL;
    }

    vertexShader = std::move(shader);
    return S_OK;
}

HRESULT CRC::Device::CreatePixelShader(CRC::IDesc &desc, std::unique_ptr<CRC::IShader> &pixelShader)
{
    // Create pixel shader using the factory.
    std::unique_ptr<CRC::IProduct> product = pixelShaderFactory_->Create(desc);
    if (!product)
    {
        CRCDevice::CoutWrn({"Failed to create pixel shader."});
        return E_FAIL;
    }

    // Cast the product to IShader.
    std::unique_ptr<CRC::IShader> shader = WACore::UniqueAs<CRC::IShader>(product);
    if (!shader)
    {
        CRCDevice::CoutWrn({"Failed to cast product to IShader.", "Pixel shader factory may be out of order."});
        return E_FAIL;
    }

    pixelShader = std::move(shader);
    return S_OK;
}

HRESULT CRC::Device::CreateBlendState(CRC::IDesc &desc, std::unique_ptr<CRC::IState> &blendState)
{
    // Create blend state using the factory.
    std::unique_ptr<CRC::IProduct> product = blendStateFactory_->Create(desc);
    if (!product)
    {
        CRCDevice::CoutWrn({"Failed to create blend state."});
        return E_FAIL;
    }

    // Cast the product to IState.
    std::unique_ptr<CRC::IState> state = WACore::UniqueAs<CRC::IState>(product);
    if (!state)
    {
        CRCDevice::CoutWrn({"Failed to cast product to IState.", "Blend state factory may be out of order."});
        return E_FAIL;
    }

    blendState = std::move(state);
    return S_OK;
}

HRESULT CRC::Device::CreateDepthStencilState(CRC::IDesc &desc, std::unique_ptr<CRC::IState> &depthStencilState)
{
    // Create depth stencil state using the factory.
    std::unique_ptr<CRC::IProduct> product = depthStencilStateFactory_->Create(desc);
    if (!product)
    {
        CRCDevice::CoutWrn({"Failed to create depth stencil state."});
        return E_FAIL;
    }

    // Cast the product to IState.
    std::unique_ptr<CRC::IState> state = WACore::UniqueAs<CRC::IState>(product);
    if (!state)
    {
        CRCDevice::CoutWrn({"Failed to cast product to IState.", "Depth stencil state factory may be out of order."});
        return E_FAIL;
    }

    depthStencilState = std::move(state);
    return S_OK;
}

HRESULT CRC::Device::CreateRasterizerState(CRC::IDesc &desc, std::unique_ptr<CRC::IState> &rasterizerState)
{
    // Create rasterizer state using the factory.
    std::unique_ptr<CRC::IProduct> product = rasterizerStateFactory_->Create(desc);
    if (!product)
    {
        CRCDevice::CoutWrn({"Failed to create rasterizer state."});
        return E_FAIL;
    }

    // Cast the product to IState.
    std::unique_ptr<CRC::IState> state = WACore::UniqueAs<CRC::IState>(product);
    if (!state)
    {
        CRCDevice::CoutWrn({"Failed to cast product to IState.", "Rasterizer state factory may be out of order."});
        return E_FAIL;
    }

    rasterizerState = std::move(state);
    return S_OK;
}

HRESULT CRC::Device::CreateSamplerState(CRC::IDesc &desc, std::unique_ptr<CRC::IState> &samplerState)
{
    // Create sampler state using the factory.
    std::unique_ptr<CRC::IProduct> product = samplerStateFactory_->Create(desc);
    if (!product)
    {
        CRCDevice::CoutWrn({"Failed to create sampler state."});
        return E_FAIL;
    }

    // Cast the product to IState.
    std::unique_ptr<CRC::IState> state = WACore::UniqueAs<CRC::IState>(product);
    if (!state)
    {
        CRCDevice::CoutWrn({"Failed to cast product to IState.", "Sampler state factory may be out of order."});
        return E_FAIL;
    }

    samplerState = std::move(state);
    return S_OK;
}

HRESULT CRC::Device::GetDeviceContext(std::unique_ptr<CRC::IDeviceContext> &context)
{
    return E_NOTIMPL;
}
