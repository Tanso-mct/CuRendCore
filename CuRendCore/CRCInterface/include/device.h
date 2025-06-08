#pragma once

#include "CRCInterface/include/unknown.h"

#include <memory>

namespace CRC
{

class IDesc;
class IResource;
class IView;
class IShader;
class IState;
class IDeviceContext;

class IDevice : public IUnknown
{
public:
    virtual ~IDevice() = default;

    virtual HRESULT CreateBuffer(IDesc& desc, std::unique_ptr<IResource>& buffer) = 0;
    virtual HRESULT CreateTexture1D(IDesc& desc, std::unique_ptr<IResource>& texture1d) = 0;
    virtual HRESULT CreateTexture2D(IDesc& desc, std::unique_ptr<IResource>& texture2d) = 0;
    virtual HRESULT CreateTexture3D(IDesc& desc, std::unique_ptr<IResource>& texture3d) = 0;

    virtual HRESULT CreateShaderResourceView(IDesc& desc, std::unique_ptr<IView>& srv) = 0;
    virtual HRESULT CreateRenderTargetView(IDesc& desc, std::unique_ptr<IView>& rtv) = 0;
    virtual HRESULT CreateDepthStencilView(IDesc& desc, std::unique_ptr<IView>& dsv) = 0;

    virtual HRESULT CreateVertexShader(IDesc& desc, std::unique_ptr<IShader>& vertexShader) = 0;
    virtual HRESULT CreatePixelShader(IDesc& desc, std::unique_ptr<IShader>& pixelShader) = 0;

    virtual HRESULT CreateBlendState(IDesc& desc, std::unique_ptr<IState>& blendState) = 0;
    virtual HRESULT CreateDepthStencilState(IDesc& desc, std::unique_ptr<IState>& depthStencilState) = 0;
    virtual HRESULT CreateRasterizerState(IDesc& desc, std::unique_ptr<IState>& rasterizerState) = 0;
    virtual HRESULT CreateSamplerState(IDesc& desc, std::unique_ptr<IState>& samplerState) = 0;

    virtual HRESULT GetDeviceContext(std::unique_ptr<IDeviceContext>& context) = 0;
};

} // namespace CRC