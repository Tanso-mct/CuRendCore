#pragma once

#include "CRCDevice/include/config.h"

#include "WinAppCore/include/WACore.h"

#include "CRCInterface/include/factory.h"
#include "CRCInterface/include/device.h"

namespace CRC
{

class CRC_DEVICE Device : public IDevice, public WACore::IContainable
{
private:
    const std::unique_ptr<IFactory> bufferFactory_;
    const std::unique_ptr<IFactory> texture1dFactory_;
    const std::unique_ptr<IFactory> texture2dFactory_;
    const std::unique_ptr<IFactory> texture3dFactory_;
    const std::unique_ptr<IFactory> shaderResourceViewFactory_;
    const std::unique_ptr<IFactory> renderTargetViewFactory_;
    const std::unique_ptr<IFactory> depthStencilViewFactory_;
    const std::unique_ptr<IFactory> vertexShaderFactory_;
    const std::unique_ptr<IFactory> pixelShaderFactory_;
    const std::unique_ptr<IFactory> blendStateFactory_;
    const std::unique_ptr<IFactory> depthStencilStateFactory_;
    const std::unique_ptr<IFactory> rasterizerStateFactory_;
    const std::unique_ptr<IFactory> samplerStateFactory_;

public:
    Device() = delete;
    Device
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
    );
    virtual ~Device() = default;

    //*************************************************************************************************************** */
    // IDevice
    //*************************************************************************************************************** */

    HRESULT CreateBuffer(IDesc& desc, std::unique_ptr<IResource>& buffer) override;
    HRESULT CreateTexture1D(IDesc& desc, std::unique_ptr<IResource>& texture1d) override;
    HRESULT CreateTexture2D(IDesc& desc, std::unique_ptr<IResource>& texture2d) override;
    HRESULT CreateTexture3D(IDesc& desc, std::unique_ptr<IResource>& texture3d) override;

    HRESULT CreateShaderResourceView(IDesc& desc, std::unique_ptr<IView>& srv) override;
    HRESULT CreateRenderTargetView(IDesc& desc, std::unique_ptr<IView>& rtv) override;
    HRESULT CreateDepthStencilView(IDesc& desc, std::unique_ptr<IView>& dsv) override;

    HRESULT CreateVertexShader(IDesc& desc, std::unique_ptr<IShader>& vertexShader) override;
    HRESULT CreatePixelShader(IDesc& desc, std::unique_ptr<IShader>& pixelShader) override;

    HRESULT CreateBlendState(IDesc& desc, std::unique_ptr<IState>& blendState) override;
    HRESULT CreateDepthStencilState(IDesc& desc, std::unique_ptr<IState>& depthStencilState) override;
    HRESULT CreateRasterizerState(IDesc& desc, std::unique_ptr<IState>& rasterizerState) override;
    HRESULT CreateSamplerState(IDesc& desc, std::unique_ptr<IState>& samplerState) override;

    HRESULT GetDeviceContext(std::unique_ptr<IDeviceContext>& context) override;
};

} // namespace CRC