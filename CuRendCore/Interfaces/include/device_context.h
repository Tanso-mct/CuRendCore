#pragma once

#include "Interfaces/include/device_child.h"
#include "Interfaces/include/resource.h"
#include "Interfaces/include/viewport.h"

namespace CRC
{

enum class MAP_TYPE : UINT
{
    UNKNOWN = 1 << 0,
    READ = 1 << 1,
    WRITE = 1 << 2,
    READ_WRITE = 1 << 3,
    WRITE_DISCARD = 1 << 4,
    WRITE_NO_OVERWRITE = 1 << 5,
};

struct MAPPED_SUBRESOURCE
{
    void* data = nullptr;
    UINT rowPitch = 0;
    UINT depthPitch = 0;
    UINT size = 0;
};

enum class PRIMITIVE_TOPOLOGY : UINT
{
    UNKNOWN = 0,
    TRIANGLELIST,
};

class IDeviceContext : public IDeviceChild
{
    virtual HRESULT VSSetConstantBuffer(UINT targetSlot, std::unique_ptr<IResource>& buffer) = 0;
    virtual HRESULT VSSetShaderResource(UINT targetSlot, std::unique_ptr<IView>& shaderResourceView) = 0;
    virtual HRESULT VSSetSampler(UINT targetSlot, std::unique_ptr<IState>& samplerState) = 0;
    virtual HRESULT VSSetShader(std::unique_ptr<IShader>& vertexShader) = 0;

    virtual HRESULT PSSetConstantBuffer(UINT targetSlot, std::unique_ptr<IResource>& buffer) = 0;
    virtual HRESULT PSSetShaderResource(UINT targetSlot, std::unique_ptr<IView>& shaderResourceView) = 0;
    virtual HRESULT PSSetSampler(UINT targetSlot, std::unique_ptr<IState>& samplerState) = 0;
    virtual HRESULT PSSetShader(std::unique_ptr<IShader>& pixelShader) = 0;

    virtual void IASetVertexBuffer(UINT targetSlot, std::unique_ptr<IResource>& vertexBuffer) = 0;
    virtual void IASetIndexBuffer(UINT targetSlot, std::unique_ptr<IResource>& indexBuffer) = 0;
    virtual void IASetPrimitiveTopology(PRIMITIVE_TOPOLOGY topology) = 0;

    virtual void RSSetViewport(UINT targetSlot, std::unique_ptr<IViewport>& viewport) = 0;

    virtual HRESULT DrawIndexed(UINT indexCount, UINT startIndexLocation, INT baseVertexLocation) = 0;
    virtual HRESULT DrawIndexedInstanced
    (
        UINT indexCountPerInstance, UINT instanceCount, 
        UINT startIndexLocation, INT baseVertexLocation, 
        UINT startInstanceLocation
    ) = 0;

    virtual HRESULT SetRenderTarget
    (
        UINT numViews,
        std::unique_ptr<IView>& renderTargetView, std::unique_ptr<IView>& depthStencilView
    ) = 0;

    virtual HRESULT ClearRenderTargetView
    (
        std::unique_ptr<IView>& renderTargetView, const float colorRGBA[4]
    ) = 0;
    virtual HRESULT ClearDepthStencilView
    (
        std::unique_ptr<IView>& depthStencilView, UINT clearFlags, float depth, UINT8 stencil
    ) = 0;

    virtual HRESULT Map
    ( 
        std::unique_ptr<IResource>& resource,
        UINT subresource,
        MAP_TYPE mapType,
        UINT mapFlags,
        MAPPED_SUBRESOURCE& mappedResource
    ) = 0;
    virtual void Unmap
    ( 
        std::unique_ptr<IResource>& resource,
        UINT subresource
    ) = 0;
    virtual void UpdateSubresource
    ( 
        std::unique_ptr<IResource>& resource,
        UINT subresource,
        const void *src, const UINT& srcByteWidth
    ) = 0;
};

} // namespace CRC