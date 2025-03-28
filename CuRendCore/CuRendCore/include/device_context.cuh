#pragma once

#include "CuRendCore/include/config.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>

class ICRCContainable;

class CRC_API ICRCDeviceContext
{
public:
    virtual ~ICRCDeviceContext() = default;

    // virtual void VSSetConstantBuffers
    // ( 
    //     _In_range_( 0, D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT - 1 ) UINT startSlot,
    //     _In_range_( 0, D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT - startSlot ) UINT numBuffers,
    //     _In_reads_opt_(numSamplers) ICRCBuffer *const *constantBuffers
    // ) = 0;

    // virtual void VSSetShaderResources
    // ( 
    //     _In_range_( 0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT - 1 ) UINT startSlot,
    //     _In_range_( 0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT - startSlot ) UINT numViews,
    //     _In_reads_opt_(numViews) ID3D11ShaderResourceView *const *shaderResourceViews
    // ) = 0;
    
    // virtual void VSSetSamplers
    // ( 
    //     _In_range_( 0, D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT - 1 ) UINT startSlot,
    //     _In_range_( 0, D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT - startSlot ) UINT NumSamplers,
    //     _In_reads_opt_(NumSamplers) ID3D11SamplerState *const *samplers
    // ) = 0;

    // virtual void PSSetConstantBuffers
    // ( 
    //     _In_range_( 0, D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT - 1 ) UINT startSlot,
    //     _In_range_( 0, D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT - startSlot ) UINT numBuffers,
    //     _In_reads_opt_(numBuffers) ID3D11Buffer *const *constantBuffers
    // ) = 0;
    
    // virtual void PSSetShaderResources
    // ( 
    //     _In_range_( 0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT - 1 ) UINT startSlot,
    //     _In_range_( 0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT - startSlot ) UINT numViews,
    //     _In_reads_opt_(numSamplers) ID3D11ShaderResourceView *const *shaderResourceViews
    // ) = 0;
    
    // virtual void PSSetSamplers
    // ( 
    //     _In_range_( 0, D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT - 1 ) UINT startSlot,
    //     _In_range_( 0, D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT - startSlot ) UINT numSamplers,
    //     _In_reads_opt_(numSamplers) ID3D11SamplerState *const *samplers
    // ) = 0;
    
    // virtual void VSSetShader(ID3D11VertexShader *vertexShader) = 0;
    // virtual void PSSetShader(ID3D11PixelShader *pixelShader) = 0;
    
    // virtual void DrawIndexed
    // ( 
    //     UINT indexCount, UINT startIndexLocation, _In_  INT baseVertexLocation
    // ) = 0;
    
    virtual HRESULT Map
    ( 
        std::unique_ptr<ICRCContainable>& resource,
        UINT subresource,
        D3D11_MAP mapType,
        UINT mapFlags,
        D3D11_MAPPED_SUBRESOURCE *mappedResource
    ) = 0;
    
    virtual void Unmap
    ( 
        std::unique_ptr<ICRCContainable>& resource,
        UINT subresource
    ) = 0;

    virtual void UpdateSubresource
    ( 
        std::unique_ptr<ICRCContainable>& dst,
        const void *src, UINT srcByteWidth
    ) = 0;
    
    // virtual void IASetInputLayout(ID3D11InputLayout *pInputLayout) = 0;
    
    // virtual void IASetVertexBuffers
    // ( 
    //     _In_range_( 0, D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT - 1 ) UINT startSlot,
    //     _In_range_( 0, D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT - startSlot ) UINT numBuffers,
    //     _In_reads_opt_(numBuffers) ICRCBuffer *const *vertexBuffers,
    //     _In_reads_opt_(numBuffers) const UINT *strides,
    //     _In_reads_opt_(numBuffers) const UINT *offsets
    // ) = 0;
    
    // virtual void IASetIndexBuffer
    // ( 
    //     std::unique_ptr<ICRCResource>& indexBuffer,
    //     DXGI_FORMAT format,
    //     UINT offset
    // ) = 0;

    // virtual void IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY topology) = 0;

    // virtual void OMSetRenderTargets
    // ( 
    //     _In_range_( 0, D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT ) UINT numViews,
    //     _In_reads_opt_(numViews)  ID3D11RenderTargetView *const *renderTargetViews,
    //     ID3D11DepthStencilView *depthStencilView
    // ) = 0;

    // virtual void RSSetViewports
    // ( 
    //     _In_range_(0, D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE) UINT numViewports,
    //     _In_reads_opt_(numViewports) const D3D11_VIEWPORT *viewports
    // ) = 0;

    // virtual void ClearRenderTargetView
    // ( 
    //     _In_ ID3D11RenderTargetView *renderTargetView,
    //     _In_ const float colorRGBA[4]
    // ) = 0;

    // virtual void ClearDepthStencilView
    // ( 
    //     _In_ ID3D11DepthStencilView *depthStencilView,
    //     _In_ UINT clearFlags,
    //     _In_ float depth,
    //     _In_ UINT8 stencil
    // ) = 0;
};

class CRC_API CRCImmediateContext : public ICRCDeviceContext
{
public:
    CRCImmediateContext();
    virtual ~CRCImmediateContext() override;

    // ICRCDeviceContext
    virtual HRESULT Map
    ( 
        std::unique_ptr<ICRCContainable>& resource,
        UINT subresource,
        D3D11_MAP mapType,
        UINT mapFlags,
        D3D11_MAPPED_SUBRESOURCE *mappedResource
    ) override;
    
    virtual void Unmap
    ( 
        std::unique_ptr<ICRCContainable>& resource,
        UINT subresource
    ) override;

    virtual void UpdateSubresource
    ( 
        std::unique_ptr<ICRCContainable>& dst,
        const void *src, UINT srcByteWidth
    ) override;
};

class CRC_API CRCID3D11Context : public ICRCDeviceContext
{
private:
    ID3D11DeviceContext* d3d11DeviceContext;

public:
    CRCID3D11Context(ID3D11DeviceContext** d3d11DeviceContext);
    virtual ~CRCID3D11Context() override;

    // ICRCDeviceContext
    virtual HRESULT Map
    ( 
        std::unique_ptr<ICRCContainable>& resource,
        UINT subresource,
        D3D11_MAP mapType,
        UINT mapFlags,
        D3D11_MAPPED_SUBRESOURCE *mappedResource
    ) override;
    
    virtual void Unmap
    ( 
        std::unique_ptr<ICRCContainable>& resource,
        UINT subresource
    ) override;

    virtual void UpdateSubresource
    ( 
        std::unique_ptr<ICRCContainable>& dst,
        const void *src, UINT srcByteWidth
    ) override;
};