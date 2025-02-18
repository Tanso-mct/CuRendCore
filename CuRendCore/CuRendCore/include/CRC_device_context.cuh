#pragma once

#include "CRC_config.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class ICRCDeviceContext
{
public:
    virtual ~ICRCDeviceContext() = default;

    // HRESULT Map
    // (
    //     ID3D11Resource *resource,
    //     UINT subresource,
    //     D3D11_MAP mapType,
    //     UINT mapFlags,
    //     D3D11_MAPPED_SUBRESOURCE* mappedResource
    // );

    // void Unmap
    // (
    //     ID3D11Resource *pResource,
    //     UINT Subresource
    // );
};

class CRCImmediateContext : public ICRCDeviceContext
{
public:
    ~CRCImmediateContext() override = default;
};

class CRCID3D11Context : public ICRCDeviceContext
{
private:
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> d3d11DeviceContext;

public:
    ~CRCID3D11Context() override = default;
};