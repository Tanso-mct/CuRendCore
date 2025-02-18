#pragma once

#include "CRC_config.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class ICRCDevice
{
public:
    virtual ~ICRCDevice() = default;
};

class CRCDevice : public ICRCDevice
{
public:
    ~CRCDevice() override = default;
};

class CRCID3D11Device : public ICRCDevice
{
private:
    Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;

public:
    ~CRCID3D11Device() override = default;
};