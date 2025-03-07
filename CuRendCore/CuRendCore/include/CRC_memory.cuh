#pragma once

#include "CRC_config.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Windows.h>
#include <memory>

class CRC_API ICRCMemory
{
public:
    virtual ~ICRCMemory() = default;

    virtual void Malloc(UINT byteWidth);
    virtual void Free();
    virtual void HostMalloc(UINT byteWidth);
    virtual void HostFree();
    virtual void Assign(void* const mem, UINT byteWidth);
    virtual void Unassign();

    virtual const UINT& GetByteWidth();
    virtual void* const GetHostPtr();

    virtual HRESULT SendHostToDevice();
    virtual HRESULT SendDeviceToHost();
};