﻿#pragma once

#include "CuRendCore/include/config.h"

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

    virtual const UINT& GetByteWidth() = 0;
    virtual const UINT& GetRowPitch() = 0;
    virtual const UINT& GetDepthPitch() = 0;
    virtual void* const GetHostPtr() = 0;

    virtual HRESULT SendHostToDevice() = 0;
    virtual HRESULT SendHostToDevice(const void *src, UINT srcByteWidth) = 0;
    virtual HRESULT SendDeviceToHost() = 0;

    virtual bool IsCpuAccessible() = 0;
};