#pragma once

#include "CRC_config.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Windows.h>
#include <memory>

class CRC_API ICRCMem
{
public:
    virtual ~ICRCMem() = default;

    virtual void* const GetMem() = 0;
    virtual void*& GetMemPtr() = 0;
};

class CRC_API ICRCAllocatedMem : public ICRCMem
{
public:
    virtual ~ICRCAllocatedMem() = default;

    virtual void Malloc
    (
        UINT byteWidth, UINT pitch = 1, UINT slicePitch = 1,
        DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN
    ) = 0;
    virtual void Free() = 0;
};

class CRC_API ICRCAttachedMem : public ICRCMem
{
public:
    virtual ~ICRCAttachedMem() = default;

    virtual void Assign(void* const mem, UINT byteWidth, UINT pitch = 1, UINT slicePitch = 1) = 0;
    virtual void Unassign() = 0;
};