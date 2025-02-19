#pragma once

#include "CRC_config.h"
#include "CRC_container.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

enum class CRC_API CRC_USAGE : UINT
{
    DEFAULT = 0,
    IMMUTABLE = 1,
    DYNAMIC = 2,
    STAGING = 3
};

enum class CRC_API CRC_FORMAT : UINT
{
    UNKNOWN = 0,
    R8G8B8A8_UNORM = 28,
    D24_UNORM_S8_UINT = 45,
};

class CRC_API CRC_SUBRESOURCE_DATA
{
public:
    ~CRC_SUBRESOURCE_DATA() = default;

    const void* pSysMem_;
    UINT sysMemPitch_;
    UINT sysMemSlicePitch_;
};

class CRC_API ICRCResource
{
public:
    virtual ~ICRCResource() = default;
    virtual void* GetMem() const = 0;
    virtual std::size_t GetSize() const = 0;
};