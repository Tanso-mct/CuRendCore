#pragma once

#include "CRC_config.h"
#include "CRC_container.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CRC_API ICRCResource
{
public:
    virtual ~ICRCResource() = default;
    virtual void* GetMem() const = 0;
    virtual std::size_t GetSize() const = 0;
};