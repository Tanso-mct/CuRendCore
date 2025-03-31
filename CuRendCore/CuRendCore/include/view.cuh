#pragma once

#include "CuRendCore/include/config.h"
#include "packages/WinAppCore/include/WACore.h"

#include "CuRendCore/include/resource.cuh"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>

class CRC_API ICRCView
{
public:
    virtual ~ICRCView() = default;
    virtual std::unique_ptr<WACore::IContainable>& GetResource() = 0;
};

