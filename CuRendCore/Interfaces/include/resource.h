#pragma once

#include "Interfaces/include/device_child.h"
#include "Interfaces/include/factory.h"

namespace CRC
{

enum class RESOURCE_TYPE : UINT
{
    UNKNOWN = 1 << 0,

    BUFFER = 1 << 1,
    TEXTURE1D = 1 << 2,
    TEXTURE2D = 1 << 3,
    TEXTURE3D = 1 << 4,

    CPU_R = 1 << 5,
    CPU_W = 1 << 6,
    GPU_R = 1 << 7,
    GPU_W = 1 << 8,
};

class IResource : public IDeviceChild
{
public:
    virtual ~IResource() = default;
    
    virtual HRESULT GetType(UINT& type) = 0;
    virtual void GetDesc(IDesc *desc) = 0;

    virtual HRESULT GetDataDeviceSide(UINT& size, void** data) = 0;
    virtual HRESULT GetDataHostSide(UINT& size, void** data) = 0;
};

} // namespace CRC