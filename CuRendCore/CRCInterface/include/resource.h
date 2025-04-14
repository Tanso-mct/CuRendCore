#pragma once

#include "CRCInterface/include/unknown.h"

namespace CRC
{

class IDesc;

enum class RW_FLAG : UINT
{
    NONE = 0,
    READ = 1 << 1,
    WRITE = 1 << 2,
};

enum class RESOURCE_TYPE : UINT
{
    UNKNOWN = 0,

    BUFFER = 1 << 1,
    TEXTURE1D = 1 << 2,
    TEXTURE2D = 1 << 3,
    TEXTURE3D = 1 << 4,

    CPU_R = 1 << 5,
    CPU_W = 1 << 6,
    GPU_R = 1 << 7,
    GPU_W = 1 << 8,
};

class IResource : public IUnknown
{
public:
    virtual ~IResource() = default;
    
    virtual HRESULT GetType(UINT& type) const = 0;
    virtual void GetDesc(IDesc *desc) const = 0;
};

} // namespace CRC