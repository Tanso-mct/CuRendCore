#pragma once

#include "CRCInterface/include/device_child.h"

namespace CRC
{

enum class SHADER_TYPE : UINT
{
    UNKNOWN = 1 << 0,
    VERTEX = 1 << 1,
    PIXEL = 1 << 2,
};

class IShader : public IDeviceChild
{
public:
    virtual ~IShader() = default;
    virtual HRESULT GetType(SHADER_TYPE& type) = 0;
};

} // namespace CRC