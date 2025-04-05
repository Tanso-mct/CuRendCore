#pragma once

#include "CRCInterface/include/device_child.h"

namespace CRC
{

enum class STATE_TYPE : UINT
{
    UNKNOWN = 1 << 0,
    DEPTH_STENCIL = 1 << 1,
    BLEND = 1 << 2,
    RASTERIZER = 1 << 3,
    SAMPLER = 1 << 4,
};

class IState : public IDeviceChild
{
public:
    virtual ~IState() = default;
    virtual HRESULT GetType(STATE_TYPE& type) = 0;
};

} // namespace CRC