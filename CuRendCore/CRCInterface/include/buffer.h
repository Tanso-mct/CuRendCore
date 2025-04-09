#pragma once

#include "CRCInterface/include/resource.h"

namespace CRC
{

class IBuffer : public IResource
{
public:
    virtual ~IBuffer() = default;
    virtual HRESULT GetDataDeviceSide(UINT& size, void** data) = 0;
    virtual HRESULT GetDataHostSide(UINT& size, void** data) = 0;
};

} // namespace CRC