#pragma once

#include "CRCInterface/include/resource.h"

namespace CRC
{

class IBuffer : public IResource
{
public:
    virtual ~IBuffer() = default;
    virtual HRESULT GetSize(UINT& size) = 0;
    virtual HRESULT GetDataDeviceSide(void** data) = 0;
    virtual HRESULT GetDataHostSide(void** data) = 0;
};

} // namespace CRC