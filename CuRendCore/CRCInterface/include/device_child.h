#pragma once

#include "CRCInterface/include/Unknown.h"

#include <memory>

namespace CRC
{

class IDevice;

class IDeviceChild : public IUnknown
{
public:
    virtual ~IDeviceChild() = default;
    virtual HRESULT GetDevice(std::unique_ptr<IDevice>*& device) = 0;
};

} // namespace CRC