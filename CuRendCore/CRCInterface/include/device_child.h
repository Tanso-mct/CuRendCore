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
    virtual HRESULT GetDevice(const std::unique_ptr<IDevice>*& device) const = 0;
};

} // namespace CRC