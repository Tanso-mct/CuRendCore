#pragma once

#include "Interfaces/include/Unknown.h"
#include "Interfaces/include/device.h"

#include <memory>

namespace CRC
{
class IDeviceChild : public IUnknown
{
public:
    virtual ~IDeviceChild() = default;
    virtual HRESULT GetDevice(std::unique_ptr<IDevice>* device) = 0;
};

} // namespace CRC