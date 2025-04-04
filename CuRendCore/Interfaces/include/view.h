#pragma once

#include "Interfaces/include/device_child.h"
#include "Interfaces/include/resource.h"
#include "Interfaces/include/factory.h"

namespace CRC
{

class IView : public IDeviceChild
{
public:
    virtual ~IView() = default;
    virtual HRESULT GetResource(std::unique_ptr<IResource>* resource) = 0;
    virtual void GetDesc(IDesc *desc) = 0;
};

} // namespace CRC