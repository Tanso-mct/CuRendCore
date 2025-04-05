#pragma once

#include "CRCInterface/include/device_child.h"

namespace CRC
{

class IResource;
class IDesc;

class IView : public IDeviceChild
{
public:
    virtual ~IView() = default;
    virtual HRESULT GetResource(std::unique_ptr<IResource>* resource) = 0;
    virtual void GetDesc(IDesc *desc) = 0;
};

} // namespace CRC