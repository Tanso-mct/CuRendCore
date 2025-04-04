#pragma once

#include "Interfaces/include/unknown.h"
#include "Interfaces/include/factory.h"

namespace CRC
{

class IViewport : public IUnknown
{
public:
    virtual ~IViewport() = default;
    virtual void GetDesc(IDesc *desc) = 0;
};

} // namespace CRC