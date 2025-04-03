#pragma once

#include "Factory/include/config.h"
#include "Factory/include/desc.h"

#include <memory>

namespace FAC
{

class FACTORY_API IProduct
{
public:
    virtual ~IProduct() = default;
};

class FACTORY_API IFactory
{
public:
    virtual ~IFactory() = default;
    virtual std::unique_ptr<IProduct> Create(IDesc& desc) const = 0;
};

} // namespace FAC