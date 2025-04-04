#pragma once

#include <memory>

namespace FACTORY
{

class IDesc
{
public:
    virtual ~IDesc() = default;
};

class IProduct
{
public:
    virtual ~IProduct() = default;
};

class IFactory
{
public:
    virtual ~IFactory() = default;
    virtual std::unique_ptr<IProduct> Create(IDesc& desc) const = 0;
};

} // namespace FACTORY