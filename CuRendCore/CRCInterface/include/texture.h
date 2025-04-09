#pragma once

#include "CRCInterface/include/resource.h"

struct cudaArray;

namespace CRC
{

class IDesc;

class ITexture
{
public:
    virtual ~ITexture() = default;
    virtual HRESULT GetArray(UINT& stride, UINT& width, UINT& height, cudaArray** array) = 0;
    virtual HRESULT GetObject(UINT& stride, UINT& width, UINT& height, unsigned long long* object) = 0;
    virtual HRESULT GetDataHostSide(UINT& size, void** data) = 0;
};

} // namespace CRC