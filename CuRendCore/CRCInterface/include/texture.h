#pragma once

#include "CRCInterface/include/resource.h"

struct cudaArray;
struct cudaChannelFormatDesc;

namespace CRC
{

class ITexture : public IResource
{
public:
    virtual ~ITexture() = default;
    virtual HRESULT GetSize(UINT& size) = 0;
    virtual HRESULT GetStride(UINT& stride) = 0;
    virtual HRESULT GetWidth(UINT& width) = 0;
    virtual HRESULT GetHeight(UINT& height) = 0;
    virtual HRESULT GetFormat(cudaChannelFormatDesc& channelDesc) = 0;
    virtual HRESULT GetArray(cudaArray** array) = 0;
    virtual HRESULT GetObj(unsigned long long* object) = 0;
    virtual HRESULT GetDataHostSide(void** data) = 0;
};

} // namespace CRC