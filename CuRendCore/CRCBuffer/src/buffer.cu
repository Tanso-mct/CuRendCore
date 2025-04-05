#include "CRCBuffer/include/pch.h"
#include "CRCBuffer/include/buffer.cuh"

CRC::Buffer::Buffer(std::unique_ptr<IDevice> &device)
: device_(device)
{
    // Initialize the buffer with default values
    isValid_ = true;
    type_ = 0;
    size_ = 0;
    dData_ = nullptr;
    hData_ = nullptr;
}

CRC::Buffer::~Buffer()
{
    if (isValid_) Release();
}

HRESULT CRC::Buffer::Release()
{


    isValid_ = false;
    return S_OK;
}

HRESULT CRC::Buffer::GetDevice(std::unique_ptr<CRC::IDevice>*& device)
{
    if (!device_) return E_FAIL;

    device = &device_;
    return S_OK;
}

HRESULT CRC::Buffer::GetType(UINT &type)
{
    if (type == 0) return E_FAIL;
    type = type_;
    return S_OK;
}

void CRC::Buffer::GetDesc(IDesc *desc)
{
}

HRESULT CRC::Buffer::GetDataDeviceSide(UINT &size, void **data)
{
    if (size == 0 || data == nullptr) return E_INVALIDARG;

    size = size_;
    data = &dData_;

    return S_OK;
}

HRESULT CRC::Buffer::GetDataHostSide(UINT &size, void **data)
{
    if (size == 0 || data == nullptr) return E_INVALIDARG;

    size = size_;
    data = &hData_;

    return S_OK;
}

CRC::BufferDesc::BufferDesc(std::unique_ptr<IDevice> &device)
: device_(device)
{
    
}

std::unique_ptr<CRC::IProduct> CRC::BufferFactory::Create(CRC::IDesc &desc) const
{
    CRC::BufferDesc* bufferDesc = WACore::As<CRC::BufferDesc>(&desc);
    if (!bufferDesc)
    {
        CRCBuffer::CoutWrn({"Failed to create buffer from desc.", "Desc is not CRC::BufferDesc."});
        return nullptr;
    }

    return std::make_unique<CRC::Buffer>(bufferDesc->device_);
}