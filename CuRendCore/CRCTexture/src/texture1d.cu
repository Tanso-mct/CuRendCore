#include "CRCTexture/include/pch.h"
#include "CRCTexture/include/texture1d.cuh"

CRC::Texture1d::Texture1d
(
    std::unique_ptr<CRC::IDevice> &device, 
    UINT cpuRWFlags, UINT gpuRWFlags, cudaChannelFormatDesc channelDesc,
    UINT stride, UINT width
): device_(device),
type_(
    ((cpuRWFlags & (UINT)CRC::RW_FLAG::READ) ? (UINT)CRC::RESOURCE_TYPE::CPU_R : 0) |
    ((cpuRWFlags & (UINT)CRC::RW_FLAG::WRITE) ? (UINT)CRC::RESOURCE_TYPE::CPU_W : 0) |
    ((gpuRWFlags & (UINT)CRC::RW_FLAG::READ) ? (UINT)CRC::RESOURCE_TYPE::GPU_R : 0) |
    ((gpuRWFlags & (UINT)CRC::RW_FLAG::WRITE) ? (UINT)CRC::RESOURCE_TYPE::GPU_W : 0) |
    (UINT)CRC::RESOURCE_TYPE::TEXTURE2D),
channelDesc_(channelDesc), stride_(stride), width_(width)
{
    // Initialize the texture with default values
    isValid_ = true;
    dArray_ = nullptr;
    hPtr_ = nullptr;
    object_ = 0;
}

CRC::Texture1d::~Texture1d()
{
    if (isValid_) Release();
}

HRESULT CRC::Texture1d::Release()
{
    if (type_ & (UINT)CRC::RESOURCE_TYPE::GPU_W)
    {
        CudaCore::DestroySurfaceObj(&object_);
        object_ = 0;
    }
    else if (type_ & (UINT)CRC::RESOURCE_TYPE::GPU_R)
    {
        CudaCore::DestroyTextureObj(&object_);
        object_ = 0;
    }

    if (dArray_)
    {
        CudaCore::FreeArray(&dArray_);
        dArray_ = nullptr;
    }
    if (hPtr_)
    {
        CudaCore::FreeHost(&hPtr_);
        hPtr_ = nullptr;
    }

    isValid_ = false;
    return S_OK;
}

HRESULT CRC::Texture1d::GetDevice(const std::unique_ptr<CRC::IDevice> *&device) const
{
    if (!device_) return E_FAIL;

    device = &device_;
    return S_OK;
}

HRESULT CRC::Texture1d::GetType(UINT &type) const
{
    if (type == 0) return E_FAIL;
    type = type_;
    return S_OK;
}

void CRC::Texture1d::GetDesc(IDesc *desc) const
{
}

HRESULT CRC::Texture1d::GetSize(UINT &size) const
{
    if (!isValid_) return E_FAIL;

    size = width_ * stride_;
    return S_OK;
}

HRESULT CRC::Texture1d::GetStride(UINT &stride) const
{
    if (!isValid_) return E_FAIL;

    stride = stride_;
    return S_OK;
}

HRESULT CRC::Texture1d::GetWidth(UINT &width) const
{
    if (!isValid_) return E_FAIL;

    width = width_;
    return S_OK;
}

HRESULT CRC::Texture1d::GetHeight(UINT &height) const
{
    if (!isValid_) return E_FAIL;

    height = 1;
    return S_OK;
}

HRESULT CRC::Texture1d::GetDepth(UINT &depth) const
{
    if (!isValid_) return E_FAIL;

    depth = 1;
    return S_OK;
}

HRESULT CRC::Texture1d::GetFormat(cudaChannelFormatDesc &channelDesc) const
{
    if (!isValid_) return E_FAIL;

    channelDesc = channelDesc_;
    return S_OK;
}

HRESULT CRC::Texture1d::GetArray(cudaArray **array)
{
    if (!isValid_) return E_FAIL;

    array = &dArray_;
    return S_OK;
}

HRESULT CRC::Texture1d::GetObj(unsigned long long *object)
{
    if (!isValid_) return E_FAIL;

    object = &object_;
    return S_OK;
}

HRESULT CRC::Texture1d::GetDataHostSide(void **data)
{
    if (!isValid_) return E_FAIL;

    data = &hPtr_;
    return S_OK;
}

CRC::Texture1dDesc::Texture1dDesc(std::unique_ptr<CRC::IDevice> &device)
: device_(device)
{
    // Initialize the texture descriptor with default values
    cpuRWFlags_ = 0;
    gpuRWFlags_ = 0;

    stride_ = 0;
    width_ = 0;
}

std::unique_ptr<CRC::IProduct> CRC::Texture1dFactory::Create(CRC::IDesc &desc) const
{
    CRC::Texture1dDesc* texture1dDesc = WACore::As<CRC::Texture1dDesc>(&desc);
    if (!texture1dDesc)
    {
        CRCTexture::CoutWrn({"Failed to create texture1d from desc.", "Desc is not CRC::Texture1dDesc."});
        return nullptr;
    }

    std::unique_ptr<CRC::IProduct> product = std::make_unique<CRC::Texture1d>
    (
        texture1dDesc->device_, 
        texture1dDesc->cpuRWFlags_, texture1dDesc->gpuRWFlags_,
        texture1dDesc->channelDesc_,
        texture1dDesc->stride_, texture1dDesc->width_
    );

    {
        WACore::RevertCast<CRC::Texture1d, CRC::IProduct> texture1d(product);

        UINT type = 0;
        texture1d()->GetType(type);

        cudaArray* dArray = nullptr;
        texture1d()->GetArray(&dArray);
        if (type & (UINT)CRC::RESOURCE_TYPE::GPU_R || type & (UINT)CRC::RESOURCE_TYPE::GPU_W)
        {
            UINT width, height;
            cudaChannelFormatDesc channelDesc;
            texture1d()->GetWidth(width);
            texture1d()->GetHeight(height);
            texture1d()->GetFormat(channelDesc);
            
            CudaCore::MallocArray
            (
                &dArray, &texture1dDesc->channelDesc_, 
                width, height
            );
        }

        struct cudaResourceDesc resDesc;
        ZeroMemory(&resDesc, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = dArray;

        unsigned long long* object = 0;
        texture1d()->GetObj(object);

        if (type & (UINT)CRC::RESOURCE_TYPE::GPU_R && !(type & (UINT)CRC::RESOURCE_TYPE::GPU_W))
        {
            CudaCore::CreateTextureObj(object, &resDesc, &texture1dDesc->cudaTextureDesc_, 0);
        }
        else if (type & (UINT)CRC::RESOURCE_TYPE::GPU_W)
        {
            CudaCore::CreateSurfaceObj(object, &resDesc);
        }

        if (type & (UINT)CRC::RESOURCE_TYPE::CPU_R || type & (UINT)CRC::RESOURCE_TYPE::CPU_W)
        {
            void** data = nullptr;
            texture1d()->GetDataHostSide(data);

            UINT size = 0;
            texture1d()->GetSize(size);

            CudaCore::MallocHost(data, size);
        }
    }

    return product;
}
