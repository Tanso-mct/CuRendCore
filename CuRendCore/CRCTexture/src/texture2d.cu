#include "CRCTexture/include/pch.h"
#include "CRCTexture/include/texture2d.cuh"

CRC::Texture2d::Texture2d
(
    UINT cpuRWFlags, UINT gpuRWFlags, cudaChannelFormatDesc channelDesc,
    UINT stride, UINT width, UINT height
):
type_(
    ((cpuRWFlags & (UINT)CRC::RW_FLAG::READ) ? (UINT)CRC::RESOURCE_TYPE::CPU_R : 0) |
    ((cpuRWFlags & (UINT)CRC::RW_FLAG::WRITE) ? (UINT)CRC::RESOURCE_TYPE::CPU_W : 0) |
    ((gpuRWFlags & (UINT)CRC::RW_FLAG::READ) ? (UINT)CRC::RESOURCE_TYPE::GPU_R : 0) |
    ((gpuRWFlags & (UINT)CRC::RW_FLAG::WRITE) ? (UINT)CRC::RESOURCE_TYPE::GPU_W : 0) |
    (UINT)CRC::RESOURCE_TYPE::TEXTURE2D),
channelDesc_(channelDesc), stride_(stride), width_(width), height_(height)
{
    // Initialize the texture with default values
    isValid_ = true;
    dArray_ = nullptr;
    hPtr_ = nullptr;
    object_ = 0;
}

CRC::Texture2d::~Texture2d()
{
    if (isValid_) Release();
}

HRESULT CRC::Texture2d::Release()
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

HRESULT CRC::Texture2d::GetType(UINT &type) const
{
    if (type == 0) return E_FAIL;
    type = type_;
    return S_OK;
}

void CRC::Texture2d::GetDesc(IDesc *desc) const
{
}

HRESULT CRC::Texture2d::GetSize(UINT &size) const
{
    if (!isValid_) return E_FAIL;

    size = width_ * height_ * stride_;
    return S_OK;
}

HRESULT CRC::Texture2d::GetStride(UINT &stride) const
{
    if (!isValid_) return E_FAIL;

    stride = stride_;
    return S_OK;
}

HRESULT CRC::Texture2d::GetWidth(UINT &width) const
{
    if (!isValid_) return E_FAIL;

    width = width_;
    return S_OK;
}

HRESULT CRC::Texture2d::GetHeight(UINT &height) const
{
    if (!isValid_) return E_FAIL;

    height = height_;
    return S_OK;
}

HRESULT CRC::Texture2d::GetDepth(UINT &depth) const
{
    if (!isValid_) return E_FAIL;

    depth = 1;
    return S_OK;
}

HRESULT CRC::Texture2d::GetFormat(cudaChannelFormatDesc &channelDesc) const
{
    if (!isValid_) return E_FAIL;

    channelDesc = channelDesc_;
    return S_OK;
}

HRESULT CRC::Texture2d::GetArray(cudaArray **array)
{
    if (!isValid_) return E_FAIL;

    array = &dArray_;
    return S_OK;
}

HRESULT CRC::Texture2d::GetObj(unsigned long long *object)
{
    if (!isValid_) return E_FAIL;

    object = &object_;
    return S_OK;
}

HRESULT CRC::Texture2d::GetDataHostSide(void **data)
{
    if (!isValid_) return E_FAIL;

    data = &hPtr_;
    return S_OK;
}

CRC::Texture2dDesc::Texture2dDesc()
{
    // Initialize the texture descriptor with default values
    cpuRWFlags_ = 0;
    gpuRWFlags_ = 0;

    stride_ = 0;
    width_ = 0;
    height_ = 0;
}

std::unique_ptr<CRC::IProduct> CRC::Texture2dFactory::Create(CRC::IDesc &desc) const
{
    CRC::Texture2dDesc* texture2dDesc = WACore::As<CRC::Texture2dDesc>(&desc);
    if (!texture2dDesc)
    {
        CRCTexture::CoutWrn({"Failed to create texture2d from desc.", "Desc is not CRC::Texture2dDesc."});
        return nullptr;
    }

    std::unique_ptr<CRC::IProduct> product = std::make_unique<CRC::Texture2d>
    (
        texture2dDesc->cpuRWFlags_, texture2dDesc->gpuRWFlags_,
        texture2dDesc->channelDesc_,
        texture2dDesc->stride_, texture2dDesc->width_, texture2dDesc->height_
    );

    {
        WACore::RevertCast<CRC::Texture2d, CRC::IProduct> texture2d(product);

        UINT type = 0;
        texture2d()->GetType(type);

        cudaArray* dArray = nullptr;
        texture2d()->GetArray(&dArray);
        if (type & (UINT)CRC::RESOURCE_TYPE::GPU_R || type & (UINT)CRC::RESOURCE_TYPE::GPU_W)
        {
            UINT width, height;
            cudaChannelFormatDesc channelDesc;
            texture2d()->GetWidth(width);
            texture2d()->GetHeight(height);
            texture2d()->GetFormat(channelDesc);
            
            CudaCore::MallocArray
            (
                &dArray, &texture2dDesc->channelDesc_, 
                width, height
            );
        }

        struct cudaResourceDesc resDesc;
        ZeroMemory(&resDesc, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = dArray;

        unsigned long long* object = 0;
        texture2d()->GetObj(object);

        if (type & (UINT)CRC::RESOURCE_TYPE::GPU_R && !(type & (UINT)CRC::RESOURCE_TYPE::GPU_W))
        {
            CudaCore::CreateTextureObj(object, &resDesc, &texture2dDesc->cudaTextureDesc_, 0);
        }
        else if (type & (UINT)CRC::RESOURCE_TYPE::GPU_W)
        {
            CudaCore::CreateSurfaceObj(object, &resDesc);
        }

        if (type & (UINT)CRC::RESOURCE_TYPE::CPU_R || type & (UINT)CRC::RESOURCE_TYPE::CPU_W)
        {
            void** data = nullptr;
            texture2d()->GetDataHostSide(data);

            UINT size = 0;
            texture2d()->GetSize(size);

            CudaCore::MallocHost(data, size);
        }
    }

    return product;
}
