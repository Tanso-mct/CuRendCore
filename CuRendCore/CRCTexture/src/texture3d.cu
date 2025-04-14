#include "CRCTexture/include/pch.h"
#include "CRCTexture/include/texture3d.cuh"

CRC::Texture3d::Texture3d
(
    std::unique_ptr<CRC::IDevice> &device, 
    UINT cpuRWFlags, UINT gpuRWFlags, cudaChannelFormatDesc channelDesc,
    UINT stride, UINT width, UINT height, UINT depth
): device_(device),
type_(
    ((cpuRWFlags & (UINT)CRC::RW_FLAG::READ) ? (UINT)CRC::RESOURCE_TYPE::CPU_R : 0) |
    ((cpuRWFlags & (UINT)CRC::RW_FLAG::WRITE) ? (UINT)CRC::RESOURCE_TYPE::CPU_W : 0) |
    ((gpuRWFlags & (UINT)CRC::RW_FLAG::READ) ? (UINT)CRC::RESOURCE_TYPE::GPU_R : 0) |
    ((gpuRWFlags & (UINT)CRC::RW_FLAG::WRITE) ? (UINT)CRC::RESOURCE_TYPE::GPU_W : 0) |
    (UINT)CRC::RESOURCE_TYPE::TEXTURE3D),
channelDesc_(channelDesc), stride_(stride), width_(width), height_(height), depth_(depth)
{
    // Initialize the texture with default values
    isValid_ = true;
    dArray_ = nullptr;
    hPtr_ = nullptr;
    object_ = 0;
}

CRC::Texture3d::~Texture3d()
{
    if (isValid_) Release();
}

HRESULT CRC::Texture3d::Release()
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
        CudaCore::Free3D(&dArray_);
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

HRESULT CRC::Texture3d::GetType(UINT &type) const
{
    if (type == 0) return E_FAIL;
    type = type_;
    return S_OK;
}

void CRC::Texture3d::GetDesc(IDesc *desc) const
{
}

HRESULT CRC::Texture3d::GetSize(UINT &size) const
{
    if (!isValid_) return E_FAIL;

    size = width_ * height_ * depth_ * stride_;
    return S_OK;
}

HRESULT CRC::Texture3d::GetStride(UINT &stride) const
{
    if (!isValid_) return E_FAIL;

    stride = stride_;
    return S_OK;
}

HRESULT CRC::Texture3d::GetWidth(UINT &width) const
{
    if (!isValid_) return E_FAIL;

    width = width_;
    return S_OK;
}

HRESULT CRC::Texture3d::GetHeight(UINT &height) const
{
    if (!isValid_) return E_FAIL;

    height = height_;
    return S_OK;
}

HRESULT CRC::Texture3d::GetDepth(UINT &depth) const
{
    if (!isValid_) return E_FAIL;

    depth = depth_;
    return S_OK;
}

HRESULT CRC::Texture3d::GetFormat(cudaChannelFormatDesc &channelDesc) const
{
    if (!isValid_) return E_FAIL;

    channelDesc = channelDesc_;
    return S_OK;
}

HRESULT CRC::Texture3d::GetArray(cudaArray **array)
{
    if (!isValid_) return E_FAIL;

    array = &dArray_;
    return S_OK;
}

HRESULT CRC::Texture3d::GetObj(unsigned long long *object)
{
    if (!isValid_) return E_FAIL;

    object = &object_;
    return S_OK;
}

HRESULT CRC::Texture3d::GetDataHostSide(void **data)
{
    if (!isValid_) return E_FAIL;

    data = &hPtr_;
    return S_OK;
}

CRC::Texture3dDesc::Texture3dDesc(std::unique_ptr<CRC::IDevice> &device)
: device_(device)
{
    // Initialize the texture descriptor with default values
    cpuRWFlags_ = 0;
    gpuRWFlags_ = 0;

    stride_ = 0;
    width_ = 0;
    height_ = 0;
    depth_ = 0;
}

std::unique_ptr<CRC::IProduct> CRC::Texture3dFactory::Create(CRC::IDesc &desc) const
{
    CRC::Texture3dDesc* texture3dDesc = WACore::As<CRC::Texture3dDesc>(&desc);
    if (!texture3dDesc)
    {
        CRCTexture::CoutWrn({"Failed to create texture3d from desc.", "Desc is not CRC::Texture3dDesc."});
        return nullptr;
    }

    std::unique_ptr<CRC::IProduct> product = std::make_unique<CRC::Texture3d>
    (
        texture3dDesc->device_, 
        texture3dDesc->cpuRWFlags_, texture3dDesc->gpuRWFlags_,
        texture3dDesc->channelDesc_,
        texture3dDesc->stride_, texture3dDesc->width_, texture3dDesc->height_, texture3dDesc->depth_
    );

    {
        WACore::RevertCast<CRC::Texture3d, CRC::IProduct> texture3d(product);

        UINT type = 0;
        texture3d()->GetType(type);

        cudaArray* dArray = nullptr;
        texture3d()->GetArray(&dArray);
        if (type & (UINT)CRC::RESOURCE_TYPE::GPU_R || type & (UINT)CRC::RESOURCE_TYPE::GPU_W)
        {
            UINT width, height, depth;
            cudaChannelFormatDesc channelDesc;
            texture3d()->GetWidth(width);
            texture3d()->GetHeight(height);
            texture3d()->GetDepth(depth);
            texture3d()->GetFormat(channelDesc);
            
            CudaCore::Malloc3DArray
            (
                &dArray, &texture3dDesc->channelDesc_,
                width, height, depth
            );
        }

        struct cudaResourceDesc resDesc;
        ZeroMemory(&resDesc, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = dArray;

        unsigned long long* object = 0;
        texture3d()->GetObj(object);

        if (type & (UINT)CRC::RESOURCE_TYPE::GPU_R && !(type & (UINT)CRC::RESOURCE_TYPE::GPU_W))
        {
            CudaCore::CreateTextureObj(object, &resDesc, &texture3dDesc->cudaTextureDesc_, 0);
        }
        else if (type & (UINT)CRC::RESOURCE_TYPE::GPU_W)
        {
            CudaCore::CreateSurfaceObj(object, &resDesc);
        }

        if (type & (UINT)CRC::RESOURCE_TYPE::CPU_R || type & (UINT)CRC::RESOURCE_TYPE::CPU_W)
        {
            void** data = nullptr;
            texture3d()->GetDataHostSide(data);

            UINT size = 0;
            texture3d()->GetSize(size);

            CudaCore::MallocHost(data, size);
        }
    }

    return product;
}
