#include "CRCTexture/include/pch.h"
#include "CRCTexture/include/texture2d.cuh"
#include "texture2d.cuh"

CRC::Texture2d::Texture2d(std::unique_ptr<CRC::IDevice> &device, UINT cpuRWFlags, UINT gpuRWFlags)
: device_(device),
type_(
    ((cpuRWFlags & (UINT)CRC::RW_FLAG::READ) ? (UINT)CRC::RESOURCE_TYPE::CPU_R : 0) |
    ((cpuRWFlags & (UINT)CRC::RW_FLAG::WRITE) ? (UINT)CRC::RESOURCE_TYPE::CPU_W : 0) |
    ((gpuRWFlags & (UINT)CRC::RW_FLAG::READ) ? (UINT)CRC::RESOURCE_TYPE::GPU_R : 0) |
    ((gpuRWFlags & (UINT)CRC::RW_FLAG::WRITE) ? (UINT)CRC::RESOURCE_TYPE::GPU_W : 0)
){
    // Initialize the texture with default values
    isValid_ = true;
    width_ = 0;
    height_ = 0;
    stride_ = 0;
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
    width_ = 0;
    height_ = 0;
    stride_ = 0;

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

HRESULT CRC::Texture2d::GetDevice(std::unique_ptr<CRC::IDevice> *&device)
{
    if (!device_) return E_FAIL;

    device = &device_;
    return S_OK;
}

HRESULT CRC::Texture2d::GetType(UINT &type)
{
    if (type == 0) return E_FAIL;
    type = type_;
    return S_OK;
}

void CRC::Texture2d::GetDesc(IDesc *desc)
{
}

HRESULT CRC::Texture2d::GetDataDeviceSide(UINT &size, void **data)
{
    if (!isValid_) return E_FAIL;

    size = stride_ * width_ * height_;
    void* dData = &object_;
    data =  &dData;

    return S_OK;
}

HRESULT CRC::Texture2d::GetDataHostSide(UINT &size, void **data)
{
    if (!isValid_) return E_FAIL;

    size = stride_ * width_ * height_;
    data = &hPtr_;

    return S_OK;
}

HRESULT CRC::Texture2d::GetArray(UINT &size, cudaArray **array)
{
    if (!isValid_) return E_FAIL;

    size = stride_ * width_ * height_;
    array = &dArray_;

    return S_OK;
}

CRC::Texture2dDesc::Texture2dDesc(std::unique_ptr<CRC::IDevice> &device)
: device_(device)
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
        texture2dDesc->device_, 
        texture2dDesc->cpuRWFlags_, texture2dDesc->gpuRWFlags_
    );

    {
        WACore::RevertCast<CRC::Texture2d, CRC::IProduct> texture2d(product);

        UINT type = 0;
        texture2d()->GetType(type);

        if (type & (UINT)CRC::RESOURCE_TYPE::GPU_R || type & (UINT)CRC::RESOURCE_TYPE::GPU_W)
        {
            UINT size = 0;
            cudaArray* dArray = nullptr;
        }

        if (type & (UINT)CRC::RESOURCE_TYPE::GPU_R && !(type & (UINT)CRC::RESOURCE_TYPE::GPU_W))
        {
            
        }
        else if (type & (UINT)CRC::RESOURCE_TYPE::GPU_W)
        {

        }

        if (type & (UINT)CRC::RESOURCE_TYPE::CPU_R || type & (UINT)CRC::RESOURCE_TYPE::CPU_W)
        {
            
        }
    }

    return product;
}
