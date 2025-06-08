#include "pch.h"

#include "CRCTexture/include/texture.h"
#pragma comment(lib, "CRCTexture.lib")

TEST(CRCTexture, Texture1D_GpuRW_CpuRW)
{
    CRC::Texture1dDesc desc;
    desc.cpuRWFlags_ = (UINT)CRC::RW_FLAG::READ | (UINT)CRC::RW_FLAG::WRITE;
    desc.gpuRWFlags_ = (UINT)CRC::RW_FLAG::READ | (UINT)CRC::RW_FLAG::WRITE;
    desc.channelDesc_ = cudaCreateChannelDesc<float>();
    desc.stride_ = sizeof(float);
    desc.width_ = 1024;
    ZeroMemory(&desc.cudaTextureDesc_, sizeof(desc.cudaTextureDesc_));
    desc.cudaTextureDesc_.addressMode[0] = cudaAddressModeClamp;
    desc.cudaTextureDesc_.addressMode[1] = cudaAddressModeClamp;
    desc.cudaTextureDesc_.filterMode = cudaFilterModePoint;
    desc.cudaTextureDesc_.readMode = cudaReadModeElementType;

    CRC::Texture1dFactory factory;
    std::unique_ptr<CRC::IProduct> product = factory.Create(desc);

    {
        WACore::RevertCast<CRC::IUnknown, CRC::IProduct> unknown(product);
        unknown()->Release();
    }
}