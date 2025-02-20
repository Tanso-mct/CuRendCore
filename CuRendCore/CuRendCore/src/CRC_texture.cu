#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_texture.cuh"

std::unique_ptr<ICRCContainable> CRCTextureFactory::Create(IDESC &desc) const
{
    return nullptr;
}

void *CRCTexture2D::GetMem() const
{
    return nullptr;
}

std::size_t CRCTexture2D::GetSize() const
{
    return 0;
}

void *CRCID3D11Texture2D::GetMem() const
{
    return nullptr;
}

std::size_t CRCID3D11Texture2D::GetSize() const
{
    return 0;
}
