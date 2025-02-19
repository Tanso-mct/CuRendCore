#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_texture.cuh"

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
