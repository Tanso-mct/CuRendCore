#include "CRC_pch.h"
#include "CRC_funcs.h"

#include "CRC_buffer.cuh"

void * CRCBuffer::GetMem() const
{
    return nullptr;
}

std::size_t CRCBuffer::GetSize() const
{
    return 0;
}

void *CRCID3D11Buffer::GetMem() const
{
    return nullptr;
}

std::size_t CRCID3D11Buffer::GetSize() const
{
    return 0;
}
