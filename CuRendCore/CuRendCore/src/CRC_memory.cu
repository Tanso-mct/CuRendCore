#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_memory.cuh"

void ICRCMemory::Malloc(UINT byteWidth)
{
#ifndef NDEBUG
    CRC::CoutError("ICRCMemory::Malloc(UINT byteWidth) is not implemented.");
#endif
    throw std::runtime_error("ICRCMemory::Malloc(UINT byteWidth) is not implemented.");
}

void ICRCMemory::Free()
{
#ifndef NDEBUG
    CRC::CoutError("ICRCMemory::Free() is not implemented.");
#endif
    throw std::runtime_error("ICRCMemory::Free() is not implemented.");
}

void ICRCMemory::HostMalloc(UINT byteWidth)
{
#ifndef NDEBUG
    CRC::CoutError("ICRCMemory::HostMalloc(UINT byteWidth) is not implemented.");
#endif
    throw std::runtime_error("ICRCMemory::HostMalloc(UINT byteWidth) is not implemented.");
}

void ICRCMemory::HostFree()
{
#ifndef NDEBUG
    CRC::CoutError("ICRCMemory::HostFree() is not implemented.");
#endif
    throw std::runtime_error("ICRCMemory::HostFree() is not implemented.");
}

void ICRCMemory::Assign(void *const mem, UINT byteWidth)
{
#ifndef NDEBUG
    CRC::CoutError("ICRCMemory::Assign(void* const mem, UINT byteWidth) is not implemented.");
#endif
    throw std::runtime_error("ICRCMemory::Assign(void* const mem, UINT byteWidth) is not implemented.");
}

void ICRCMemory::Unassign()
{
#ifndef NDEBUG
    CRC::CoutError("ICRCMemory::Unassign() is not implemented.");
#endif
    throw std::runtime_error("ICRCMemory::Unassign() is not implemented.");
}

const UINT & ICRCMemory::GetByteWidth()
{
#ifndef NDEBUG
    CRC::CoutError("ICRCMemory::GetByteWidth() is not implemented.");
#endif
    throw std::runtime_error("ICRCMemory::GetByteWidth() is not implemented.");

    return 0;
}

void *const ICRCMemory::GetHostPtr()
{
#ifndef NDEBUG
    CRC::CoutError("ICRCMemory::GetHostPtr() is not implemented.");
#endif
    throw std::runtime_error("ICRCMemory::GetHostPtr() is not implemented.");

    return nullptr;
}

HRESULT ICRCMemory::SendHostToDevice()
{
#ifndef NDEBUG
    CRC::CoutError("ICRCMemory::SendHostToDevice() is not implemented.");
#endif
    throw std::runtime_error("ICRCMemory::SendHostToDevice() is not implemented.");

    return S_OK;
}

HRESULT ICRCMemory::SendDeviceToHost()
{
#ifndef NDEBUG
    CRC::CoutError("ICRCMemory::SendDeviceToHost() is not implemented.");
#endif
    throw std::runtime_error("ICRCMemory::SendDeviceToHost() is not implemented.");

    return S_OK;
}
