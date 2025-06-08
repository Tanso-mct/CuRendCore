#include "CuRendCore/include/pch.h"
#include "CuRendCore/include/funcs.cuh"

#include "CuRendCore/include/memory.cuh"

void ICRCMemory::Malloc(UINT byteWidth)
{
#ifndef NDEBUG
    CRC::CoutErr({"ICRCMemory::Malloc(UINT byteWidth) is not implemented."});
#endif
    throw std::runtime_error("ICRCMemory::Malloc(UINT byteWidth) is not implemented.");
}

void ICRCMemory::Free()
{
#ifndef NDEBUG
    CRC::CoutErr({"ICRCMemory::Free() is not implemented."});
#endif
    throw std::runtime_error("ICRCMemory::Free() is not implemented.");
}

void ICRCMemory::HostMalloc(UINT byteWidth)
{
#ifndef NDEBUG
    CRC::CoutErr({"ICRCMemory::HostMalloc(UINT byteWidth) is not implemented."});
#endif
    throw std::runtime_error("ICRCMemory::HostMalloc(UINT byteWidth) is not implemented.");
}

void ICRCMemory::HostFree()
{
#ifndef NDEBUG
    CRC::CoutErr({"ICRCMemory::HostFree() is not implemented."});
#endif
    throw std::runtime_error("ICRCMemory::HostFree() is not implemented.");
}

void ICRCMemory::Assign(void *const mem, UINT byteWidth)
{
#ifndef NDEBUG
    CRC::CoutErr({"ICRCMemory::Assign(void* const mem, UINT byteWidth) is not implemented."});
#endif
    throw std::runtime_error("ICRCMemory::Assign(void* const mem, UINT byteWidth) is not implemented.");
}

void ICRCMemory::Unassign()
{
#ifndef NDEBUG
    CRC::CoutErr({"ICRCMemory::Unassign() is not implemented."});
#endif
    throw std::runtime_error("ICRCMemory::Unassign() is not implemented.");
}
