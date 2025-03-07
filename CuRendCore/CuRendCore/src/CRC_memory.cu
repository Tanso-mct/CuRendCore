#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_memory.cuh"

void ICRCMemory::Malloc(UINT byteWidth)
{
#ifndef NDEBUG
    CRC::CoutWarning("ICRCMemory::Malloc(UINT byteWidth) is not implemented.");
#endif
}

void ICRCMemory::Free()
{
#ifndef NDEBUG
    CRC::CoutWarning("ICRCMemory::Free() is not implemented.");
#endif
}

void ICRCMemory::HostMalloc(UINT byteWidth)
{
#ifndef NDEBUG
    CRC::CoutWarning("ICRCMemory::HostMalloc(UINT byteWidth) is not implemented.");
#endif
}

void ICRCMemory::HostFree()
{
#ifndef NDEBUG
    CRC::CoutWarning("ICRCMemory::HostFree() is not implemented.");
#endif
}

void ICRCMemory::Assign(void *const mem, UINT byteWidth)
{
#ifndef NDEBUG
    CRC::CoutWarning("ICRCMemory::Assign(void* const mem, UINT byteWidth) is not implemented.");
#endif
}

void ICRCMemory::Unassign()
{
#ifndef NDEBUG
    CRC::CoutWarning("ICRCMemory::Unassign() is not implemented.");
#endif
}
