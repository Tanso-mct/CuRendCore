#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_memory.cuh"

void ICRCMemory::Malloc(UINT byteWidth)
{
    CRC::CoutWarning("ICRCMemory::Malloc(UINT byteWidth) is not implemented.");
}

void ICRCMemory::Free()
{
    CRC::CoutWarning("ICRCMemory::Free() is not implemented.");
}

void ICRCMemory::Assign(void *const mem, UINT byteWidth)
{
    CRC::CoutWarning("ICRCMemory::Assign(void* const mem, UINT byteWidth) is not implemented.");
}

void ICRCMemory::Unassign()
{
    CRC::CoutWarning("ICRCMemory::Unassign() is not implemented.");
}
