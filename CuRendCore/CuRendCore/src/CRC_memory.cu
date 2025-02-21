#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_memory.cuh"

CRCHostMem::~CRCHostMem()
{
    if (ptr_) Free();
}

void CRCHostMem::Malloc(const UINT &byteWidth, const UINT &pitch, const UINT &slicePitch)
{
    if (ptr_) throw std::runtime_error("Memory already allocated.");

    byteWidth_ = byteWidth;
    pitch_ = pitch;
    slicePitch_ = slicePitch;

    CRC::CheckCuda(cudaMallocHost(&ptr_, byteWidth_));

    CRC::CoutMsg("Host memory allocated.");
}

void CRCHostMem::Free()
{
    if (!ptr_) throw std::runtime_error("Memory not allocated.");

    byteWidth_ = 0;
    pitch_ = 0;
    slicePitch_ = 0;

    CRC::CheckCuda(cudaFreeHost(ptr_));
    ptr_ = nullptr;

    CRC::CoutMsg("Host memory free.");
}

CRCDeviceMem::~CRCDeviceMem()
{
    if (ptr_) Free();
}

void CRCDeviceMem::Malloc(const UINT &byteWidth, const UINT &pitch, const UINT &slicePitch)
{
    if (ptr_) throw std::runtime_error("Memory already allocated.");

    byteWidth_ = byteWidth;
    pitch_ = pitch;
    slicePitch_ = slicePitch;

    CRC::CheckCuda(cudaMalloc(&ptr_, byteWidth_));

    CRC::CoutMsg("Device memory allocated.");
}

void CRCDeviceMem::Free()
{
    if (!ptr_) throw std::runtime_error("Memory not allocated.");

    byteWidth_ = 0;
    pitch_ = 0;
    slicePitch_ = 0;

    CRC::CheckCuda(cudaFree(ptr_));
    ptr_ = nullptr;

    CRC::CoutMsg("Device memory free.");
}
