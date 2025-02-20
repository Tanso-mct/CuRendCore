#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_memory.cuh"



// CRC_API HRESULT CRC::MallocMem(CRCCudaMem &mem, const std::size_t &size)
// {
//     mem.size_ = size;

//     if (mem.dPtr_)
//     {
//         throw std::runtime_error("Device memory already allocated.");
//         return E_FAIL;
//     }

//     CRC::CheckCuda(cudaMalloc((void**)&mem.dPtr_, size));

//     if (CRC::As<CRCAccessEnabled>(&mem.hReadAccess) || CRC::As<CRCAccessEnabled>(&mem.hWriteAccess))
//     {
//         if (mem.hPtr_)
//         {
//             throw std::runtime_error("Host memory already allocated.");
//             return E_FAIL;
//         }
        
//         CRC::CheckCuda(cudaMallocHost((void**)&mem.hPtr_, size));
//     }
// }

// CRC_API HRESULT CRC::SetHostMem(CRCCudaMem &mem, const D3D11_SUBRESOURCE_DATA &initialData)
// {
//     if (!mem.hPtr_)
//     {
//         throw std::runtime_error("Host memory not allocated.");
//         return E_FAIL;
//     }

//     CRC::CheckCuda(cudaMemcpy((void**)&mem.hPtr_, (void**)&initialData.pSysMem, mem.size_, cudaMemcpyHostToHost));
// }

// CRC_API HRESULT CRC::SetDeviceMem(CRCCudaMem &mem, const D3D11_SUBRESOURCE_DATA &initialData)
// {
//     if (!mem.dPtr_)
//     {
//         throw std::runtime_error("Device memory not allocated.");
//         return E_FAIL;
//     }

//     CRC::CheckCuda(cudaMemcpy((void**)&mem.dPtr_, (void**)&initialData.pSysMem, mem.size_, cudaMemcpyHostToDevice));
// }

// CRC_API HRESULT CRC::SetMem(CRCCudaMem &mem, const D3D11_SUBRESOURCE_DATA &initialData)
// {
//     CRC::SetHostMem(mem, initialData);
//     CRC::SetDeviceMem(mem, initialData);
// }

// CRC_API HRESULT CRC::FreeHostMem(CRCCudaMem &mem)
// {
//     if (!mem.hPtr_)
//     {
//         throw std::runtime_error("Host memory not allocated.");
//         return E_FAIL;
//     }

//     CRC::CheckCuda(cudaFreeHost((void**)&mem.hPtr_));
// }

// CRC_API HRESULT CRC::FreeDeviceMem(CRCCudaMem &mem)
// {
//     if (!mem.dPtr_)
//     {
//         throw std::runtime_error("Device memory not allocated.");
//         return E_FAIL;
//     }

//     CRC::CheckCuda(cudaFree((void**)&mem.dPtr_));
// }

// CRC_API HRESULT CRC::FreeMem(CRCCudaMem &mem)
// {
//     CRC::FreeHostMem(mem);
//     CRC::FreeDeviceMem(mem);
// }

CRC_API HRESULT CRC::MallocCudaMem
(
    CRCCudaMem &mem, const std::size_t &size, const UINT& pitch, const UINT& slicePitch
){
    if (mem.host || mem.device)
    {
        throw std::runtime_error("Memory already allocated.");
        return E_FAIL;
    }

    CRC::CheckCuda(cudaMallocHost((void**)&mem.host->Mem(), size));
    CRC::CheckCuda(cudaMalloc((void**)&mem.host->Mem(), size));

    CRC::CheckCuda(cudaMalloc((void**)&mem.device, sizeof(CRCMem)));

    return S_OK;
}

CRC_API HRESULT CRC::SetCudaMem(CRCCudaMem &mem, const void* sysMem)
{
    if (!mem.host || !mem.device)
    {
        throw std::runtime_error("Memory not allocated.");
        return E_FAIL;
    }

    CRC::CheckCuda(cudaMemcpy
    (
        (void*)mem.host->Mem(), sysMem, 
        mem.host->Size(), cudaMemcpyHostToHost
    ));

    CRC::CheckCuda(cudaMemcpy
    (
        (void*)mem.device, mem.host, 
        sizeof(CRCMem), cudaMemcpyHostToDevice
    ));

    return S_OK;
}

CRC_API HRESULT CRC::FreeCudaMem(CRCCudaMem &mem)
{
    if (!mem.host || !mem.device)
    {
        throw std::runtime_error("Memory not allocated.");
        return E_FAIL;
    }

    CRC::CheckCuda(cudaFree((void**)&mem.host->Mem()));
    CRC::CheckCuda(cudaFreeHost((void**)&mem.host));

    CRC::CheckCuda(cudaFree((void*)mem.device));
    return S_OK;
}
