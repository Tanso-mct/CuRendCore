#include "CudaCore/include/pch.h"
#include "CudaCore/include/memory.cuh"

#include "CudaCore/include/console.cuh"

CUDA_CORE bool CudaCore::CheckCudaErr(cudaError_t call)
{
    if (call != cudaSuccess) return false;
    return true;
}

namespace
{

std::string GetPointerAddress(const void *ptr)
{
    std::stringstream ss;
    ss << std::hex << std::uppercase << std::setw(16) << std::setfill('0') << reinterpret_cast<uintptr_t>(ptr);
    return ss.str();
}

}

CUDA_CORE void CudaCore::Malloc(void **devPtr, size_t size)
{
    if (CheckCudaErr(cudaMalloc(devPtr, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA malloc succeeded.", 
            "address:" + ::GetPointerAddress(*devPtr),
            "size:" + std::to_string(size),
        });
#endif
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA malloc failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::Malloc failed");
    }
}

CUDA_CORE bool CudaCore::TryMalloc(void **devPtr, size_t size)
{
    if (CheckCudaErr(cudaMalloc(devPtr, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA malloc succeeded.", 
            "address:" + ::GetPointerAddress(*devPtr),
            "size:" + std::to_string(size),
        });
#endif
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA malloc failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::Free(void **devPtr)
{
    if (CheckCudaErr(cudaFree(*devPtr)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA free succeeded.", 
            "address:" + ::GetPointerAddress(*devPtr),
        });
#endif
        *devPtr = nullptr;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA free failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::Free failed");
    }
}
CUDA_CORE bool CudaCore::TryFree(void **devPtr)
{
    if (CheckCudaErr(cudaFree(*devPtr)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA free succeeded.", 
            "address:" + ::GetPointerAddress(*devPtr),
        });
#endif
        *devPtr = nullptr;
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA free failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::MallocHost(void **ptr, size_t size)
{
    if (CheckCudaErr(cudaMallocHost(ptr, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA mallocHost succeeded.", 
            "address:" + ::GetPointerAddress(*ptr),
            "size:" + std::to_string(size),
        });
#endif
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA mallocHost failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::MallocHost failed");
    }
}

CUDA_CORE bool CudaCore::TryMallocHost(void **ptr, size_t size)
{
    if (CheckCudaErr(cudaMallocHost(ptr, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA mallocHost succeeded.", 
            "address:" + ::GetPointerAddress(*ptr),
            "size:" + std::to_string(size),
        });
#endif
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA mallocHost failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::FreeHost(void **ptr)
{
    if (CheckCudaErr(cudaFreeHost(*ptr)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA freeHost succeeded.", 
            "address:" + ::GetPointerAddress(*ptr),
        });
#endif
        *ptr = nullptr;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA freeHost failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::FreeHost failed");
    }
}

CUDA_CORE bool CudaCore::TryFreeHost(void **ptr)
{
    if (CheckCudaErr(cudaFreeHost(*ptr)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA freeHost succeeded.", 
            "address:" + ::GetPointerAddress(*ptr),
        });
#endif
        *ptr = nullptr;
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA freeHost failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::MallocManaged(void **devPtr, size_t size)
{
    if (CheckCudaErr(cudaMallocManaged(devPtr, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA mallocManaged succeeded.", 
            "address:" + ::GetPointerAddress(*devPtr),
            "size:" + std::to_string(size),
        });
#endif
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA mallocManaged failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::MallocManaged failed");
    }
}

CUDA_CORE bool CudaCore::TryMallocManaged(void **devPtr, size_t size)
{
    if (CheckCudaErr(cudaMallocManaged(devPtr, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA mallocManaged succeeded.", 
            "address:" + ::GetPointerAddress(*devPtr),
            "size:" + std::to_string(size),
        });
#endif
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA mallocManaged failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::FreeManaged(void **devPtr)
{
    if (CheckCudaErr(cudaFree(*devPtr)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA freeManaged succeeded.", 
            "address:" + ::GetPointerAddress(*devPtr),
        });
#endif
        *devPtr = nullptr;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA freeManaged failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::FreeManaged failed");
    }
}

CUDA_CORE bool CudaCore::TryFreeManaged(void **devPtr)
{
    if (CheckCudaErr(cudaFree(*devPtr)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA freeManaged succeeded.", 
            "address:" + ::GetPointerAddress(*devPtr),
        });
#endif
        *devPtr = nullptr;
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA freeManaged failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::MallocArray
(
    cudaArray_t *array, const cudaChannelFormatDesc *desc, 
    size_t width, size_t height, unsigned int flags
){
    if (CheckCudaErr(cudaMallocArray(array, desc, width, height, flags)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA mallocArray succeeded.", 
            "address:" + ::GetPointerAddress(*array),
            "width:" + std::to_string(width),
            "height:" + std::to_string(height),
        });
#endif
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA mallocArray failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::MallocArray failed");
    }
}

CUDA_CORE bool CudaCore::TryMallocArray
(
    cudaArray_t *array, const cudaChannelFormatDesc *desc, 
    size_t width, size_t height, unsigned int flags
){
    if (CheckCudaErr(cudaMallocArray(array, desc, width, height, flags)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA mallocArray succeeded.", 
            "address:" + ::GetPointerAddress(*array),
            "width:" + std::to_string(width),
            "height:" + std::to_string(height),
        });
#endif
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA mallocArray failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::FreeArray(cudaArray_t *array)
{
    if (CheckCudaErr(cudaFreeArray(*array)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA freeArray succeeded.", 
            "address:" + ::GetPointerAddress(*array),
        });
#endif
        *array = nullptr;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA freeArray failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::FreeArray failed");
    }
}

CUDA_CORE bool CudaCore::TryFreeArray(cudaArray_t *array)
{
    if (CheckCudaErr(cudaFreeArray(*array)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA freeArray succeeded.", 
            "address:" + ::GetPointerAddress(*array),
        });
#endif
        *array = nullptr;
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA freeArray failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::Memcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind)
{
    if (CheckCudaErr(cudaMemcpy(dst, src, size, kind)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA memcpy succeeded.", 
            "dst address:" + ::GetPointerAddress(dst),
            "src address:" + ::GetPointerAddress(src),
            "size:" + std::to_string(size),
        });
#endif
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA memcpy failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::Memcpy failed");
    }
}

CUDA_CORE bool CudaCore::TryMemcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind)
{
    if (CheckCudaErr(cudaMemcpy(dst, src, size, kind)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA memcpy succeeded.", 
            "dst address:" + ::GetPointerAddress(dst),
            "src address:" + ::GetPointerAddress(src),
            "size:" + std::to_string(size),
        });
#endif
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA memcpy failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::Memset(void *devPtr, int value, size_t size)
{
    if (CheckCudaErr(cudaMemset(devPtr, value, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA memset succeeded.", 
            "address:" + ::GetPointerAddress(devPtr),
            "value:" + std::to_string(value),
            "size:" + std::to_string(size),
        });
#endif
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA memset failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::Memset failed");
    }
}

CUDA_CORE bool CudaCore::TryMemset(void *devPtr, int value, size_t size)
{
    if (CheckCudaErr(cudaMemset(devPtr, value, size)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA memset succeeded.", 
            "address:" + ::GetPointerAddress(devPtr),
            "value:" + std::to_string(value),
            "size:" + std::to_string(size),
        });
#endif
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA memset failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::CreateSurfaceObj(cudaSurfaceObject_t *obj, const cudaResourceDesc *rResDesc)
{
    if (CheckCudaErr(cudaCreateSurfaceObject(obj, rResDesc)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA create surface object succeeded.", 
            "object:" + std::to_string(*obj),
            "array address:" + ::GetPointerAddress(rResDesc->res.array.array),
        });
#endif
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA create surface object failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::CreateSurfaceObj failed");
    }
}

CUDA_CORE bool CudaCore::TryCreateSurfaceObj(cudaSurfaceObject_t *obj, const cudaResourceDesc *rResDesc)
{
    if (CheckCudaErr(cudaCreateSurfaceObject(obj, rResDesc)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA create surface object succeeded.", 
            "object:" + std::to_string(*obj),
            "array address:" + ::GetPointerAddress(rResDesc->res.array.array),
        });
#endif
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA create surface object failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::DestroySurfaceObj(cudaSurfaceObject_t *obj)
{
    if (CheckCudaErr(cudaDestroySurfaceObject(*obj)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA destroy surface object succeeded.", 
            "object:" + std::to_string(*obj),
        });
#endif
        *obj = 0;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA destroy surface object failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::DestroySurfaceObj failed");
    }
}

CUDA_CORE bool CudaCore::TryDestroySurfaceObj(cudaSurfaceObject_t *obj)
{
    if (CheckCudaErr(cudaDestroySurfaceObject(*obj)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA destroy surface object succeeded.", 
            "object:" + std::to_string(*obj),
        });
#endif
        *obj = 0;
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA destroy surface object failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::CreateTextureObj
(
    cudaTextureObject_t *obj, 
    const cudaResourceDesc *resDesc, 
    const cudaTextureDesc *texDesc, 
    const cudaResourceViewDesc *resViewDesc
){
    if (CheckCudaErr(cudaCreateTextureObject(obj, resDesc, texDesc, resViewDesc)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA create texture object succeeded.", 
            "object:" + std::to_string(*obj),
            "array address:" + ::GetPointerAddress(resDesc->res.array.array),
        });
#endif
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA create texture object failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::CreateTextureObj failed");
    }
}

CUDA_CORE bool CudaCore::TryCreateTextureObj
(
    cudaTextureObject_t *obj, 
    const cudaResourceDesc *resDesc, 
    const cudaTextureDesc *texDesc, 
    const cudaResourceViewDesc *resViewDesc
){
    if (CheckCudaErr(cudaCreateTextureObject(obj, resDesc, texDesc, resViewDesc)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA create texture object succeeded.", 
            "object:" + std::to_string(*obj),
            "array address:" + ::GetPointerAddress(resDesc->res.array.array),
        });
#endif
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA create texture object failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::DestroyTextureObj(cudaTextureObject_t *obj)
{
    if (CheckCudaErr(cudaDestroyTextureObject(*obj)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA destroy texture object succeeded.", 
            "object:" + std::to_string(*obj),
        });
#endif
        *obj = 0;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA destroy texture object failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::DestroyTextureObj failed");
    }
}

CUDA_CORE bool CudaCore::TryDestroyTextureObj(cudaTextureObject_t *obj)
{
    if (CheckCudaErr(cudaDestroyTextureObject(*obj)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA destroy texture object succeeded.", 
            "object:" + std::to_string(*obj),
        });
#endif
        *obj = 0;
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA destroy texture object failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::RegisterResource
(
    cudaGraphicsResource_t *resource, ID3D11Resource *d3d11Resource, unsigned int flags
){
    if (CheckCudaErr(cudaGraphicsD3D11RegisterResource(resource, d3d11Resource, flags)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA register resource succeeded.", 
            "resource address:" + ::GetPointerAddress(*resource),
            "ID3D11Resource address:" + ::GetPointerAddress(d3d11Resource),
        });
#endif
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA register resource failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::RegisterResource failed");
    }
}

CUDA_CORE bool CudaCore::TryRegisterResource(cudaGraphicsResource_t *resource, ID3D11Resource *d3d11Resource, unsigned int flags)
{
    if (CheckCudaErr(cudaGraphicsD3D11RegisterResource(resource, d3d11Resource, flags)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA register resource succeeded.", 
            "resource address:" + ::GetPointerAddress(*resource),
            "ID3D11Resource address:" + ::GetPointerAddress(d3d11Resource),
        });
#endif
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA register resource failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::UnregisterResource(cudaGraphicsResource_t *resource)
{
    if (CheckCudaErr(cudaGraphicsUnregisterResource(*resource)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA unregister resource succeeded.", 
            "resource address:" + ::GetPointerAddress(*resource),
        });
#endif
        *resource = nullptr;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA unregister resource failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::UnregisterResource failed");
    }
}

CUDA_CORE bool CudaCore::TryUnregisterResource(cudaGraphicsResource_t *resource)
{
    if (CheckCudaErr(cudaGraphicsUnregisterResource(*resource)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA unregister resource succeeded.", 
            "resource address:" + ::GetPointerAddress(*resource),
        });
#endif
        *resource = nullptr;
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA unregister resource failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::MapResource(int count, cudaGraphicsResource_t *resource, cudaStream_t stream)
{
    if (CheckCudaErr(cudaGraphicsMapResources(count, resource, stream)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA map resource succeeded.", 
            "resource address:" + ::GetPointerAddress(*resource),
            "stream address:" + ::GetPointerAddress(stream),
            "count:" + std::to_string(count),
        });
#endif
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA map resource failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::MapResource failed");
    }
}

CUDA_CORE bool CudaCore::TryMapResource(int count, cudaGraphicsResource_t *resource, cudaStream_t stream)
{
    if (CheckCudaErr(cudaGraphicsMapResources(count, resource, stream)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA map resource succeeded.", 
            "resource address:" + ::GetPointerAddress(*resource),
            "stream address:" + ::GetPointerAddress(stream),
            "count:" + std::to_string(count),
        });
#endif
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA map resource failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::UnmapResource(int count, cudaGraphicsResource_t *resource, cudaStream_t stream)
{
    if (CheckCudaErr(cudaGraphicsUnmapResources(count, resource, stream)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA unmap resource succeeded.", 
            "resource address:" + ::GetPointerAddress(*resource),
            "stream address:" + ::GetPointerAddress(stream),
            "count:" + std::to_string(count),
        });
#endif
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA unmap resource failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::UnmapResource failed");
    }
}

CUDA_CORE bool CudaCore::TryUnmapResource(int count, cudaGraphicsResource_t *resource, cudaStream_t stream)
{
    if (CheckCudaErr(cudaGraphicsUnmapResources(count, resource, stream)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA unmap resource succeeded.", 
            "resource address:" + ::GetPointerAddress(*resource),
            "stream address:" + ::GetPointerAddress(stream),
            "count:" + std::to_string(count),
        });
#endif
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA unmap resource failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::GetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource)
{
    if (CheckCudaErr(cudaGraphicsResourceGetMappedPointer(devPtr, size, resource)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA get mapped pointer succeeded.", 
            "pointer address:" + ::GetPointerAddress(*devPtr),
            "size:" + std::to_string(*size),
            "resource address:" + ::GetPointerAddress(resource),
        });
#endif
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA get mapped pointer failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::GetMappedPointer failed");
    }
}

CUDA_CORE bool CudaCore::TryGetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource)
{
    if (CheckCudaErr(cudaGraphicsResourceGetMappedPointer(devPtr, size, resource)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA get mapped pointer succeeded.", 
            "pointer address:" + ::GetPointerAddress(*devPtr),
            "size:" + std::to_string(*size),
            "resource address:" + ::GetPointerAddress(resource),
        });
#endif
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA get mapped pointer failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}

CUDA_CORE void CudaCore::GetMappedArray
(
    cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel
){
    if (CheckCudaErr(cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA get mapped array succeeded.", 
            "array address:" + ::GetPointerAddress(*array),
            "resource address:" + ::GetPointerAddress(resource),
            "array index:" + std::to_string(arrayIndex),
            "mip level:" + std::to_string(mipLevel),
        });
#endif
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA get mapped array failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        throw std::runtime_error("CudaCore::GetMappedArray failed");
    }
}

CUDA_CORE bool CudaCore::TryGetMappedArray(cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel)
{
    if (CheckCudaErr(cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel)))
    {
#ifndef NDEBUG
        CudaCore::CoutDebug
        ({
            "CUDA get mapped array succeeded.", 
            "array address:" + ::GetPointerAddress(*array),
            "resource address:" + ::GetPointerAddress(resource),
            "array index:" + std::to_string(arrayIndex),
            "mip level:" + std::to_string(mipLevel),
        });
#endif
        return true;
    }
    else
    {
        cudaError_t err = cudaGetLastError();
        CudaCore::CoutErr
        ({
            "CUDA get mapped array failed.", 
            "code:" + std::to_string(err),
            "reason:" + std::string(cudaGetErrorString(err))
        });

        return false;
    }
}
