#pragma once

#include "CudaCore/include/config.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

namespace CudaCore
{

CUDA_CORE bool CheckCudaErr(cudaError_t call);

CUDA_CORE void Malloc(void **devPtr, size_t size);
CUDA_CORE bool TryMalloc(void **devPtr, size_t size);
CUDA_CORE void Free(void **devPtr);
CUDA_CORE bool TryFree(void **devPtr);

CUDA_CORE void MallocHost(void **ptr, size_t size);
CUDA_CORE bool TryMallocHost(void **ptr, size_t size);
CUDA_CORE void FreeHost(void **ptr);
CUDA_CORE bool TryFreeHost(void **ptr);

CUDA_CORE void MallocManaged(void **devPtr, size_t size);
CUDA_CORE bool TryMallocManaged(void **devPtr, size_t size);
CUDA_CORE void FreeManaged(void **devPtr);
CUDA_CORE bool TryFreeManaged(void **devPtr);

CUDA_CORE void MallocArray
(
    cudaArray_t *array, const cudaChannelFormatDesc *desc, 
    size_t width, size_t height, unsigned int flags
);
CUDA_CORE bool TryMallocArray
(
    cudaArray_t *array, const cudaChannelFormatDesc *desc, 
    size_t width, size_t height, unsigned int flags
);
CUDA_CORE void FreeArray(cudaArray_t *array);
CUDA_CORE bool TryFreeArray(cudaArray_t *array);

CUDA_CORE void Memcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind);
CUDA_CORE bool TryMemcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind);

CUDA_CORE void Memset(void *devPtr, int value, size_t size);
CUDA_CORE bool TryMemset(void *devPtr, int value, size_t size);

CUDA_CORE void CreateSurfaceObj(cudaSurfaceObject_t *obj, const cudaResourceDesc *rResDesc);
CUDA_CORE bool TryCreateSurfaceObj(cudaSurfaceObject_t *obj, const cudaResourceDesc *rResDesc);
CUDA_CORE void DestroySurfaceObj(cudaSurfaceObject_t *obj);
CUDA_CORE bool TryDestroySurfaceObj(cudaSurfaceObject_t *obj);

CUDA_CORE void CreateTextureObj
(
    cudaTextureObject_t *obj, 
    const cudaResourceDesc *resDesc,
    const cudaTextureDesc *texDesc, 
    const cudaResourceViewDesc *resViewDesc
);
CUDA_CORE bool TryCreateTextureObj
(
    cudaTextureObject_t *obj, 
    const cudaResourceDesc *resDesc,
    const cudaTextureDesc *texDesc, 
    const cudaResourceViewDesc *resViewDesc
);
CUDA_CORE void DestroyTextureObj(cudaTextureObject_t *obj);
CUDA_CORE bool TryDestroyTextureObj(cudaTextureObject_t *obj);

CUDA_CORE void RegisterResource
(
    cudaGraphicsResource_t *resource, ID3D11Resource *d3d11Resource, 
    unsigned int flags = cudaGraphicsRegisterFlagsNone
);
CUDA_CORE bool TryRegisterResource
(
    cudaGraphicsResource_t *resource, ID3D11Resource *d3d11Resource, 
    unsigned int flags = cudaGraphicsRegisterFlagsNone
);
CUDA_CORE void UnregisterResource(cudaGraphicsResource_t *resource);
CUDA_CORE bool TryUnregisterResource(cudaGraphicsResource_t *resource);

CUDA_CORE void MapResource(int count, cudaGraphicsResource_t *resource, cudaStream_t stream = 0);
CUDA_CORE bool TryMapResource(int count, cudaGraphicsResource_t *resource, cudaStream_t stream = 0);
CUDA_CORE void UnmapResource(int count, cudaGraphicsResource_t *resource, cudaStream_t stream = 0);
CUDA_CORE bool TryUnmapResource(int count, cudaGraphicsResource_t *resource, cudaStream_t stream = 0);

CUDA_CORE void GetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource);
CUDA_CORE bool TryGetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource);

CUDA_CORE void GetMappedArray
(
    cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel
);
CUDA_CORE bool TryGetMappedArray
(
    cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel
);


} // namespace CudaCore