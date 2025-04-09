#pragma once

#include "CRCTexture/include/config.h"

#include "WinAppCore/include/WACore.h"

#include "CRCInterface/include/resource.h"
#include "CRCInterface/include/factory.h"
#include "CRCInterface/include/device.h"
#include "CRCInterface/include/texture.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CRC
{

class CRC_TEXTURE Texture2d : public IResource, public ITexture, public WACore::IContainable, public IProduct
{
private:
    std::unique_ptr<IDevice>& device_;

    bool isValid_ = false;
    const UINT type_ = 0;

    cudaArray* dArray_ = nullptr;
    unsigned long long object_ = 0;
    void* hPtr_ = nullptr;

    UINT stride_ = 0;
    UINT width_ = 0;
    UINT height_ = 0;

public:
    Texture2d() = delete;
    Texture2d(std::unique_ptr<IDevice>& device, UINT cpuRWFlags, UINT gpuRWFlags);
    ~Texture2d() override;

    //*************************************************************************************************************** */
    // IUnknown
    /**************************************************************************************************************** */

    HRESULT Release() override;

    //*************************************************************************************************************** */
    // IDeviceChild
    //*************************************************************************************************************** */

    HRESULT GetDevice(std::unique_ptr<IDevice>*& device) override;

    //*************************************************************************************************************** */
    // IResource
    //*************************************************************************************************************** */

    HRESULT GetType(UINT& type) override;
    void GetDesc(IDesc *desc) override;

    HRESULT GetDataDeviceSide(UINT& size, void** data) override; // Obtain a CUDA object
    HRESULT GetDataHostSide(UINT& size, void** data) override;

    //*************************************************************************************************************** */
    // ITexture
    //*************************************************************************************************************** */
    
    HRESULT GetArray(UINT& size, cudaArray** array) override; // Obtain a CUDA array object
};

class CRC_TEXTURE Texture2dDesc : public IDesc
{
public:
    Texture2dDesc() = delete;
    Texture2dDesc(std::unique_ptr<IDevice>& device);
    ~Texture2dDesc() override = default;

    std::unique_ptr<IDevice>& device_;

    UINT cpuRWFlags_ = 0;
    UINT gpuRWFlags_ = 0;

    UINT stride_ = 0;
    UINT width_ = 0;
    UINT height_ = 0;
};

class CRC_TEXTURE Texture2dFactory : public IFactory
{
public:
    ~Texture2dFactory() override = default;
    std::unique_ptr<IProduct> Create(IDesc& desc) const override;
};

}