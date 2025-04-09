#pragma once

#include "CRCTexture/include/config.h"

#include "WinAppCore/include/WACore.h"

#include "CRCInterface/include/texture.h"
#include "CRCInterface/include/factory.h"
#include "CRCInterface/include/device.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CRC
{

class CRC_TEXTURE Texture2d : public ITexture, public WACore::IContainable, public IProduct
{
private:
    std::unique_ptr<IDevice>& device_;

    bool isValid_ = false;
    const UINT type_;
    const cudaChannelFormatDesc channelDesc_;

    cudaArray* dArray_ = nullptr;
    unsigned long long object_ = 0;
    void* hPtr_ = nullptr;

    UINT stride_ = 0;
    UINT width_ = 0;
    UINT height_ = 0;

public:
    Texture2d() = delete;
    Texture2d(std::unique_ptr<IDevice>& device, UINT cpuRWFlags, UINT gpuRWFlags, cudaChannelFormatDesc channelDesc);
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

    //*************************************************************************************************************** */
    // ITexture
    //*************************************************************************************************************** */
    
    HRESULT GetSize(UINT& size) override;
    HRESULT GetStride(UINT& stride) override;;
    HRESULT GetWidth(UINT& width) override;;
    HRESULT GetHeight(UINT& height) override;;
    HRESULT GetFormat(cudaChannelFormatDesc& channelDesc) override;;
    HRESULT GetArray(cudaArray** array) override;;
    HRESULT GetObj(unsigned long long* object) override;;
    HRESULT GetDataHostSide(void** data) override;;

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
    cudaChannelFormatDesc channelDesc_;

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