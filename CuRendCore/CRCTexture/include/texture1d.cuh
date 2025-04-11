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

class CRC_TEXTURE Texture1d : public ITexture, public WACore::IContainable, public IProduct
{
private:
    const std::unique_ptr<IDevice>& device_;

    bool isValid_ = false;
    const UINT type_;
    const cudaChannelFormatDesc channelDesc_;

    cudaArray* dArray_ = nullptr;
    unsigned long long object_ = 0;
    void* hPtr_ = nullptr;

    const UINT stride_ = 0;
    const UINT width_ = 0;

public:
    Texture1d() = delete;
    Texture1d
    (
        std::unique_ptr<IDevice>& device, UINT cpuRWFlags, UINT gpuRWFlags, cudaChannelFormatDesc channelDesc,
        UINT stride, UINT width
    );
    ~Texture1d() override;

    //*************************************************************************************************************** */
    // IUnknown
    /**************************************************************************************************************** */

    HRESULT Release() override;

    //*************************************************************************************************************** */
    // IDeviceChild
    //*************************************************************************************************************** */

    HRESULT GetDevice(const std::unique_ptr<IDevice>*& device) const override;

    //*************************************************************************************************************** */
    // IResource
    //*************************************************************************************************************** */

    HRESULT GetType(UINT& type) const override;
    void GetDesc(IDesc *desc) const override;

    //*************************************************************************************************************** */
    // ITexture
    //*************************************************************************************************************** */
    
    HRESULT GetSize(UINT& size) const override;
    HRESULT GetStride(UINT& stride) const override;
    HRESULT GetWidth(UINT& width) const override;
    HRESULT GetHeight(UINT& height) const override;
    HRESULT GetDepth(UINT& depth) const override;
    HRESULT GetFormat(cudaChannelFormatDesc& channelDesc) const override;
    HRESULT GetArray(cudaArray** array) override;
    HRESULT GetObj(unsigned long long* object) override;
    HRESULT GetDataHostSide(void** data) override;

};

class CRC_TEXTURE Texture1dDesc : public IDesc
{
public:
    Texture1dDesc() = delete;
    Texture1dDesc(std::unique_ptr<IDevice>& device);
    ~Texture1dDesc() override = default;

    std::unique_ptr<IDevice>& device_;

    UINT cpuRWFlags_ = 0;
    UINT gpuRWFlags_ = 0;
    cudaChannelFormatDesc channelDesc_;

    UINT stride_ = 0;
    UINT width_ = 0;

    cudaTextureDesc cudaTextureDesc_;
};

class CRC_TEXTURE Texture1dFactory : public IFactory
{
public:
    ~Texture1dFactory() override = default;
    std::unique_ptr<IProduct> Create(IDesc& desc) const override;
};

}