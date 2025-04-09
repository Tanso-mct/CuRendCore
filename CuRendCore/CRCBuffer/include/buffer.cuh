#pragma once

#include "CRCBuffer/include/config.h"

#include "WinAppCore/include/WACore.h"

#include "CRCInterface/include/resource.h"
#include "CRCInterface/include/factory.h"
#include "CRCInterface/include/device.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CRC
{

class CRC_BUFFER Buffer : public IResource, public WACore::IContainable, public IProduct
{
private:
    std::unique_ptr<IDevice>& device_;

    bool isValid_ = false;
    const UINT type_ = 0;

    UINT size_ = 0;
    void* dData_ = nullptr;
    void* hData_ = nullptr;

public:
    Buffer() = delete;
    Buffer(std::unique_ptr<IDevice>& device, UINT cpuRWFlags, UINT gpuRWFlags);
    ~Buffer() override;

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

    HRESULT GetDataDeviceSide(UINT& size, void** data) override;
    HRESULT GetDataHostSide(UINT& size, void** data) override;
};

class CRC_BUFFER BufferDesc : public IDesc
{
public:
    BufferDesc() = delete;
    BufferDesc(std::unique_ptr<IDevice>& device);
    ~BufferDesc() override = default;

    std::unique_ptr<IDevice>& device_;

    UINT cpuRWFlags = 0;
    UINT gpuRWFlags = 0;

    UINT size_ = 0;
};

class CRC_BUFFER BufferFactory : public IFactory
{
public:
    ~BufferFactory() override = default;
    std::unique_ptr<IProduct> Create(IDesc& desc) const override;
};

}