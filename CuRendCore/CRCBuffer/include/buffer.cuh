﻿#pragma once

#include "CRCBuffer/include/config.h"

#include "WinAppCore/include/WACore.h"

#include "CRCInterface/include/buffer.h"
#include "CRCInterface/include/factory.h"
#include "CRCInterface/include/device.h"

#include <cuda_d3d11_interop.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CRC
{

class CRC_BUFFER Buffer : public IBuffer, public WACore::IContainable, public IProduct
{
private:
    bool isValid_ = false;
    const UINT type_ = 0;

    const UINT size_ = 0;
    void* dData_ = nullptr;
    void* hData_ = nullptr;

public:
    Buffer() = delete;
    Buffer(UINT cpuRWFlags, UINT gpuRWFlags, UINT size);
    ~Buffer() override;

    //*************************************************************************************************************** */
    // IUnknown
    /**************************************************************************************************************** */

    HRESULT Release() override;

    //*************************************************************************************************************** */
    // IResource
    //*************************************************************************************************************** */

    HRESULT GetType(UINT& type) const override;
    void GetDesc(IDesc *desc) const override;

    //*************************************************************************************************************** */
    // IBuffer
    //*************************************************************************************************************** */

    HRESULT GetSize(UINT& size) const override;
    HRESULT GetDataDeviceSide(void**& data) override;
    HRESULT GetDataHostSide(void**& data) override;
};

class CRC_BUFFER BufferDesc : public IDesc
{
public:
    BufferDesc();
    ~BufferDesc() override = default;

    UINT cpuRWFlags_ = 0;
    UINT gpuRWFlags_ = 0;

    UINT size_ = 0;
};

class CRC_BUFFER BufferFactory : public IFactory
{
public:
    ~BufferFactory() override = default;
    std::unique_ptr<IProduct> Create(IDesc& desc) const override;
};

}