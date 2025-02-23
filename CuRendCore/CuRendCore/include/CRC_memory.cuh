#pragma once

#include "CRC_config.h"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Windows.h>
#include <memory>

class CRC_API ICRCMem
{
public:
    virtual ~ICRCMem() = default;

    virtual void* Get() = 0;
    virtual void*& GetPtr() = 0;

    virtual const UINT& GetByteWidth() const = 0;
    virtual const UINT& GetPitch() const = 0;
    virtual const UINT& GetSlicePitch() const = 0;

    virtual void Malloc(const UINT& byteWidth, const UINT& pitch, const UINT& slicePitch) = 0;
    virtual void Free() = 0;
};

class CRC_API CRCHostMem : public ICRCMem
{
private:
    void* ptr_ = nullptr;
    UINT byteWidth_ = 0;
    UINT pitch_ = 0;
    UINT slicePitch_ = 0;

public:
    CRCHostMem() = default;
    ~CRCHostMem() override;

    virtual void* Get() override { return ptr_; }
    virtual void*& GetPtr() override { return ptr_; }

    const UINT& GetByteWidth() const override { return byteWidth_; }
    const UINT& GetPitch() const override { return pitch_; }
    const UINT& GetSlicePitch() const override { return slicePitch_; }

    void Malloc(const UINT& byteWidth, const UINT& pitch, const UINT& slicePitch) override;
    void Free() override;
};

class CRC_API CRCDeviceMem : public ICRCMem
{
private:
    void* ptr_ = nullptr;
    UINT byteWidth_ = 0;
    UINT pitch_ = 0;
    UINT slicePitch_ = 0;

public:
    CRCDeviceMem() = default;
    ~CRCDeviceMem() override;

    virtual void* Get() override { return ptr_; }
    virtual void*& GetPtr() override { return ptr_; }

    const UINT& GetByteWidth() const override { return byteWidth_; }
    const UINT& GetPitch() const override { return pitch_; }
    const UINT& GetSlicePitch() const override { return slicePitch_; }

    void Malloc(const UINT& byteWidth, const UINT& pitch, const UINT& slicePitch) override;
    void Free() override;
};