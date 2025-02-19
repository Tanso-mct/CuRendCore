#pragma once

#include "CRC_config.h"
#include "CRC_container.h"
#include "CRC_resource.cuh"
#include "CRC_factory.h"
#include "CRC_buffer.cuh"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CRC_API CRC_TEXTURE_DESC : public IDESC
{
public:
    ~CRC_TEXTURE_DESC() override = default;

    UINT width_;
    UINT height_;
    UINT mipLevels_;
    UINT arraySize_;
    CRC_FORMAT format_;
    UINT sampleCount_;
    UINT sampleQuality_;
    CRC_USAGE usage_;
    UINT bindFlags_;
    UINT cpuAccessFlags_;
    UINT miscFlags_;
};

class CRC_API CRCTextureFactory : public ICRCFactory
{
public:
    ~CRCTextureFactory() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API ICRCTexture2D
{
public:
    virtual ~ICRCTexture2D() = default;
};

class CRC_API CRCTexture2D : public ICRCContainable, public ICRCResource, public ICRCTexture2D
{
public:
    ~CRCTexture2D() override = default;

    virtual void* GetMem() const override;
    virtual std::size_t GetSize() const override;
};

class CRC_API CRCID3D11Texture2D : public ICRCContainable, public ICRCResource, public ICRCTexture2D
{
private:
    Microsoft::WRL::ComPtr<ID3D11Texture2D> d3d11Texture2D;

public:
    ~CRCID3D11Texture2D() override = default;

    virtual void* GetMem() const override;
    virtual std::size_t GetSize() const override;
};
