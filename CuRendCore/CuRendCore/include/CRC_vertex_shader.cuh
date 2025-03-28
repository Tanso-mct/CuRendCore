#pragma once

#include "CRC_config.h"

#include "CRC_container.h"
#include "CRC_factory.h"
#include "CRC_shader.cuh"

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CRC_VS_INPUT void* vsInput
#define CRC_VS_OUTPUT void* vsOutput

namespace CRC
{

using VertexShaderKernel = void (*)(CRC_SHADER_REGISTER, CRC_VS_INPUT, CRC_VS_OUTPUT);

}

CRC_API class CRC_VERTEX_SHADER_DESC : public IDESC
{
public:
    CRC_VERTEX_SHADER_DESC(CRC::VertexShaderKernel kernel) : kernel_(kernel) {};
    ~CRC_VERTEX_SHADER_DESC() override = default;

    const CRC::VertexShaderKernel kernel_ = nullptr;
};

CRC_API class CRC_ID3D11_VERTEX_SHADER_DESC : public IDESC
{
public:
    CRC_ID3D11_VERTEX_SHADER_DESC(
        Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device,
        const void *shaderBytecode,
        SIZE_T bytecodeLength,
        ID3D11ClassLinkage *classLinkage
    ) : d3d11Device_(d3d11Device),
    shaderBytecode_(shaderBytecode), bytecodeLength_(bytecodeLength), classLinkage_(classLinkage) {};

    ~CRC_ID3D11_VERTEX_SHADER_DESC() override = default;

    Microsoft::WRL::ComPtr<ID3D11Device>& d3d11Device_;

    const void *shaderBytecode_ = nullptr;
    SIZE_T bytecodeLength_ = 0;
    ID3D11ClassLinkage *classLinkage_ = nullptr;
};

CRC_API class ICRCVertexShader
{
public:
    virtual ~ICRCVertexShader() = default;
};

class CRC_API CRCVertexShaderFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCVertexShaderFactoryL0_0() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCVertexShader : public ICRCContainable, public ICRCShader, public ICRCVertexShader
{
private:
    CRC::VertexShaderKernel kernel_ = nullptr;
    UINT shaderType_ = 0;

public:
    CRCVertexShader() = delete;

    CRCVertexShader(CRC_VERTEX_SHADER_DESC& desc);
    virtual ~CRCVertexShader() override;

    virtual void RunKernel(dim3 gridDim, dim3 blockDim, CRC_SHADER_REGISTER, CRC_VS_INPUT, CRC_VS_OUTPUT) const;

    // ICRCShader
    virtual HRESULT GetShaderType(UINT& shaderType) override;
};

class CRC_API CRCID3D11VertexShaderFactoryL0_0 : public ICRCFactory
{
public:
    ~CRCID3D11VertexShaderFactoryL0_0() override = default;
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const override;
};

class CRC_API CRCID3D11VertexShader : public ICRCContainable, public ICRCShader, public ICRCVertexShader
{
private:
    Microsoft::WRL::ComPtr<ID3D11VertexShader> vertexShader_ = nullptr;
    UINT shaderType_ = 0;

public:
    CRCID3D11VertexShader() = delete;

    CRCID3D11VertexShader(CRC_ID3D11_VERTEX_SHADER_DESC& desc);
    virtual ~CRCID3D11VertexShader() override;

    Microsoft::WRL::ComPtr<ID3D11VertexShader>& Get() { return vertexShader_; }

    // ICRCShader
    virtual HRESULT GetShaderType(UINT& shaderType) override;
};