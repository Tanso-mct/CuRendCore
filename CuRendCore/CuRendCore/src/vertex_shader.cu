#include "CuRendCore/include/pch.h"
#include "CuRendCore/include/funcs.cuh"

#include "CuRendCore/include/vertex_shader.cuh"

std::unique_ptr<ICRCContainable> CRCVertexShaderFactoryL0_0::Create(IDESC &desc) const
{
    CRC_VERTEX_SHADER_DESC* vertexShaderDesc = CRC::As<CRC_VERTEX_SHADER_DESC>(&desc);
    if (!vertexShaderDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to create vertex shader from desc. Desc is not CRC_VERTEX_SHADER_DESC.");
#endif
        return nullptr;
    }

    std::unique_ptr<CRCVertexShader> vertexShader = std::make_unique<CRCVertexShader>(*vertexShaderDesc);
    return vertexShader;
}

CRCVertexShader::CRCVertexShader(CRC_VERTEX_SHADER_DESC &desc)
{
    kernel_ = desc.kernel_;

#ifndef NDEBUG
    CRC::Cout("Created CRC Vertex Shader");
#endif
}

CRCVertexShader::~CRCVertexShader()
{
#ifndef NDEBUG
    CRC::Cout("Destroyed CRC Vertex Shader");
#endif
}

void CRCVertexShader::RunKernel(dim3 gridDim, dim3 blockDim, CRC_SHADER_REGISTER, CRC_VS_INPUT, CRC_VS_OUTPUT) const
{
    kernel_<<<gridDim, blockDim>>>(registerBuffers, registerTextures, registerSamplers, vsInput, vsOutput);
}

HRESULT CRCVertexShader::GetShaderType(UINT &shaderType)
{
    shaderType = shaderType_;
    return S_OK;
}

std::unique_ptr<ICRCContainable> CRCID3D11VertexShaderFactoryL0_0::Create(IDESC &desc) const
{
    CRC_ID3D11_VERTEX_SHADER_DESC* vertexShaderDesc = CRC::As<CRC_ID3D11_VERTEX_SHADER_DESC>(&desc);
    if (!vertexShaderDesc)
    {
#ifndef NDEBUG
        CRC::CoutWarning("Failed to create vertex shader from desc. Desc is not CRC_ID3D11_VERTEX_SHADER_DESC.");
#endif
        return nullptr;
    }

    std::unique_ptr<CRCID3D11VertexShader> vertexShader = std::make_unique<CRCID3D11VertexShader>(*vertexShaderDesc);
    return vertexShader;
}

CRCID3D11VertexShader::CRCID3D11VertexShader(CRC_ID3D11_VERTEX_SHADER_DESC &desc)
{
    HRESULT hr = desc.d3d11Device_->CreateVertexShader
    (
        desc.shaderBytecode_, desc.bytecodeLength_, desc.classLinkage_, vertexShader_.GetAddressOf()
    );

    if (FAILED(hr))
    {
#ifndef NDEBUG
        CRC::CoutError("Failed to create ID3D11VertexShader.");
#endif
        throw std::runtime_error("Failed to create ID3D11VertexShader.");
    }

#ifndef NDEBUG
    CRC::Cout("Created ID3D11 Vertex Shader");
#endif
}

CRCID3D11VertexShader::~CRCID3D11VertexShader()
{
#ifndef NDEBUG
    CRC::Cout("Destroyed ID3D11 Vertex Shader");
#endif
}

HRESULT CRCID3D11VertexShader::GetShaderType(UINT &shaderType)
{
    shaderType = shaderType_;
    return S_OK;
}
