#include "CRC_pch.h"
#include "CRC_funcs.cuh"

#include "CRC_buffer.cuh"

std::unique_ptr<ICRCContainable> CRCBufferFactoryL0_0::Create(IDESC &desc) const
{
    CRC_BUFFER_DESC* bufferDesc = CRC::As<CRC_BUFFER_DESC>(&desc);
    if (!bufferDesc) return nullptr;

    std::unique_ptr<CRCBuffer> buffer = std::make_unique<CRCBuffer>(*bufferDesc);
    return buffer;
}

CRCBuffer::CRCBuffer()
{
    dMem_ = std::make_unique<CRCDeviceMem>();
}

CRCBuffer::CRCBuffer(CRC_BUFFER_DESC &desc)
{
    dMem_ = std::make_unique<CRCDeviceMem>();

    dMem_->Malloc(desc.ByteWidth(), 1, 1);
    
    if (desc.SysMem())
    {
        CRC::CheckCuda(cudaMemcpy
        (
            dMem_.get(), desc.SysMem(), desc.ByteWidth(), cudaMemcpyHostToDevice
        ));
    }

    desc_ = desc.Desc();
}

const void CRCBuffer::GetDesc(D3D11_BUFFER_DESC *dst)
{
    std::memcpy(dst, &desc_, sizeof(D3D11_BUFFER_DESC));
}

std::unique_ptr<ICRCContainable> CRCID3D11BufferFactoryL0_0::Create(IDESC &desc) const
{
    CRC_BUFFER_DESC* bufferDesc = CRC::As<CRC_BUFFER_DESC>(&desc);
    if (!bufferDesc) return nullptr;

    std::unique_ptr<CRCID3D11Buffer> buffer = std::make_unique<CRCID3D11Buffer>();

    D3D11_SUBRESOURCE_DATA* initialData = nullptr;
    if (bufferDesc->SysMem()) initialData = &bufferDesc->InitialData();

    if (!bufferDesc->d3d11Device_) throw std::runtime_error("Device not set.");

    HRESULT hr = bufferDesc->d3d11Device_->CreateBuffer
    (
        &bufferDesc->Desc(), initialData, buffer->Get().GetAddressOf()
    );
    if (FAILED(hr)) return nullptr;

    return buffer;
}

Microsoft::WRL::ComPtr<ID3D11Resource> &CRCID3D11Buffer::GetResource()
{
    Microsoft::WRL::ComPtr<ID3D11Resource> resource;
    d3d11Buffer_.As(&resource);

    return resource;
}

const UINT &CRCID3D11Buffer::GetByteWidth() const
{
    D3D11_BUFFER_DESC desc;
    d3d11Buffer_->GetDesc(&desc);

    return desc.ByteWidth;
}

const void CRCID3D11Buffer::GetDesc(D3D11_BUFFER_DESC *dst)
{
    d3d11Buffer_->GetDesc(dst);
}