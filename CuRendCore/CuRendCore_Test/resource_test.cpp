#include "pch.h"
#include "CuRendCore/include/CuRendCore.h"

TEST(CuRendCore, CreateBuffer)
{
    Microsoft::WRL::ComPtr<ID3D11Device> device;

    CRCBufferFactoryL0_0 factory;
    CRC_BUFFER_DESC desc(device);

    D3D11_BUFFER_DESC& bufferDesc = desc.desc_;
    bufferDesc.ByteWidth = 1024;

    std::unique_ptr<ICRCContainable> buffer = factory.Create(desc);

    EXPECT_NE(buffer.get(), nullptr);
}

TEST(CuRendCore, CreateTexture2D)
{
    Microsoft::WRL::ComPtr<ID3D11Device> device;

    CRCTexture2DFactoryL0_0 factory;
    CRC_TEXTURE2D_DESC desc(device);

    D3D11_TEXTURE2D_DESC& textureDesc = desc.desc_;
    textureDesc.Width = 1920;
    textureDesc.Height = 1080;
    textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;

    std::unique_ptr<ICRCContainable> texture = factory.Create(desc);

    EXPECT_NE(texture.get(), nullptr);
}
