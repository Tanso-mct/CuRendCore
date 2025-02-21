#include "pch.h"
#include "CuRendCore/include/CuRendCore.h"

TEST(CuRendCore, CreateBuffer)
{
    Microsoft::WRL::ComPtr<ID3D11Device> device;

    CRCBufferFactory factory;
    CRC_BUFFER_DESC desc(device);
    desc.ByteWidth() = 1024;

    std::unique_ptr<ICRCContainable> buffer = factory.Create(desc);

    EXPECT_NE(buffer.get(), nullptr);
}

TEST(CuRendCore, CreateTexture2D)
{
    Microsoft::WRL::ComPtr<ID3D11Device> device;

    CRCTexture2DFactory factory;
    CRC_TEXTURE2D_DESC desc(device);
    desc.Width() = 1920;
    desc.Height() = 1080;
    desc.Format() = DXGI_FORMAT_R8G8B8A8_UNORM;

    std::unique_ptr<ICRCContainable> texture = factory.Create(desc);

    EXPECT_NE(texture.get(), nullptr);
}
