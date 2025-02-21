#include "pch.h"

#include "CuRendCore/include/CuRendCore.h"

TEST(TestCaseName, TestName) 
{
    std::cout << std::endl;

    Microsoft::WRL::ComPtr<ID3D11Device> device;

    {
        CRC::CoutMsg("Creat buffer");
        CRCBufferFactory factory;
        CRC_BUFFER_DESC desc(device);
        desc.ByteWidth() = 1024;

        std::unique_ptr<ICRCContainable> buffer = factory.Create(desc);
    }

    std::cout << std::endl;

    {
        CRC::CoutMsg("Creat texture2D");
        CRCTexture2DFactory factory;
        CRC_TEXTURE2D_DESC desc(device);
        desc.Width() = 1920;
        desc.Height() = 1080;
        desc.Format() = DXGI_FORMAT_R8G8B8A8_UNORM;

        std::unique_ptr<ICRCContainable> texture = factory.Create(desc);
    }

    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
