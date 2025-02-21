#include "pch.h"
#include "CuRendCore/include/CuRendCore.h"

TEST(CuRendCore, CreateDeviceAndSwapChain) 
{
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<IDXGISwapChain> swapChain;

    CRC::CreateDeviceAndSwapChain(GetConsoleWindow(), device, swapChain);

    EXPECT_NE(device.Get(), nullptr);
    EXPECT_NE(swapChain.Get(), nullptr);
}