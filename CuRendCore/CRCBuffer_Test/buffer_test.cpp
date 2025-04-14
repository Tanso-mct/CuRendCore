#include "pch.h"

#include "CRCBuffer/include/buffer.cuh"
#pragma comment(lib, "CRCBuffer.lib")

TEST(CRCBuffer, CreateAndRelease_CpuRW_GpuRW)
{
    CRC::BufferDesc desc;
    desc.cpuRWFlags_ = (UINT)CRC::RW_FLAG::READ | (UINT)CRC::RW_FLAG::WRITE;
    desc.gpuRWFlags_ = (UINT)CRC::RW_FLAG::READ | (UINT)CRC::RW_FLAG::WRITE;
    desc.size_ = sizeof(float) * 1024;

    CRC::BufferFactory factory;
    std::unique_ptr<CRC::IProduct> buffer = factory.Create(desc);

    {
        WACore::RevertCast<CRC::IUnknown, CRC::IProduct> unknown(buffer);
        unknown()->Release();
    }
}

TEST(CRCBuffer, CreateAndRelease_CpuRW)
{
    CRC::BufferDesc desc;
    desc.cpuRWFlags_ = (UINT)CRC::RW_FLAG::READ | (UINT)CRC::RW_FLAG::WRITE;
    desc.size_ = sizeof(float) * 1024;

    CRC::BufferFactory factory;
    std::unique_ptr<CRC::IProduct> buffer = factory.Create(desc);

    {
        WACore::RevertCast<CRC::IUnknown, CRC::IProduct> unknown(buffer);
        unknown()->Release();
    }
}

TEST(CRCBuffer, CreateAndRelease_GpuRW)
{
    CRC::BufferDesc desc;
    desc.gpuRWFlags_ = (UINT)CRC::RW_FLAG::READ | (UINT)CRC::RW_FLAG::WRITE;
    desc.size_ = sizeof(float) * 1024;

    CRC::BufferFactory factory;
    std::unique_ptr<CRC::IProduct> buffer = factory.Create(desc);

    {
        WACore::RevertCast<CRC::IUnknown, CRC::IProduct> unknown(buffer);
        unknown()->Release();
    }
}