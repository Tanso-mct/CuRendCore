#pragma once

#include <memory>
#include <Windows.h>
#include <mutex>

class CRCComponent
{
public:
    virtual ~CRCComponent() = default;
};

class CRCFactory
{
public:
    virtual ~CRCFactory() = default;
    virtual std::unique_ptr<CRCComponent> Create() = 0;
};

class CRCData
{
public:
    virtual ~CRCData() = default;
};

class CRCContainer
{
public:
    std::mutex mtx;
    virtual ~CRCContainer() = default;

    virtual int Add(std::unique_ptr<CRCData>& data) = 0;

    virtual std::unique_ptr<CRCData> Take(int id) = 0;
    virtual HRESULT Set(int id, std::unique_ptr<CRCData>& data) = 0;

    virtual UINT GetSize() = 0;

    virtual void Clear(int id) = 0;
    virtual void Clear() = 0;
};

class CRCLayout
{
public:
    virtual ~CRCLayout() = default;
};

class CRCBuffer
{
public:
    virtual ~CRCBuffer() = default;

    virtual void Create(std::unique_ptr<CRCData>& data, CRCLayout& layout) = 0;
    virtual void Update(std::unique_ptr<CRCData>& data, CRCLayout& layout) = 0;
    virtual void Destroy() = 0;
};

class CRCShader
{
public:
    virtual ~CRCShader() = default;
};

class CRCRasterizer
{
public:
    virtual ~CRCRasterizer() = default;
};