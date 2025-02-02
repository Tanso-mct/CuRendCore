#pragma once

#include "CRC_config.h"

#include <memory>
#include <Windows.h>

class CRC_API CRCComponent
{
public:
    virtual ~CRCComponent() = default;
};

class CRC_API CRCFactory
{
public:
    virtual ~CRCFactory() = default;
    virtual std::unique_ptr<CRCComponent> Create() = 0;
};

class CRC_API CRCData
{
public:
    virtual ~CRCData() = default;
};

class CRC_API CRCContainer
{
public:
    virtual ~CRCContainer() = default;

    virtual int Add(std::unique_ptr<CRCData>& data) = 0;
    virtual HRESULT Remove(int id) = 0;

    virtual CRCData* Get(int id) = 0;
    virtual int GetSize() = 0;

    virtual void Clear() = 0;
};

class CRC_API CRCLayout
{
public:
    virtual ~CRCLayout() = default;
};

class CRC_API CRCBuffer
{
public:
    virtual ~CRCBuffer() = default;

    virtual void Create(std::unique_ptr<CRCData>& data, CRCLayout& layout) = 0;
    virtual void Update(std::unique_ptr<CRCData>& data, CRCLayout& layout) = 0;
    virtual void Destroy() = 0;
};

class CRC_API CRCShader
{
public:
    virtual ~CRCShader() = default;
};

class CRC_API CRCRasterizer
{
public:
    virtual ~CRCRasterizer() = default;
};