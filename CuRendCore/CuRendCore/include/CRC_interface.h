#pragma once

#include "CRC_config.h"

#include <memory>
#include <Windows.h>

class CRC_API ICRCComponent
{
public:
    virtual ~ICRCComponent() = default;
};

class CRC_API ICRCFactory
{
public:
    virtual ~ICRCFactory() = default;
    virtual std::unique_ptr<ICRCComponent> Create() = 0;
};

class CRC_API ICRCContainable
{
public:
    virtual ~ICRCContainable() = default;
};

class CRC_API ICRCContainer
{
public:
    virtual ~ICRCContainer() = default;

    virtual int Add(std::unique_ptr<ICRCContainable> data) = 0;
    virtual HRESULT Remove(int id) = 0;

    virtual ICRCContainable* Get(int id) = 0;
    virtual int GetSize() = 0;

    virtual void Clear() = 0;
};

class CRC_API ICRCLayout
{
public:
    virtual ~ICRCLayout() = default;
};

class CRC_API ICRCBuffer
{
public:
    virtual ~ICRCBuffer() = default;

    virtual void Create(std::unique_ptr<ICRCContainable>& data, ICRCLayout& layout) = 0;
    virtual void Update(std::unique_ptr<ICRCContainable>& data, ICRCLayout& layout) = 0;
    virtual void Destroy() = 0;
};

class CRC_API ICRCShader
{
public:
    virtual ~ICRCShader() = default;
};

class CRC_API ICRCRasterizer
{
public:
    virtual ~ICRCRasterizer() = default;
};

class CRC_API ICRCPhaseMethod : public ICRCContainable
{
public:
    virtual ~ICRCPhaseMethod() = default;

    virtual void Awake() = 0;
    virtual void Update() = 0;
    virtual void Show() = 0;
    virtual void Hide() = 0;
    virtual void End() = 0;
};