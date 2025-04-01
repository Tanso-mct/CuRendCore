#pragma once

#include "CuRendCore/include/config.h"
#include "CuRendCore/include/WinAppCore/WACore.h"

#include <Windows.h>
#include <memory>

class CRC_API IDESC
{
public:
    virtual ~IDESC() = default;
};

class CRC_API ICRCFactory
{
public:
    virtual ~ICRCFactory() = default;
    virtual std::unique_ptr<WACore::IContainable> Create(IDESC& desc) const = 0;
};