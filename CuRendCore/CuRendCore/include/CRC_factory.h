﻿#pragma once

#include "CRC_config.h"
#include "CRC_container.h"

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
    virtual std::unique_ptr<ICRCContainable> Create(IDESC& desc) const = 0;
};