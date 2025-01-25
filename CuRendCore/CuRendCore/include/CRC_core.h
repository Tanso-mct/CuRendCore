#pragma once

#include "CRC_config.h"
#include "CRC_interface.h"

class CRCCore
{
private:
    virtual void Shutdown();

public:
    CRCCore();
    virtual ~CRCCore();

    virtual void Initialize();
    virtual void Run();
};