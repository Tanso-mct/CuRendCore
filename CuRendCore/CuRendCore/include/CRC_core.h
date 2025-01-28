#pragma once

#include <memory>
#include <Windows.h>

#include "CRC_config.h"
#include "CRC_interface.h"

class CRCWindowContainer;

class CRCCore
{
private:
    std::unique_ptr<CRCWindowContainer> windowContainer_ = nullptr;

public:
    CRCCore();
    virtual ~CRCCore();

    virtual void Initialize();
    virtual void Run();
    virtual int Shutdown();

    virtual HRESULT MoveWindowContainer(std::unique_ptr<CRCContainer>& container);
};