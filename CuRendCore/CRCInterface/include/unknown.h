#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace CRC
{

class IUnknown
{
public:
    virtual ~IUnknown() = default;
    virtual HRESULT Release() = 0;
};

}