#pragma once

#include "CRC_config.h"
#include "CRC_interface.h"

class CRCCore;
class CRCWindowAttr;

namespace CRC
{

template <typename T, typename S>
std::unique_ptr<T> GetAs(std::unique_ptr<S> source)
{
    T* target = dynamic_cast<T*>(source.get());

    if (target)
    {
        return std::unique_ptr<T>(static_cast<T*>(source.release()));
    }
    else return nullptr;
}

template <typename T, typename S>
std::unique_ptr<T> CastMove(std::unique_ptr<S>& source)
{
    T* target = dynamic_cast<T*>(source.get());

    if (target)
    {
        return std::unique_ptr<T>(static_cast<T*>(source.release()));
    }
    else return nullptr;
}

template <typename T, typename S>
std::unique_ptr<T>& CastRef(std::unique_ptr<S>& source)
{
    T* target = dynamic_cast<T*>(source.get());

    if (target)
    {
        return *reinterpret_cast<std::unique_ptr<T>*>(&source);
    }
    else
    {
        static std::unique_ptr<T> emptyData = nullptr;
        return emptyData;
    }
}

std::unique_ptr<CRCCore> CRC_API CreateCRCCore();

std::unique_ptr<CRCContainer> CRC_API CreateWindowContainer();
std::unique_ptr<CRCData> CRC_API CreateWindowData(CRCWindowAttr& attr);

}