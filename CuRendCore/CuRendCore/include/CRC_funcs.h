#pragma once

#include "CRC_config.h"
#include "CRC_interface.h"

#include <memory>

class CRCCore;
class CRCContainer;

struct CRCWindowAttr; 
struct CRCSceneAttr;

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

CRC_API std::unique_ptr<CRCCore>& Core();

CRC_API std::unique_ptr<CRCData> CreateWindowData(std::unique_ptr<CRCWindowAttr>& attr);
CRC_API void CreateWindowsAsync();

CRC_API std::unique_ptr<CRCData> CreateSceneData(std::unique_ptr<CRCSceneAttr>& attr);
CRC_API void CreateScenesAsync();



}