#pragma once

#include "CRC_config.h"

#include <memory>
#include <Windows.h>
#include <utility>

class CRCCore;
struct CRCWindowSrc; 
struct CRCSceneSrc;
class ICRCContainable;

class ICRCWinMsgListener;

namespace CRC
{

template <typename T, typename S>
T* PtrAs(S* source)
{
    T* target = dynamic_cast<T*>(source);
    return target;
}

template <typename T, typename S>
std::unique_ptr<T> UniqueAs(std::unique_ptr<S>& source)
{
    T* target = dynamic_cast<T*>(source.get());

    if (target) return std::unique_ptr<T>(static_cast<T*>(source.release()));
    else return nullptr;
}

CRC_API std::unique_ptr<CRCCore>& Core();

CRC_API std::unique_ptr<ICRCContainable> CreateWindowAttr(std::unique_ptr<CRCWindowSrc> attr);
CRC_API std::unique_ptr<ICRCContainable> CreateSceneAttr(std::unique_ptr<CRCSceneSrc> attr);

CRC_API HRESULT CreateWindowCRC(std::unique_ptr<ICRCContainable>& windowAttr);
CRC_API HRESULT ShowWindowCRC(std::unique_ptr<ICRCContainable>& windowAttr);

CRC_API HRESULT CreateScene(std::unique_ptr<ICRCContainable>& sceneAttr);

CRC_API HRESULT CreateSwapChain(std::unique_ptr<ICRCContainable>& windowAttr);

struct PairHash 
{
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const 
    {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

struct PairEqual 
{
    template <typename T1, typename T2>
    bool operator()(const std::pair<T1, T2>& lhs, const std::pair<T1, T2>& rhs) const 
    {
        return lhs == rhs;
    }
};

}