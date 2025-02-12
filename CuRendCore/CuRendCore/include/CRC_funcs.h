#pragma once

#include "CRC_config.h"

#include <memory>
#include <Windows.h>

class CRCCore;
struct CRCWindowSrc; 
struct CRCSceneSrc;
class ICRCContainable;

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

    if (target)
    {
        return std::unique_ptr<T>(static_cast<T*>(source.release()));
    }
    else return nullptr;
}

CRC_API std::unique_ptr<CRCCore>& Core();

CRC_API std::unique_ptr<ICRCContainable> CreateWindowAttr(std::unique_ptr<CRCWindowSrc> attr);
CRC_API std::unique_ptr<ICRCContainable> CreateSceneAttr(std::unique_ptr<CRCSceneSrc> attr);

CRC_API HRESULT CreateWindowCRC(std::unique_ptr<ICRCContainable>& windowAttr);
CRC_API HRESULT ShowWindowCRC(std::unique_ptr<ICRCContainable>& windowAttr);

CRC_API HRESULT CreateScene(std::unique_ptr<ICRCContainable>& sceneAttr);

}