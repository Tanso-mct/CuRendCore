#pragma once

#include "CRC_config.h"

#include <memory>
#include <Windows.h>
#include <utility>
#include <iostream>

#include <d3d11.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CRCCore;
class CRC_WINDOW_DESC; 
struct CRC_SCENE_DESC;

class ICRCContainable;
class ICRCContainer;

class ICRCWinMsgEvent;

namespace CRC
{

template <typename T, typename S>
T* As(S* source)
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

CRC_API HRESULT ShowWindowCRC(HWND& hWnd);
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

CRC_API void CheckCuda(cudaError_t call);

}