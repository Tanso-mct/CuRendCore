﻿#pragma once

#include "CRC_config.h"

#include <memory>
#include <Windows.h>
#include <utility>
#include <iostream>
#include <string_view>
#include <initializer_list>

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
CRC_API HRESULT CreateDeviceAndSwapChain
(
    const HWND& hWnd,
    Microsoft::WRL::ComPtr<ID3D11Device>& device,
    Microsoft::WRL::ComPtr<IDXGISwapChain>& swapChain
);

UINT GetBytesPerPixel(const DXGI_FORMAT& format);

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

template <typename... Args>
CRC_API void Cout(Args&... args)
{
    std::cout << CRC::C_COLOR_MSG << CRC::C_TAG << CRC::C_COLOR_RESET << " ";
    (void)std::initializer_list<int>{(std::cout << args << " ", 0)...};
    std::cout << std::endl << CRC::C_COLOR_MSG << CRC::C_TAG_END << CRC::C_COLOR_RESET << std::endl;
}

template <typename... Args>
CRC_API void CoutError(Args&... args)
{
    std::cout << CRC::C_COLOR_ERROR << CRC::C_TAG << CRC::C_COLOR_RESET << " ";
    (void)std::initializer_list<int>{(std::cout << args << " ", 0)...};
    std::cout << std::endl << CRC::C_COLOR_ERROR << CRC::C_TAG_END << CRC::C_COLOR_RESET << std::endl;
}

template <typename... Args>
CRC_API void CoutWarning(Args&... args)
{
    std::cout << CRC::C_COLOR_WARNING << CRC::C_TAG << CRC::C_COLOR_RESET << " ";
    (void)std::initializer_list<int>{(std::cout << args << " ", 0)...};
    std::cout << std::endl << CRC::C_COLOR_WARNING << CRC::C_TAG_END << CRC::C_COLOR_RESET << std::endl;
}

}