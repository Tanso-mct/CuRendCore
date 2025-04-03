#pragma once

#include "CudaCore/include/config.h"
#include "WinAppCore/include/WACore.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CudaCore
{

CUDA_CORE std::unique_ptr<WACore::ConsoleOuter>& GetConsoleOuter();
CUDA_CORE void Cout(std::initializer_list<std::string_view> args);
CUDA_CORE void CoutErr(std::initializer_list<std::string_view> args);
CUDA_CORE void CoutWrn(std::initializer_list<std::string_view> args);
CUDA_CORE void CoutInfo(std::initializer_list<std::string_view> args);
CUDA_CORE void CoutDebug(std::initializer_list<std::string_view> args);


} // namespace CudaCore
