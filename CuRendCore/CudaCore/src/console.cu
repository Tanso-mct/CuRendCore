#include "CudaCore/include/pch.h"
#include "CudaCore/include/console.cuh"

CUDA_CORE std::unique_ptr<WACore::ConsoleOuter> &CudaCore::GetConsoleOuter()
{
    static std::unique_ptr<WACore::ConsoleOuter> consoleOuter = std::make_unique<WACore::ConsoleOuter>();
    consoleOuter->startTag_ = "[ CudaCore ]";
    return consoleOuter;
}

CUDA_CORE void CudaCore::Cout(std::initializer_list<std::string_view> args)
{
    CudaCore::GetConsoleOuter()->Cout(args);
}

CUDA_CORE void CudaCore::CoutErr(std::initializer_list<std::string_view> args)
{
    CudaCore::GetConsoleOuter()->CoutErr(args);
}

CUDA_CORE void CudaCore::CoutWrn(std::initializer_list<std::string_view> args)
{
    CudaCore::GetConsoleOuter()->CoutWrn(args);
}

CUDA_CORE void CudaCore::CoutInfo(std::initializer_list<std::string_view> args)
{
    CudaCore::GetConsoleOuter()->CoutInfo(args);
}

CUDA_CORE void CudaCore::CoutDebug(std::initializer_list<std::string_view> args)
{
    CudaCore::GetConsoleOuter()->CoutDebug(args);
}
