#include "CRCDevice/include/console.h"

CRC_DEVICE std::unique_ptr<WACore::ConsoleOuter> &CRCDevice::GetConsoleOuter()
{
    static std::unique_ptr<WACore::ConsoleOuter> consoleOuter = std::make_unique<WACore::ConsoleOuter>();
    consoleOuter->startTag_ = "[CuRendCore/CRCDevice]";
    return consoleOuter;
}

CRC_DEVICE void CRCDevice::Cout(std::initializer_list<std::string_view> args)
{
    CRCDevice::GetConsoleOuter()->Cout(args);
}

CRC_DEVICE void CRCDevice::CoutErr(std::initializer_list<std::string_view> args)
{
    CRCDevice::GetConsoleOuter()->CoutErr(args);
}

CRC_DEVICE void CRCDevice::CoutWrn(std::initializer_list<std::string_view> args)
{
    CRCDevice::GetConsoleOuter()->CoutWrn(args);
}

CRC_DEVICE void CRCDevice::CoutInfo(std::initializer_list<std::string_view> args)
{
    CRCDevice::GetConsoleOuter()->CoutInfo(args);
}

CRC_DEVICE void CRCDevice::CoutDebug(std::initializer_list<std::string_view> args)
{
    CRCDevice::GetConsoleOuter()->CoutDebug(args);
}
