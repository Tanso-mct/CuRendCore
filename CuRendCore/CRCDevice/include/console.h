#pragma once

#include "CRCDevice/include/config.h"

#include "WinAppCore/include/WACore.h"

namespace CRCDevice
{

CRC_DEVICE std::unique_ptr<WACore::ConsoleOuter>& GetConsoleOuter();
CRC_DEVICE void Cout(std::initializer_list<std::string_view> args);
CRC_DEVICE void CoutErr(std::initializer_list<std::string_view> args);
CRC_DEVICE void CoutWrn(std::initializer_list<std::string_view> args);
CRC_DEVICE void CoutInfo(std::initializer_list<std::string_view> args);
CRC_DEVICE void CoutDebug(std::initializer_list<std::string_view> args);


} // namespace CRCDevice
