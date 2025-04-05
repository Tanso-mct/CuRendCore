#pragma once

#include "CRCBuffer/include/config.h"

#include "WinAppCore/include/WACore.h"

namespace CRCBuffer
{

CRC_BUFFER std::unique_ptr<WACore::ConsoleOuter>& GetConsoleOuter();
CRC_BUFFER void Cout(std::initializer_list<std::string_view> args);
CRC_BUFFER void CoutErr(std::initializer_list<std::string_view> args);
CRC_BUFFER void CoutWrn(std::initializer_list<std::string_view> args);
CRC_BUFFER void CoutInfo(std::initializer_list<std::string_view> args);
CRC_BUFFER void CoutDebug(std::initializer_list<std::string_view> args);


} // namespace CRCBuffer
