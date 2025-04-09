#pragma once

#include "CRCTexture/include/config.h"

#include "WinAppCore/include/WACore.h"

namespace CRCTexture
{

CRC_TEXTURE std::unique_ptr<WACore::ConsoleOuter>& GetConsoleOuter();
CRC_TEXTURE void Cout(std::initializer_list<std::string_view> args);
CRC_TEXTURE void CoutErr(std::initializer_list<std::string_view> args);
CRC_TEXTURE void CoutWrn(std::initializer_list<std::string_view> args);
CRC_TEXTURE void CoutInfo(std::initializer_list<std::string_view> args);
CRC_TEXTURE void CoutDebug(std::initializer_list<std::string_view> args);


} // namespace CRCTexture
