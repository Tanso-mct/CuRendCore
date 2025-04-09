#include "CRCTexture/include/pch.h"
#include "CRCTexture/include/console.h"

CRC_TEXTURE std::unique_ptr<WACore::ConsoleOuter> &CRCTexture::GetConsoleOuter()
{
    static std::unique_ptr<WACore::ConsoleOuter> consoleOuter = std::make_unique<WACore::ConsoleOuter>();
    consoleOuter->startTag_ = "[CuRendCore/CRCTexture]";
    return consoleOuter;
}

CRC_TEXTURE void CRCTexture::Cout(std::initializer_list<std::string_view> args)
{
    CRCTexture::GetConsoleOuter()->Cout(args);
}

CRC_TEXTURE void CRCTexture::CoutErr(std::initializer_list<std::string_view> args)
{
    CRCTexture::GetConsoleOuter()->CoutErr(args);
}

CRC_TEXTURE void CRCTexture::CoutWrn(std::initializer_list<std::string_view> args)
{
    CRCTexture::GetConsoleOuter()->CoutWrn(args);
}

CRC_TEXTURE void CRCTexture::CoutInfo(std::initializer_list<std::string_view> args)
{
    CRCTexture::GetConsoleOuter()->CoutInfo(args);
}

CRC_TEXTURE void CRCTexture::CoutDebug(std::initializer_list<std::string_view> args)
{
    CRCTexture::GetConsoleOuter()->CoutDebug(args);
}
