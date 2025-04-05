#include "CRCBuffer/include/pch.h"
#include "CRCBuffer/include/console.h"

CRC_BUFFER std::unique_ptr<WACore::ConsoleOuter> &CRCBuffer::GetConsoleOuter()
{
    static std::unique_ptr<WACore::ConsoleOuter> consoleOuter = std::make_unique<WACore::ConsoleOuter>();
    consoleOuter->startTag_ = "[CuRendCore/CRCBuffer]";
    return consoleOuter;
}

CRC_BUFFER void CRCBuffer::Cout(std::initializer_list<std::string_view> args)
{
    CRCBuffer::GetConsoleOuter()->Cout(args);
}

CRC_BUFFER void CRCBuffer::CoutErr(std::initializer_list<std::string_view> args)
{
    CRCBuffer::GetConsoleOuter()->CoutErr(args);
}

CRC_BUFFER void CRCBuffer::CoutWrn(std::initializer_list<std::string_view> args)
{
    CRCBuffer::GetConsoleOuter()->CoutWrn(args);
}

CRC_BUFFER void CRCBuffer::CoutInfo(std::initializer_list<std::string_view> args)
{
    CRCBuffer::GetConsoleOuter()->CoutInfo(args);
}

CRC_BUFFER void CRCBuffer::CoutDebug(std::initializer_list<std::string_view> args)
{
    CRCBuffer::GetConsoleOuter()->CoutDebug(args);
}
