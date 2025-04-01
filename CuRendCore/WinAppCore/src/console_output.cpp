#include "pch.h"
#include "console_output.h"

WACore::IssueNotifier::IssueNotifier(const unsigned int &errorCount, const unsigned int &warningCount)
: errorCount_(errorCount),
    warningCount_(warningCount),
    colNormal_("\033[1;32m"),
    colReset_("\033[0m"){}

WACore::IssueNotifier::~IssueNotifier()
{
    Notify();
}

void WACore::IssueNotifier::Notify() const
{
    std::cout << std::endl;

    std::cout << colNormal_ << "[==========]" << colReset_;
    std::cout << " Executing results." << std::endl;

    if (errorCount_ == 0 && warningCount_ == 0)
    {
        std::cout << colNormal_ << "[==========]" << colReset_;
        std::cout << " No issues found." << std::endl;
        return;
    }

    if (errorCount_ > 0)
    {
        std::cout << colNormal_ << "[==========]" << colReset_;
        std::cout << " " << errorCount_ << " error(s) found." << std::endl;
    }

    if (warningCount_ > 0)
    {
        std::cout << colNormal_ << "[==========]" << colReset_;
        std::cout << " " << warningCount_ << " warning(s) found." << std::endl;
    }

    std::cout << std::endl;
}

WACore::ConsoleOuter::ConsoleOuter()
: startTag_("[==========]"),
    endTag_("[      END ]"),
    colReset_("\033[0m"),
    colNormal_("\033[1;32m"),
    colError_("\033[1;31m"),
    colWarning_("\033[1;33m"),
    colInfo_("\033[1;34m"),
    colDebug_("\033[1;35m"),
    errorCount_(0),
    warningCount_(0){}

WACore::ConsoleOuter::ConsoleOuter(unsigned int errorCount, unsigned int warningCount)
: startTag_("[==========]"),
    endTag_("[      END ]"),
    colReset_("\033[0m"),
    colNormal_("\033[1;32m"),
    colError_("\033[1;31m"),
    colWarning_("\033[1;33m"),
    colInfo_("\033[1;34m"),
    colDebug_("\033[1;35m"),
    errorCount_(errorCount),
    warningCount_(warningCount){}

const unsigned int &WACore::ConsoleOuter::GetErrorCount() const
{
    return errorCount_;
}

const unsigned int &WACore::ConsoleOuter::GetWarningCount() const
{
    return warningCount_;
}

void WACore::ConsoleOuter::Cout(std::initializer_list<std::string_view> args)
{
    std::cout << colNormal_ << startTag_ << colReset_ << " ";
    for (const auto& arg : args) std::cout << arg << std::endl;
    if (args.size() != 1) std::cout << colNormal_ << endTag_ << colReset_ << std::endl;
}

void WACore::ConsoleOuter::CoutErr(std::initializer_list<std::string_view> args)
{
    std::cout << colError_ << startTag_ << colReset_ << " ";
    for (const auto& arg : args) std::cout << arg << std::endl;
    if (args.size() != 1) std::cout << colError_ << endTag_ << colReset_ << std::endl;

    errorCount_++;
}

void WACore::ConsoleOuter::CoutWrn(std::initializer_list<std::string_view> args)
{
    std::cout << colWarning_ << startTag_ << colReset_ << " ";
    for (const auto& arg : args) std::cout << arg << std::endl;
    if (args.size() != 1) std::cout << colWarning_ << endTag_ << colReset_ << std::endl;

    warningCount_++;
}

void WACore::ConsoleOuter::CoutInfo(std::initializer_list<std::string_view> args)
{
    std::cout << colInfo_ << startTag_ << colReset_ << " ";
    for (const auto& arg : args) std::cout << arg << std::endl;
    if (args.size() != 1) std::cout << colInfo_ << endTag_ << colReset_ << std::endl;
}

void WACore::ConsoleOuter::CoutDebug(std::initializer_list<std::string_view> args)
{
    std::cout << colDebug_ << startTag_ << colReset_ << " ";
    for (const auto& arg : args) std::cout << arg << std::endl;
    if (args.size() != 1) std::cout << colDebug_ << endTag_ << colReset_ << std::endl;
}
