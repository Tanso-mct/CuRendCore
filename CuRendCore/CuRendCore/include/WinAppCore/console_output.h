#pragma once
#include "config.h"

#include <iostream>
#include <initializer_list>
#include <string>
#include <string_view>

namespace WACore
{

class WIN_APP_CORE IIssueNotifier
{
public:
    virtual ~IIssueNotifier() = default;
    virtual void Notify() const = 0;
};

class WIN_APP_CORE IConsoleOuter
{
public:
    virtual ~IConsoleOuter() = default;

    virtual const unsigned int& GetErrorCount() const = 0;
    virtual const unsigned int& GetWarningCount() const = 0;

    virtual void Cout(std::initializer_list<std::string_view> args) = 0;
    virtual void CoutErr(std::initializer_list<std::string_view> args) = 0;
    virtual void CoutWrn(std::initializer_list<std::string_view> args) = 0;
    virtual void CoutInfo(std::initializer_list<std::string_view> args) = 0;
    virtual void CoutDebug(std::initializer_list<std::string_view> args) = 0;
};

class WIN_APP_CORE IssueNotifier : public IIssueNotifier
{
private:
    const unsigned int& errorCount_;
    const unsigned int& warningCount_;

    std::string colNormal_;
    std::string colReset_;

public:
    IssueNotifier() = delete;

    IssueNotifier(const unsigned int& errorCount, const unsigned int& warningCount);
    ~IssueNotifier() override;

    void Notify() const override;
};

class WIN_APP_CORE ConsoleOuter : public IConsoleOuter
{
private:
    unsigned int errorCount_ = 0;
    unsigned int warningCount_ = 0;
    
public:
    ConsoleOuter();
    ConsoleOuter(unsigned int errorCount, unsigned int warningCount);
    ~ConsoleOuter() override = default;

    std::string startTag_;
    std::string endTag_;

    std::string colReset_;
    std::string colNormal_;
    std::string colError_;
    std::string colWarning_;
    std::string colInfo_;
    std::string colDebug_;

    const unsigned int& GetErrorCount() const override;
    const unsigned int& GetWarningCount() const override;

    void Cout(std::initializer_list<std::string_view> args) override;
    void CoutErr(std::initializer_list<std::string_view> args) override;
    void CoutWrn(std::initializer_list<std::string_view> args) override;
    void CoutInfo(std::initializer_list<std::string_view> args) override;
    void CoutDebug(std::initializer_list<std::string_view> args) override;
};

} // namespace WACore