#pragma once
#include "config.h"

#include <memory>
#include <vector>
#include <Windows.h>

namespace WACore
{

class WIN_APP_CORE IContainable
{
public:
    virtual ~IContainable() = default;
};

class WIN_APP_CORE IContainer : public IContainable
{
public:
    virtual ~IContainer() = default;

    virtual int Add(std::unique_ptr<IContainable> data) = 0;
    virtual HRESULT Del(int id) = 0;

    virtual std::unique_ptr<IContainable>& Get(int id) = 0;

    virtual std::unique_ptr<IContainable> Take(int id) = 0;
    virtual HRESULT Put(int id, std::unique_ptr<IContainable> data) = 0;

    virtual int GetSize() = 0;
    virtual void Clear() = 0;
};

class WIN_APP_CORE Container : public IContainer
{
private:
    std::vector<std::unique_ptr<IContainable>> datas_;
    std::unique_ptr<IContainable> emptyData_ = nullptr;

public:
    Container() = default;
    virtual ~Container() override = default;

    // Delete copy constructor and operator=.
    Container(const Container&) = delete;
    Container& operator=(const Container&) = delete;

    int Add(std::unique_ptr<IContainable> data);
    HRESULT Del(int id) override;

    std::unique_ptr<IContainable>& Get(int id) override;

    std::unique_ptr<IContainable> Take(int id) override;
    HRESULT Put(int id, std::unique_ptr<IContainable> data) override;
    
    int GetSize() override;
    void Clear() override;
};

} // namespace WACore