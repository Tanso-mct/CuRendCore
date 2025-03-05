#pragma once

#include "CRC_config.h"

#include <memory>
#include <vector>
#include <Windows.h>
#include <stdexcept>

class CRC_API ICRCContainable
{
public:
    virtual ~ICRCContainable() = default;
};

class CRC_API ICRCContainer : public ICRCContainable
{
public:
    virtual ~ICRCContainer() = default;

    virtual int Add(std::unique_ptr<ICRCContainable> data) = 0;
    virtual HRESULT Remove(int id) = 0;

    virtual std::unique_ptr<ICRCContainable>& Get(int id) = 0;

    virtual std::unique_ptr<ICRCContainable> Take(int id) = 0;
    virtual HRESULT Put(int id, std::unique_ptr<ICRCContainable> data) = 0;

    virtual int GetSize() = 0;

    virtual void Clear() = 0;
};

class CRC_API CRCContainer : public ICRCContainer
{
private:
    std::vector<std::unique_ptr<ICRCContainable>> datas_;
    std::unique_ptr<ICRCContainable> emptyData_ = nullptr;

public:
    CRCContainer() = default;
    virtual ~CRCContainer() override = default;

    // Delete copy constructor and operator=.
    CRCContainer(const CRCContainer&) = delete;
    CRCContainer& operator=(const CRCContainer&) = delete;

    int Add(std::unique_ptr<ICRCContainable> data);
    HRESULT Remove(int id) override;

    std::unique_ptr<ICRCContainable>& Get(int id) override;

    std::unique_ptr<ICRCContainable> Take(int id) override;
    HRESULT Put(int id, std::unique_ptr<ICRCContainable> data) override;
    
    int GetSize();
    void Clear();
};