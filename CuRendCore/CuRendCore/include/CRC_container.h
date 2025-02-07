#pragma once

#include "CRC_config.h"
#include "CRC_interface.h"

#include <memory>
#include <vector>

class CRC_API CRCContainer : public ICRCContainer
{
private:
    std::vector<std::unique_ptr<ICRCContainable>> datas_;
    std::unique_ptr<ICRCContainable> nullData_ = nullptr;

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