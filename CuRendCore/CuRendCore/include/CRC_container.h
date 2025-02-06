#pragma once

#include "CRC_config.h"
#include "CRC_interface.h"

#include <memory>
#include <vector>

class CRC_API CRCContainer : public ICRCContainer
{
private:
    std::vector<std::unique_ptr<ICRCContainable>> datas_;

public:
    CRCContainer() = default;
    virtual ~CRCContainer() override = default;

    // Delete copy constructor and operator=.
    CRCContainer(const CRCContainer&) = delete;
    CRCContainer& operator=(const CRCContainer&) = delete;

    int Add(std::unique_ptr<ICRCContainable> data);
    HRESULT Remove(int id) override;

    ICRCContainable* Get(int id);
    int GetSize();

    void Clear();
};