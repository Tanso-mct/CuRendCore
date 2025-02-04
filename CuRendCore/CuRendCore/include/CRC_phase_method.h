#pragma once

#include "CRC_config.h"
#include "CRC_interface.h"

#include <memory>
#include <vector>

class CRC_API CRCPMContainer : public ICRCContainer
{
private:
    std::vector<std::unique_ptr<ICRCPhaseMethod>> methods_;

public:
    CRCPMContainer() = default;
    virtual ~CRCPMContainer() override = default;

    // Delete copy constructor and operator=.
    CRCPMContainer(const CRCPMContainer&) = delete;
    CRCPMContainer& operator=(const CRCPMContainer&) = delete;

    int Add(std::unique_ptr<ICRCContainable> data);
    HRESULT Remove(int id) override;

    ICRCContainable* Get(int id);
    int GetSize();

    void Clear();
};