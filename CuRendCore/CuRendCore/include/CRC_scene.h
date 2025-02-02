#pragma once

#include "CRC_config.h"
#include "CRC_interface.h"

#include <Windows.h>
#include <memory>
#include <vector>
#include <string>

struct CRC_API CRCSceneAttr
{
    std::string name_ = "Scene";
};

class CRC_API CRCSceneData : public ICRCData
{
public:
    virtual ~CRCSceneData() override = default;

    std::unique_ptr<CRCSceneAttr> src_ = nullptr;
};

class CRC_API CRCSceneContainer : public ICRCContainer
{
private:
    std::vector<std::unique_ptr<CRCSceneData>> data_;

public:
    CRCSceneContainer() = default;
    virtual ~CRCSceneContainer() override = default;

    // Delete copy constructor and operator=.
    CRCSceneContainer(const CRCSceneContainer&) = delete;
    CRCSceneContainer& operator=(const CRCSceneContainer&) = delete;

    virtual int Add(std::unique_ptr<ICRCData> data) override;
    virtual HRESULT Remove(int id) override;

    virtual ICRCData* Get(int id) override;
    virtual int GetSize() override;

    virtual void Clear() override;
};