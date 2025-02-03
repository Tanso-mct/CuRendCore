#pragma once

#include "CRC_config.h"
#include "CRC_interface.h"

#include <Windows.h>
#include <memory>
#include <vector>
#include <string>

struct CRC_API CRCSceneSrc
{
    std::string name_ = "Scene";
};

class CRC_API CRCSceneAttr : public ICRCContainable
{
public:
    virtual ~CRCSceneAttr() override = default;

    std::unique_ptr<CRCSceneSrc> src_ = nullptr;
};

class CRC_API CRCSceneContainer : public ICRCContainer
{
private:
    std::vector<std::unique_ptr<CRCSceneAttr>> data_;

public:
    CRCSceneContainer() = default;
    virtual ~CRCSceneContainer() override = default;

    // Delete copy constructor and operator=.
    CRCSceneContainer(const CRCSceneContainer&) = delete;
    CRCSceneContainer& operator=(const CRCSceneContainer&) = delete;

    virtual int Add(std::unique_ptr<ICRCContainable> data) override;
    virtual HRESULT Remove(int id) override;

    virtual ICRCContainable* Get(int id) override;
    virtual int GetSize() override;

    virtual void Clear() override;
};