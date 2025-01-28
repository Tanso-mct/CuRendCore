#pragma once

#include "CRC_interface.h"

#include <Windows.h>
#include <memory>
#include <vector>
#include <string>

class CRCSceneData : public CRCData
{
public:
    virtual ~CRCSceneData() = default;
};

class CRCSceneAttr
{
public:
    std::string name_ = "Scene";
};

class CRCSceneContainer : public CRCContainer
{
private:
    std::vector<std::unique_ptr<CRCSceneData>> data_;

public:
    virtual ~CRCSceneContainer() = default;

    virtual int Add(std::unique_ptr<CRCData>& data) override;
    virtual HRESULT Remove(int id) override;

    virtual std::unique_ptr<CRCData>& Get(int id) override;
    virtual int GetSize() override;

    virtual void Clear() override;
};