#pragma once

#include "CRC_interface.h"

#include <Windows.h>
#include <memory>
#include <vector>
#include <string>
#include <mutex>

class CRCSceneData : public CRCData
{
public:
    virtual ~CRCSceneData() = default;

    // Used when creating a window. After creation, nullPtr.
    std::unique_ptr<CRCSceneAttr> src_ = nullptr;
};

struct CRCSceneAttr
{
    std::string name_ = "Scene";

    bool needCreateFlag_ = true;
};

class CRCSceneContainer : public CRCContainer
{
private:
    std::vector<std::unique_ptr<CRCSceneData>> data_;

public:
    virtual ~CRCSceneContainer() = default;

    virtual int Add(std::unique_ptr<CRCData>& data) override;

    virtual std::unique_ptr<CRCData> Take(int id) override;
    virtual HRESULT Set(int id, std::unique_ptr<CRCData>& data) override;

    virtual UINT GetSize() override;

    virtual void Clear(int id) override;
    virtual void Clear() override;
};