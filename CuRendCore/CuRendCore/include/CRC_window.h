#pragma once

#include "CRC_interface.h"

#include <Windows.h>
#include <memory>
#include <vector>

class CRCWindowAttr
{
public:
    int width;
    int height;
};

class CRCWindowData : public CRCData
{
public:
    virtual ~CRCWindowData() = default;
};

class CRCWindowContainer : public CRCContainer
{
private:
    std::vector<std::unique_ptr<CRCWindowData>> data_;

public:
    virtual ~CRCWindowContainer() = default;

    virtual int Add(std::unique_ptr<CRCData>& data) override;
    virtual HRESULT Remove(int id) override;

    virtual std::unique_ptr<CRCData>& Get(int id) override;
    virtual int GetSize() override;

    virtual void Clear() override;
};