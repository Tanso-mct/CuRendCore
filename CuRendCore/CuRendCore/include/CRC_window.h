#pragma once

#include "CRC_interface.h"

class CRCWindowData : public CRCData
{
public:
    virtual ~CRCWindowData() = default;
};

class CRCWindowContainer : public CRCContainer
{
public:
    virtual ~CRCWindowContainer() = default;

    virtual int Add(std::unique_ptr<CRCData> data) override;
    virtual void Remove(int id) override;

    virtual std::unique_ptr<CRCData>& Get(int id) override;
    virtual UINT Get() override;

    virtual void Clear(int id) override;
    virtual void Clear() override;
};