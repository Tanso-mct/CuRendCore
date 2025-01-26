#pragma once

#include "CRC_interface.h"

#include <Windows.h>
#include <memory>
#include <vector>

class CRCWindowAttr
{
public:
    WNDCLASSEX wcex_ = { 0 };
    LPCWSTR name_ = L"CuRendCore Window";
    int initialPosX_ = CW_USEDEFAULT;
    int initialPosY_ = CW_USEDEFAULT;
    unsigned int width_ = 800;
    unsigned int height_ = 600;
    DWORD style_ = WS_OVERLAPPEDWINDOW;
    HWND hWndParent_ = NULL;
    HINSTANCE hInstance = nullptr;
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