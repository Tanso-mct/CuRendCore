#pragma once

#include "CuRendCore.h"

class ExampleWndCtrl : public CRC::WindowController
{
public:
    ExampleWndCtrl();
    ~ExampleWndCtrl() override;

    HRESULT OnCreate() override;
    HRESULT OnSetFocus() override;
    HRESULT OnKillFocus() override;
    HRESULT OnMinimize() override;
    HRESULT OnMaximize() override;
    HRESULT OnRestored() override;
    HRESULT OnPaint() override;
    HRESULT OnMove() override;
    HRESULT OnClose() override;
    HRESULT OnDestroy() override;
    HRESULT OnKeyDown() override;
    HRESULT OnKeyUp() override;
    HRESULT OnMouse() override;
};

