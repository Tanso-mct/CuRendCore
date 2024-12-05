#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <string>

namespace CRC
{

class UIController;

typedef struct CRC_API _UI_ATTRIBUTES
{
    std::string name = "";
    std::shared_ptr<UIController> ctrl = nullptr;
} UIATTR;

class UI
{
private:
    std::shared_ptr<UIController> ctrl = nullptr;

    UI(UIATTR uiattr);

    CRC_SLOT thisSlot = CRC_SLOT_INVALID;
    UIATTR uiattr;

public:
    ~UI();

    CRC_SLOT GetSlot() { return thisSlot; }

    friend class UIController;
    friend class UIFactory;
};

class CRC_API UIController
{
private:
    std::weak_ptr<UI> ui;

    void SetUI(std::weak_ptr<UI> ui) { this->ui = ui; }
    std::shared_ptr<UI> GetUI() { return ui.lock(); }

public:
    UIController() { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };
    virtual ~UIController() { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };
};

class CRC_API UIFactory
{
private:
    UIFactory() { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };
    std::vector<std::shared_ptr<UI>> uis;

public:
    ~UIFactory();

    CRC_SLOT CreateUI(UIATTR uiattr);
    HRESULT DestroyUI(CRC_SLOT slotUI);
};



} // namespace CRC