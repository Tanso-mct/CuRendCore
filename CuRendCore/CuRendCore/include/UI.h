#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <string>

#include "Component.h" 
#include "Math.cuh"

namespace CRC
{

class UIController;

typedef struct CRC_API _UI_ATTRIBUTES
{
    std::string name = "";
    std::unique_ptr<UIController> ctrl = nullptr;
    CRC_SLOT slotResource = CRC_SLOT_INVALID;
} UIATTR;

class UI : public Component
{
private:
    UI(UIATTR& uiattr);

    // UI attributes.
    std::string name = "";
    std::unique_ptr<UIController> ctrl = nullptr;
    CRC_SLOT slotResource = CRC_SLOT_INVALID;

public:
    ~UI();

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

    friend class UIFactory;
};

class CRC_API UIFactory
{
private:
    UIFactory() { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };
    std::vector<std::shared_ptr<UI>> uis;

public:
    ~UIFactory();

    CRC_SLOT CreateUI(UIATTR& uiattr);
    HRESULT DestroyUI(CRC_SLOT slotUI);

    friend class Group;
};



} // namespace CRC