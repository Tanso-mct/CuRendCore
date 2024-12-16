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
    CRC_UI_TYPE type = CRC_UI_TYPE_NULL;
    CRC_SLOT slotResource = CRC_SLOT_INVALID;

    Vec3d pos = Vec3d(0.0, 0.0, 0.0);
    Vec3d rot = Vec3d(0.0, 0.0, 0.0);
    Vec3d scl = Vec3d(1.0, 1.0, 1.0);
} UI_ATTR;

class UI : public Component
{
private:
    UI(UI_ATTR& uiattr);

    // UI attributes.
    std::string name = "";
    std::unique_ptr<UIController> ctrl = nullptr;
    CRC_SLOT slotResource = CRC_SLOT_INVALID;

public:
    virtual ~UI();

    friend class UIController;
    friend class UIFactory;

    friend class Image;
    friend class Text;
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

    CRC_SLOT CreateUI(UI_ATTR& uiattr);
    std::shared_ptr<UI> GetUI(CRC_SLOT slotUI);
    HRESULT DestroyUI(CRC_SLOT slotUI);

    friend class Group;
};



} // namespace CRC