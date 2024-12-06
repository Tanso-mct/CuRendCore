#include "UI.h"

namespace CRC
{

UI::UI(UIATTR uiattr)
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    this->uiattr = uiattr;

    if (uiattr.ctrl != nullptr) ctrl = uiattr.ctrl;
    else ctrl = std::make_shared<UIController>();
}

UI::~UI()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    if (ctrl != nullptr) ctrl.reset();
}

UIFactory::~UIFactory()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");
    
    for (auto& ui : uis)
    {
        ui.reset();
    }
    uis.clear();
}

CRC_SLOT UIFactory::CreateUI(UIATTR uiattr)
{
    std::shared_ptr<UI> newUI = std::shared_ptr<UI>(new UI(uiattr));
    newUI->ctrl->SetUI(newUI);
    uis.push_back(newUI);

    newUI->thisSlot = ((CRC_SLOT)(uis.size() - 1));
    return newUI->thisSlot;
}

HRESULT UIFactory::DestroyUI(CRC_SLOT slotUI)
{
    if (slotUI >= uis.size()) return E_FAIL;
    if (uis[slotUI] == nullptr) return E_FAIL;

    uis[slotUI].reset();
    return S_OK;
}

} // namespace CRC