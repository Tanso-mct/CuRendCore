#include "UI.cuh"

#include "Image.cuh"
#include "Text.cuh"

namespace CRC
{

UI::UI(UI_ATTR& uiattr)
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    name = uiattr.name;

    if (uiattr.ctrl != nullptr) ctrl = std::move(uiattr.ctrl);
    else
    {
        UIController* newCtrl = new UIController();
        ctrl = std::unique_ptr<UIController>(newCtrl);
    }

    slotResource = uiattr.slotResource;

    pos = uiattr.pos;
    rot = uiattr.rot;
    scl = uiattr.scl;
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

CRC_SLOT UIFactory::CreateUI(UI_ATTR& uiattr)
{
    std::shared_ptr<UI> newUI;

    switch (uiattr.type)
    {
    case CRC_UI_TYPE_IMAGE:
        newUI = std::shared_ptr<UI>(new Image(uiattr));
        break;
    case CRC_UI_TYPE_TEXT:
        newUI = std::shared_ptr<UI>(new Text(uiattr));
        break;
    default:
        return CRC_SLOT_INVALID;
    }

    newUI->ctrl->SetUI(newUI);
    uis.push_back(newUI);

    newUI->thisSlot = ((CRC_SLOT)(uis.size() - 1));
    return newUI->thisSlot;
}

std::shared_ptr<UI> UIFactory::GetUI(CRC_SLOT slotUI)
{
    if (slotUI >= uis.size()) return std::shared_ptr<UI>();
    if (uis[slotUI] == nullptr) return std::shared_ptr<UI>();

    return uis[slotUI];
}

HRESULT UIFactory::DestroyUI(CRC_SLOT slotUI)
{
    if (slotUI >= uis.size()) return E_FAIL;
    if (uis[slotUI] == nullptr) return E_FAIL;

    uis[slotUI].reset();
    return S_OK;
}

} // namespace CRC