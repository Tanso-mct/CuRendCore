#include "Utility.cuh"

namespace CRC
{

Utility::Utility(UTILITYATTR& utattr)
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    name = utattr.name;

    if (utattr.ctrl != nullptr) ctrl = std::move(utattr.ctrl);
    else
    {
        UtilityController* newCtrl = new UtilityController();
        ctrl = std::unique_ptr<UtilityController>(newCtrl);
    }

    slotRc = utattr.slotRc;
}

Utility::~Utility()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    if (ctrl != nullptr) ctrl.reset();
}

UtilityFactory::~UtilityFactory()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");
    
    for (auto& Utility : utilities)
    {
        Utility.reset();
    }
    utilities.clear();
}

CRC_SLOT UtilityFactory::CreateUtility(UTILITYATTR& utattr)
{
    std::shared_ptr<Utility> newUtility = std::shared_ptr<Utility>(new Utility(utattr));
    newUtility->ctrl->SetUtility(newUtility);
    utilities.push_back(newUtility);

    newUtility->thisSlot = ((CRC_SLOT)(utilities.size() - 1));
    return newUtility->thisSlot;
}

HRESULT UtilityFactory::DestroyUtility(CRC_SLOT slotUtility)
{
    if (slotUtility >= utilities.size()) return E_FAIL;
    if (utilities[slotUtility] == nullptr) return E_FAIL;

    utilities[slotUtility].reset();
    return S_OK;
}

} // namespace CRC