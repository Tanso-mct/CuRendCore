#include "Group.h"

namespace CRC
{

Group::Group(GROUPATTR gattr)
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    this->gattr = gattr;
}

Group::~Group()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");
}

GroupFactory::~GroupFactory()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    for (auto& group : groups)
    {
        group.reset();
    }
    groups.clear();
}

CRC_SLOT GroupFactory::CreateGroup(GROUPATTR gattr)
{
    std::shared_ptr<Group> newGroup = std::shared_ptr<Group>(new Group(gattr));
    newGroup->ctrl->SetGroup(newGroup);
    groups.push_back(newGroup);

    newGroup->thisSlot = (CRC_SLOT)(groups.size() - 1);
    return (CRC_SLOT)(groups.size() - 1);
}

HRESULT GroupFactory::DestroyGroup(CRC_SLOT slotGroup)
{
    if (slotGroup >= groups.size()) return E_FAIL;
    if (groups[slotGroup] == nullptr) return E_FAIL;

    groups[slotGroup].reset();
    return S_OK;
}

CRC_SLOT GroupFactory::CreateBinder(BINDERATTR battr)
{
    std::shared_ptr<Binder> newBinder = std::shared_ptr<Binder>(new Binder(battr));
    newBinder->ctrl->SetBinder(newBinder);
    binders.push_back(newBinder);

    newBinder->thisSlot = (CRC_SLOT)(binders.size() - 1);
    return (CRC_SLOT)(binders.size() - 1);
}

HRESULT GroupFactory::DestroyBinder(CRC_SLOT slotBinder)
{
    if (slotBinder >= binders.size()) return E_FAIL;
    if (binders[slotBinder] == nullptr) return E_FAIL;

    binders[slotBinder].reset();
    return S_OK;
}

} // namespace CRC