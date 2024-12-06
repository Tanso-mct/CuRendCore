#include "Object.h"

namespace CRC
{

Object::Object(OBJECTATTR oattr)
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    this->oattr = oattr;

    if (oattr.ctrl != nullptr) ctrl = oattr.ctrl;
    else ctrl = std::make_shared<ObjectController>();
}

Object::~Object()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    if (ctrl != nullptr) ctrl.reset();
}

ObjectFactory::~ObjectFactory()
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");
    
    for (auto& object : objects)
    {
        object.reset();
    }
    objects.clear();
}

CRC_SLOT ObjectFactory::CreateObject(OBJECTATTR oattr)
{
    std::shared_ptr<Object> newObject = std::shared_ptr<Object>(new Object(oattr));
    newObject->ctrl->SetObject(newObject);
    objects.push_back(newObject);

    newObject->thisSlot = ((CRC_SLOT)(objects.size() - 1));
    return newObject->thisSlot;
}

HRESULT ObjectFactory::DestroyObject(CRC_SLOT slotObject)
{
    if (slotObject >= objects.size()) return E_FAIL;
    if (objects[slotObject] == nullptr) return E_FAIL;

    objects[slotObject].reset();
    return S_OK;
}

} // namespace CRC