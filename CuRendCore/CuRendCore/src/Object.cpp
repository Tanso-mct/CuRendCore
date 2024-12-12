#include "Object.h"

namespace CRC
{

Object::Object(OBJECTATTR& oattr)
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    name = oattr.name;

    if (oattr.ctrl != nullptr) ctrl = std::move(oattr.ctrl);
    else
    {
        ObjectController* newCtrl = new ObjectController();
        ctrl = std::unique_ptr<ObjectController>(newCtrl);
    }

    slotRc = oattr.slotRc;
    slotBaseTex = oattr.slotBaseTex;
    slotNormalTex = oattr.slotNormalTex;
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

CRC_SLOT ObjectFactory::CreateObject(OBJECTATTR& oattr)
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