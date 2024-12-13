#include "Object.cuh"

#include "FromObj.cuh"

namespace CRC
{

Object::Object(OBJECT_ATTR& oattr)
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

    pos = oattr.pos;
    rot = oattr.rot;
    scl = oattr.scl;

    // Device attributes.
    cudaHostAlloc((void**)&dattr, sizeof(OBJECT_DATTR), cudaHostAllocMapped);

    // Initialize the device attributes.
    SetWorldMatrix();
}

void Object::SetWorldMatrix()
{
    dattr->mtWorld.Identity();
    dattr->mtWorld *= MatrixScaling(scl);
    dattr->mtWorld *= MatrixRotationX(rot.x);
    dattr->mtWorld *= MatrixRotationY(rot.y);
    dattr->mtWorld *= MatrixRotationZ(rot.z);
    dattr->mtWorld *= MatrixTranslation(pos);
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

CRC_SLOT ObjectFactory::CreateObject(OBJECT_ATTR& oattr)
{
    std::shared_ptr<Object> newObject;

    switch (oattr.from)
    {
    case CRC_OBJECT_FROM_OBJ:
        newObject = std::shared_ptr<Object>(new FromObj(oattr));
        break;
    }

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

void ObjectController::Transfer(Vec3d pos, Vec3d &val)
{
    GetObject()->pos = pos;
    val = GetObject()->pos;
}

void ObjectController::AddTransfer(Vec3d pos, Vec3d &val)
{
    GetObject()->pos += pos;
    val = GetObject()->pos;
}

void ObjectController::Rotate(Vec3d rot, Vec3d &val)
{
    GetObject()->rot = rot;
    val = GetObject()->rot;
}

void ObjectController::AddRotate(Vec3d rot, Vec3d &val)
{
    GetObject()->rot += rot;
    val = GetObject()->rot;
}

void ObjectController::Scale(Vec3d scl, Vec3d &val)
{
    GetObject()->scl = scl;
    val = GetObject()->scl;
}

void ObjectController::AddScale(Vec3d scl, Vec3d &val)
{
    GetObject()->scl += scl;
    val = GetObject()->scl;
}

} // namespace CRC