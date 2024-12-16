#include "Component.h"

namespace CRC
{

void Component::SetWorldMatrix()
{
    // ddata->mtWorld.Identity();
    // ddata->mtWorld *= MatrixScaling(scl);
    // ddata->mtWorld *= MatrixRotationX(rot.x);
    // ddata->mtWorld *= MatrixRotationY(rot.y);
    // ddata->mtWorld *= MatrixRotationZ(rot.z);
    // ddata->mtWorld *= MatrixTranslation(pos);
}

Component::~Component()
{
    if (!parent.expired())
    {
        if (auto sharedParent = parent.lock()) sharedParent->RemoveChild(std::make_pair(thisType, thisSlot));
    }

    for (auto child : children)
    {
        if (auto sharedChild = child.lock()) sharedChild->ClearParent();
    }
}

HRESULT Component::SetParent(std::weak_ptr<Component> parent)
{
    if (parent.expired())
    {
        CRCErrorOutput(__FILE__, __FUNCTION__, __LINE__, "This argument parent is expired.");
        return E_FAIL;
    }

    std::shared_ptr<Component> parentPtr = parent.lock();

    if (this->parent.expired())
    {
        this->parent = parent;
        parentPtr->AddChild(shared_from_this());
        return S_OK;
    }

    parentPtr->RemoveChild(std::make_pair(thisType, thisSlot));
    this->parent = parent;

    return S_OK;
}

HRESULT Component::AddChild(std::weak_ptr<Component> child)
{
    if (child.expired())
    {
        CRCErrorOutput(__FILE__, __FUNCTION__, __LINE__, "This argument child is expired.");
        return E_FAIL;
    }
    
    std::shared_ptr<Component> childPtr = child.lock();

    if (childrenSlot.find(std::make_pair(childPtr->thisType, childPtr->thisSlot)) == childrenSlot.end())
    {
        childrenSlot[std::make_pair(childPtr->thisType, childPtr->thisSlot)] = children.size();
        children.push_back(child);
        
        childPtr->SetParent(shared_from_this());
        return S_OK;
    }

    return S_OK;
}

HRESULT Component::RemoveChild(std::pair<CRC_COMPONENT_TYPE, CRC_SLOT> slot)
{
    if (childrenSlot.find(slot) != childrenSlot.end())
    {
        children[childrenSlot[slot]].lock()->ClearParent();
        children[childrenSlot[slot]].reset();
        return S_OK;
    }

    CRCErrorOutput(__FILE__, __FUNCTION__, __LINE__, "This argument slot is not found.");
    return E_FAIL;
}

std::weak_ptr<Component> Component::GetChild(std::pair<CRC_COMPONENT_TYPE, CRC_SLOT> slot)
{
    if (childrenSlot.find(slot) != childrenSlot.end())
    {
        return children[childrenSlot[slot]];
    }

    CRCErrorOutput(__FILE__, __FUNCTION__, __LINE__, "This argument slot is not found.");
    return std::weak_ptr<Component>();
}

void Component::Transfer(Vec3d pos, Vec3d &val)
{
    this->pos = pos;
    val = this->pos;
}

void Component::AddTransfer(Vec3d pos, Vec3d &val)
{
    this->pos += pos;
    val = this->pos;
}

void Component::Rotate(Vec3d rot, Vec3d &val)
{
    this->rot = rot;
    val = this->rot;
}

void Component::AddRotate(Vec3d rot, Vec3d &val)
{
    this->rot += rot;
    val = this->rot;
}

void Component::Scale(Vec3d scl, Vec3d &val)
{
    this->scl = scl;
    val = this->scl;
}

void Component::AddScale(Vec3d scl, Vec3d &val)
{
    this->scl += scl;
    val = this->scl;
}

} // namespace CRC
