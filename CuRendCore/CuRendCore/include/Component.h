#pragma once

#include "CRCConfig.h"

#include <unordered_map>
#include <memory>

#include "Math.cuh"

namespace CRC
{

class Component : public std::enable_shared_from_this<Component>
{
private:
    Component(){CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};

    std::weak_ptr<Component> parent;
    std::unordered_map<std::pair<CRC_COMPONENT_TYPE, CRC_SLOT>, CRC_SLOT, CRC_PAIR_HASH> childrenSlot;
    std::vector<std::weak_ptr<Component>> children;

protected:
    CRC_COMPONENT_TYPE thisType = CRC_COMPONENT_TYPE_NULL;
    CRC_SLOT thisSlot = CRC_SLOT_INVALID;

    Vec3d pos;
    Vec3d rot;
    Vec3d scl;

    void SetWorldMatrix();

public:
    virtual ~Component();

    template<typename T>
    T* As() {
        return dynamic_cast<T*>(this);
    }

    HRESULT SetParent(std::weak_ptr<Component> parent);
    std::weak_ptr<Component> GetParent() {return parent;}
    void ClearParent() {parent.reset();}

    HRESULT AddChild(std::weak_ptr<Component> child);
    HRESULT RemoveChild(std::pair<CRC_COMPONENT_TYPE, CRC_SLOT> slot);
    std::weak_ptr<Component> GetChild(std::pair<CRC_COMPONENT_TYPE, CRC_SLOT> slot);
    std::vector<std::weak_ptr<Component>> GetChildren() {return children;}

    Vec3d GetPos() { return pos; }
    void Transfer(Vec3d pos) { this->pos = pos; };
    void Transfer(Vec3d pos, Vec3d& val);
    void AddTransfer(Vec3d pos) { this->pos += pos; };
    void AddTransfer(Vec3d pos, Vec3d& val);

    Vec3d GetRot() { return rot; }
    void Rotate(Vec3d rot) { this->rot = rot; };
    void Rotate(Vec3d rot, Vec3d& val);
    void AddRotate(Vec3d rot) { this->rot += rot; };
    void AddRotate(Vec3d rot, Vec3d& val);

    Vec3d GetScl() { return scl; }
    void Scale(Vec3d scl) { this->scl = scl; };
    void Scale(Vec3d scl, Vec3d& val);
    void AddScale(Vec3d scl) { this->scl += scl; };
    void AddScale(Vec3d scl, Vec3d& val);

    friend class Object;
    friend class Utility;
    friend class UI;
};

    
} // namespace CRC
