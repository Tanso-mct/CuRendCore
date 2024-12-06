#pragma once

#include "CRCConfig.h"

#include <unordered_map>
#include <memory>

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

public:
    virtual ~Component();

    HRESULT SetParent(std::weak_ptr<Component> parent);
    std::weak_ptr<Component> GetParent() {return parent;}
    void ClearParent() {parent.reset();}

    HRESULT AddChild(std::weak_ptr<Component> child);
    HRESULT RemoveChild(std::pair<CRC_COMPONENT_TYPE, CRC_SLOT> slot);
    std::weak_ptr<Component> GetChild(std::pair<CRC_COMPONENT_TYPE, CRC_SLOT> slot);
    std::vector<std::weak_ptr<Component>> GetChildren() {return children;}

    friend class Object;
    friend class Utility;
    friend class UI;
};

    
} // namespace CRC
