#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <string>

#include "Component.h" 

namespace CRC
{

class UtilityController;

typedef struct CRC_API _UTILITY_ATTRIBUTES
{
    std::string name = "";
    std::unique_ptr<UtilityController> ctrl = nullptr;
    CRC_SLOT slotRc = CRC_SLOT_INVALID;
} UTILITYATTR;

class Utility : public Component
{
private:
    Utility(UTILITYATTR& utattr);

    // Utility attributes.
    std::string name = "";
    std::unique_ptr<UtilityController> ctrl = nullptr;
    CRC_SLOT slotRc = CRC_SLOT_INVALID;

public:
    ~Utility();

    friend class UtilityController;
    friend class UtilityFactory;
};

class CRC_API UtilityController
{
private:
    std::weak_ptr<Utility> utility;

    void SetUtility(std::weak_ptr<Utility> utility) { this->utility = utility; }
    std::shared_ptr<Utility> GetUtility() { return utility.lock(); }

public:
    UtilityController() { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };
    virtual ~UtilityController() { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };

    friend class UtilityFactory;
};

class CRC_API UtilityFactory
{
private:
    UtilityFactory() { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };
    std::vector<std::shared_ptr<Utility>> utilities;

public:
    ~UtilityFactory();

    CRC_SLOT CreateUtility(UTILITYATTR& utattr);
    HRESULT DestroyUtility(CRC_SLOT slotUtility);

    friend class Group;
};



} // namespace CRC