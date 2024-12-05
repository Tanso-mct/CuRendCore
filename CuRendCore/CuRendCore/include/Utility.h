#pragma once

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <string>

namespace CRC
{

class UtilityController;

typedef struct CRC_API _UTILITY_ATTRIBUTES
{
    std::string name = "";
    std::shared_ptr<UtilityController> ctrl = nullptr;
} UTILITYATTR;

class Utility
{
private:
    std::shared_ptr<UtilityController> ctrl = nullptr;

    Utility(UTILITYATTR utattr);

    CRC_SLOT thisSlot = CRC_SLOT_INVALID;
    UTILITYATTR utattr;

public:
    ~Utility();

    CRC_SLOT GetSlot() { return thisSlot; }

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
};

class CRC_API UtilityFactory
{
private:
    UtilityFactory() { CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, ""); };
    std::vector<std::shared_ptr<Utility>> utilities;

public:
    ~UtilityFactory();

    CRC_SLOT CreateUtility(UTILITYATTR utattr);
    HRESULT DestroyUtility(CRC_SLOT slotUtility);
};



} // namespace CRC