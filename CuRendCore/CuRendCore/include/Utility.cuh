#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CRCConfig.h"

#include <vector>
#include <memory>
#include <string>

#include "Component.h" 
#include "Math.cuh"

namespace CRC
{

class UtilityController;

typedef struct CRC_API _UTILITY_ATTRIBUTES
{
    CRC_UTILITY_TYPE type = CRC_UTILITY_TYPE_NULL;
    std::string name = "";
    std::unique_ptr<UtilityController> ctrl = nullptr;
    CRC_SLOT slotRc = CRC_SLOT_INVALID;

    // Camera attributes.
    Vec3d eye = Vec3d(0.0, 0.0, 0.0);
    Vec3d at = Vec3d(0.0, 0.0, 1.0);
    float fov = 80;
    Vec2d aspectRatio = Vec2d(16, 9);
    float nearZ = 0.1;
    float farZ = 1000;
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
    virtual ~Utility();

    friend class UtilityController;
    friend class UtilityFactory;

    friend class Camera;
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