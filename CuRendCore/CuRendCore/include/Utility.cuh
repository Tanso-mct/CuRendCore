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

    Vec3d pos = Vec3d(0.0, 0.0, 0.0);
    Vec3d rot = Vec3d(0.0, 0.0, 0.0);
    Vec3d scl = Vec3d(1.0, 1.0, 1.0);

    // Camera attributes.
    Vec3d lookAt = Vec3d(0.0, 0.0, 1.0);
    float fov = 80;
    Vec2d aspectRatio = Vec2d(16, 9);
    float nearZ = 0.1;
    float farZ = 1000;
} UTILITY_ATTR;

class Utility : public Component
{
private:
    Utility(UTILITY_ATTR& utattr);

    // Utility attributes.
    std::string name = "";
    std::unique_ptr<UtilityController> ctrl = nullptr;
    CRC_SLOT slotRc = CRC_SLOT_INVALID;

public:
    virtual ~Utility();

    friend class UtilityController;
    friend class UtilityFactory;

    friend class Camera;
    friend class Light;
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

    CRC_SLOT CreateUtility(UTILITY_ATTR& utattr);
    std::shared_ptr<Utility> GetUtility(CRC_SLOT slotUtility);
    HRESULT DestroyUtility(CRC_SLOT slotUtility);

    friend class Group;
};



} // namespace CRC