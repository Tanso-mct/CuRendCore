#pragma once

#include "CRCConfig.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <vector>

#include "Resource.cuh"
#include "Math.cuh"

namespace CRC
{

class CRC_API PngFile : public Resource
{
private:
    PngFile(RESOURCE_ATTR& rattr) : Resource(rattr) {CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};

    int width = 0;
    int height = 0;
    int channels = 0;
    LPDWORD data = nullptr;

public:
    ~PngFile() override;

    HRESULT Load() override;
    HRESULT Unload() override;

    friend class ResourceFactory;
};

class CRC_API ObjFile : public Resource
{
private:
    ObjFile(RESOURCE_ATTR& rattr) : Resource(rattr) {CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");};

    std::vector<Vec3d> wv;
    std::vector<Vec2d> uv;
    std::vector<Vec3d> normal;

    BoundingBox* boundingBox = nullptr;
    std::vector<Polygon> polygons;

public:
    ~ObjFile() override;

    HRESULT Load() override;
    HRESULT Unload() override;

    friend class ResourceFactory;
};

} // namespace CRC