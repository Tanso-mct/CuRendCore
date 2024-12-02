#pragma once

#include "CRCConfig.h"

#include <string>
#include <vector>

#include "Resource.h"
#include "Math.cuh"

namespace CRC
{

class CRC_API PngFile : public Resource
{
private:
    PngFile(RESOURCEATTR rattr) : Resource(rattr) {};

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
    ObjFile(RESOURCEATTR rattr) : Resource(rattr) {};

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