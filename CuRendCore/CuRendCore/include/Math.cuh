#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CRCConfig.h"

namespace CRC
{

typedef struct CRC_API _Vec2d
{
    float x;
    float y;

    __device__ __host__ _Vec2d() : x(0), y(0) {}
    __device__ __host__ _Vec2d(float x, float y) : x(x), y(y) {}

    __device__ __host__ void set(float argX, float argY)
    {
        x = argX;
        y = argY;
    }

    __device__ __host__ _Vec2d operator+(const _Vec2d& other) const
    {
        return _Vec2d(x + other.x, y + other.y);
    }

    __device__ __host__ _Vec2d operator-(const _Vec2d& other) const
    {
        return _Vec2d(x - other.x, y - other.y);
    }

    __device__ __host__ _Vec2d operator*(float scalar) const
    {
        return _Vec2d(x * scalar, y * scalar);
    }

    __device__ __host__ _Vec2d operator/(float scalar) const
    {
        return _Vec2d(x / scalar, y / scalar);
    }

    __device__ __host__ BOOL operator==(const _Vec2d& other) const
    {
        return (x == other.x && y == other.y) ? TRUE : FALSE;
    }

    __device__ __host__ BOOL operator!=(const _Vec2d& other) const
    {
        return (!(*this == other)) ? TRUE : FALSE;
    }

    __device__ __host__ void  operator+=(const _Vec2d& other)
    {
        x += other.x;
        y += other.y;
    }

    __device__ __host__ void  operator-=(const _Vec2d& other)
    {
        x -= other.x;
        y -= other.y;
    }

    __device__ __host__ void  operator*=(float scalar)
    {
        x *= scalar;
        y *= scalar;
    }

    __device__ __host__ void  operator/=(float scalar)
    {
        x /= scalar;
        y /= scalar;
    }

    __device__ __host__ _Vec2d operator=(const _Vec2d& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }
} Vec2d;

typedef struct CRC_API _Vec3d
{
    float x;
    float y;
    float z;

    __device__ __host__ _Vec3d() : x(0), y(0), z(0) {}
    __device__ __host__ _Vec3d(float x, float y, float z) : x(x), y(y), z(z) {}

    __device__ __host__ void set(float argX, float argY, float argZ)
    {
        x = argX;
        y = argY;
        z = argZ;
    }

    __device__ __host__ _Vec3d operator+(const _Vec3d& other) const
    {
        return _Vec3d(x + other.x, y + other.y, z + other.z);
    }

    __device__ __host__ _Vec3d operator-(const _Vec3d& other) const
    {
        return _Vec3d(x - other.x, y - other.y, z - other.z);
    }

    __device__ __host__ _Vec3d operator*(float scalar) const
    {
        return _Vec3d(x * scalar, y * scalar, z * scalar);
    }

    __device__ __host__ _Vec3d operator/(float scalar) const
    {
        return _Vec3d(x / scalar, y / scalar, z / scalar);
    }

    __device__ __host__ BOOL operator==(const _Vec3d& other) const
    {
        return (x == other.x && y == other.y && z == other.z) ? TRUE : FALSE;
    }

    __device__ __host__ BOOL operator!=(const _Vec3d& other) const
    {
        return (!(*this == other)) ? TRUE : FALSE;
    }

    __device__ __host__ void operator+=(const _Vec3d& other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
    }

    __device__ __host__ void operator-=(const _Vec3d& other)
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;
    }

    __device__ __host__ void operator*=(float scalar)
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;
    }

    __device__ __host__ void operator/=(float scalar)
    {
        x /= scalar;
        y /= scalar;
        z /= scalar;
    }

    __device__ __host__ _Vec3d operator=(const _Vec3d& other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    }
    
} Vec3d;

typedef struct CRC_API _BoundingBox
{
    bool isEmpty = true;
    Vec3d origin;
    Vec3d opposite;
    Vec3d vertices[8];

    __device__ __host__ _BoundingBox() : isEmpty(true) {}

    __device__ __host__ void StartToCreate()
    {
        isEmpty = true;
    }

    __device__ __host__ void AddPoint(Vec3d point)
    {
        CRC_GIF(isEmpty == true, br2)
        {
            origin = point;
            opposite = point;
        }
        CRC_GIF(isEmpty == false, br1)
        {
            if (point.x < origin.x) origin.x = point.x;
            if (point.y < origin.y) origin.y = point.y;
            if (point.z > origin.z) origin.z = point.z;

            if (point.x > opposite.x) opposite.x = point.x;
            if (point.y > opposite.y) opposite.y = point.y;
            if (point.z < opposite.z) opposite.z = point.z;
        }

        isEmpty = false;
    }

    __device__ __host__ void Update()
    {
        vertices[0] = Vec3d(origin.x, opposite.y, origin.z);
        vertices[1] = Vec3d(opposite.x, opposite.y, origin.z);
        vertices[2] = Vec3d(opposite.x, origin.y, origin.z);
        vertices[3] = Vec3d(origin.x, origin.y, origin.z);
        vertices[4] = Vec3d(origin.x, opposite.y, opposite.z);
        vertices[5] = Vec3d(opposite.x, opposite.y, opposite.z);
        vertices[6] = Vec3d(opposite.x, origin.y, opposite.z);
        vertices[7] = Vec3d(origin.x, origin.y, opposite.z);
    }

} BoundingBox;

typedef struct CRC_API _Polygon
{
    unsigned int size;
    CRC_INDEX idWv[3];
    CRC_INDEX idUv[3];
    Vec3d normal;

    __device__ __host__ _Polygon() : size(0) {}

    __device__ __host__ void StartToCreate()
    {
        size = 0;
    }

    __device__ __host__ void AddIndex(CRC_INDEX argIdWv, CRC_INDEX argIdUv)
    {
        idWv[size] = argIdWv;
        idUv[size] = argIdUv;
        size++;
    }

    __device__ __host__ void SetNormal(Vec3d argNormal)
    {
        normal = argNormal;
    }

} Polygon;

}