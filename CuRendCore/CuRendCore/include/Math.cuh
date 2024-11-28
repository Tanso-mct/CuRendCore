#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CRCConfig.h"

namespace CRC
{

typedef struct _Vec2d
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

typedef struct _Vec3d
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

}