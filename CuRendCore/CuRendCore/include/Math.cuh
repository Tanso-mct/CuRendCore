#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>

#include "CRCConfig.h"

namespace CRC
{

constexpr const float PI = 3.14159265358979323846f;

inline float RtoD(float radian)
{
    return radian * PI / 180.0f;
}

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

    __device__ __host__ float Length()
    {
        return std::sqrt(x * x + y * y);
    }

    __device__ __host__ float Dot(const _Vec2d& other)
    {
        return x * other.x + y * other.y;
    }

    __device__ __host__ float Cross(const _Vec2d& other)
    {
        return x * other.y - y * other.x;
    }

    __device__ __host__ _Vec2d Normalize()
    {
        float length = Length();
        return _Vec2d(x / length, y / length);
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

    __device__ __host__ float Length()
    {
        return std::sqrt(x * x + y * y + z * z);
    }

    __device__ __host__ float Dot(const _Vec3d& other)
    {
        return x * other.x + y * other.y + z * other.z;
    }

    __device__ __host__ _Vec3d Cross(const _Vec3d& other)
    {
        return _Vec3d
        (
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    __device__ __host__ _Vec3d Normalize()
    {
        float length = Length();
        return _Vec3d(x / length, y / length, z / length);
    }
    
} Vec3d;

typedef struct CRC_API _Matrix
{
    float m[4][4];

    __device__ __host__ _Matrix()
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                m[i][j] = 0;
            }
        }

        m[0][0] = 1;
        m[1][1] = 1;
        m[2][2] = 1;
        m[3][3] = 1;
    }

    __device__ __host__ _Matrix
    (
        float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33
    ){
        m[0][0] = m00;
        m[0][1] = m01;
        m[0][2] = m02;
        m[0][3] = m03;

        m[1][0] = m10;
        m[1][1] = m11;
        m[1][2] = m12;
        m[1][3] = m13;

        m[2][0] = m20;
        m[2][1] = m21;
        m[2][2] = m22;
        m[2][3] = m23;

        m[3][0] = m30;
        m[3][1] = m31;
        m[3][2] = m32;
        m[3][3] = m33;
    }

    __device__ __host__ _Matrix(float list[16])
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                m[i][j] = list[i * 4 + j];
            }
        }
    }

    __device__ __host__ void Set(float list[16])
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                m[i][j] = list[i * 4 + j];
            }
        }
    }

    __device__ __host__ void Identity()
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                m[i][j] = 0;
            }
        }

        m[0][0] = 1;
        m[1][1] = 1;
        m[2][2] = 1;
        m[3][3] = 1;
    }

    __device__ __host__ _Matrix operator+(const _Matrix& other) const
    {
        _Matrix result;

        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                result.m[row][col] = m[row][col] + other.m[row][col];
            }
        }

        return result;
    }

    __device__ __host__ _Matrix operator-(const _Matrix& other) const
    {
        _Matrix result;

        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                result.m[row][col] = m[row][col] - other.m[row][col];
            }
        }

        return result;
    }

    __device__ __host__ _Matrix operator*(const _Matrix& other) const
    {
        _Matrix result;

        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                result.m[row][col] = 0;
                for (int i = 0; i < 4; i++)
                {
                    result.m[row][col] += m[row][i] * other.m[i][col];
                }
            }
        }

        return result;
    }

    __device__ __host__ _Matrix operator/(float scalar) const
    {
        _Matrix result;

        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                result.m[row][col] = m[row][col] / scalar;
            }
        }

        return result;
    }

    __device__ __host__ BOOL operator==(const _Matrix& other) const
    {
        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                if (m[row][col] != other.m[row][col]) return FALSE;
            }
        }

        return TRUE;
    }

    __device__ __host__ BOOL operator!=(const _Matrix& other) const
    {
        return (!(*this == other)) ? TRUE : FALSE;
    }

    __device__ __host__ void operator+=(const _Matrix& other)
    {
        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                m[row][col] += other.m[row][col];
            }
        }
    }

    __device__ __host__ void operator-=(const _Matrix& other)
    {
        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                m[row][col] -= other.m[row][col];
            }
        }
    }

    __device__ __host__ void operator*=(const _Matrix& other)
    {
        _Matrix result;

        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                result.m[row][col] = 0;
                for (int i = 0; i < 4; i++)
                {
                    result.m[row][col] += m[row][i] * other.m[i][col];
                }
            }
        }

        *this = result;
    }

    __device__ __host__ void operator/=(float scalar)
    {
        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                m[row][col] /= scalar;
            }
        }
    }

    __device__ __host__ _Matrix operator=(const _Matrix& other)
    {
        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                m[row][col] = other.m[row][col];
            }
        }

        return *this;
    }

    __device__ __host__ Vec3d ProductLeft(Vec3d& vec)
    {
        Vec3d result;

        result.x = m[0][0] * vec.x + m[0][1] * vec.y + m[0][2] * vec.z + m[0][3];
        result.y = m[1][0] * vec.x + m[1][1] * vec.y + m[1][2] * vec.z + m[1][3];
        result.z = m[2][0] * vec.x + m[2][1] * vec.y + m[2][2] * vec.z + m[2][3];

        return result;
    }

} Matrix;

inline Matrix MatrixSet
(
    float m00, float m01, float m02, float m03,
    float m10, float m11, float m12, float m13,
    float m20, float m21, float m22, float m23,
    float m30, float m31, float m32, float m33
){
    Matrix result;

    result.m[0][0] = m00;
    result.m[0][1] = m01;
    result.m[0][2] = m02;
    result.m[0][3] = m03;

    result.m[1][0] = m10;
    result.m[1][1] = m11;
    result.m[1][2] = m12;
    result.m[1][3] = m13;

    result.m[2][0] = m20;
    result.m[2][1] = m21;
    result.m[2][2] = m22;
    result.m[2][3] = m23;

    result.m[3][0] = m30;
    result.m[3][1] = m31;
    result.m[3][2] = m32;
    result.m[3][3] = m33;

    return result;
}

inline Matrix MatrixScaling(Vec3d& scale)
{
    return MatrixSet
    (
        scale.x, 0, 0, 0,
        0, scale.y, 0, 0,
        0, 0, scale.z, 0,
        0, 0, 0, 1
    );
}

inline Matrix MatrixRotationX(float angle)
{
    return MatrixSet
    (
        1, 0, 0, 0,
        0, std::cos(angle), -std::sin(angle), 0,
        0, std::sin(angle), std::cos(angle), 0,
        0, 0, 0, 1
    );
}

inline Matrix MatrixRotationY(float angle)
{
    return MatrixSet
    (
        std::cos(angle), 0, std::sin(angle), 0,
        0, 1, 0, 0,
        -std::sin(angle), 0, std::cos(angle), 0,
        0, 0, 0, 1
    );
}

inline Matrix MatrixRotationZ(float angle)
{
    return MatrixSet
    (
        std::cos(angle), -std::sin(angle), 0, 0,
        std::sin(angle), std::cos(angle), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    );
}

inline Matrix MatrixTranslation(Vec3d& translation)
{
    return MatrixSet
    (
        1, 0, 0, translation.x,
        0, 1, 0, translation.y,
        0, 0, 1, translation.z,
        0, 0, 0, 1
    );
}

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