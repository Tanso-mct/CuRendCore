#include "Camera.cuh"

namespace CRC
{

Camera::Camera(UTILITY_ATTR& utattr) : Utility(utattr)
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    pos = utattr.pos;
    rot = utattr.rot;
    scl = utattr.scl;

    lookAt = utattr.lookAt;
    fov = utattr.fov;
    aspectRatio = utattr.aspectRatio;
    nearZ = utattr.nearZ;
    farZ = utattr.farZ;
}

void Camera::SetViewMatrix()
{
    // ddata->mtView.Identity();

    // Vec3d invertEye = Vec3d(-eye.x, -eye.y, -eye.z);

    // Vec3d rot;
    // rot.x = -std::atan(at.y / at.z);
    // rot.y = -std::atan(at.z / at.x);
    // rot.z = -std::atan(at.x / at.y);

    // ddata->mtView = MatrixTranslation(invertEye);
    // ddata->mtView *= MatrixRotationX(rot.x);
    // ddata->mtView *= MatrixRotationY(rot.y);
    // ddata->mtView *= MatrixRotationZ(rot.z);
}

void Camera::SetViewVolume()
{
    
}

} // namespace CRC