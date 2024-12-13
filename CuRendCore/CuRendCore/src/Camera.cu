#include "Camera.cuh"

namespace CRC
{

Camera::Camera(UTILITYATTR& utattr) : Utility(utattr)
{
    CRCDebugOutput(__FILE__, __FUNCTION__, __LINE__, "");

    eye = utattr.eye;
    at = utattr.at;
    fov = utattr.fov;
    aspectRatio = utattr.aspectRatio;
    ddata->nearZ = utattr.nearZ;
    ddata->farZ = utattr.farZ;

    // Device attributes.
    cudaHostAlloc((void**)&ddata, sizeof(CAMERA_DDATA), cudaHostAllocMapped);

    // Initialize the device attributes.
    ddata->mtView.Identity();

    for (int i = 0; i < 8; i++)
    {
        ddata->viewVolumeVs[i] = Vec3d(0, 0, 0);
    }
}

void Camera::SetViewMatrix()
{
    ddata->mtView.Identity();

    Vec3d invertEye = Vec3d(-eye.x, -eye.y, -eye.z);

    Vec3d rot;
    rot.x = -std::atan(at.y / at.z);
    rot.y = -std::atan(at.z / at.x);
    rot.z = -std::atan(at.x / at.y);

    ddata->mtView = MatrixTranslation(invertEye);
    ddata->mtView *= MatrixRotationX(rot.x);
    ddata->mtView *= MatrixRotationY(rot.y);
    ddata->mtView *= MatrixRotationZ(rot.z);
}

void Camera::SetViewVolume()
{
    
}

} // namespace CRC