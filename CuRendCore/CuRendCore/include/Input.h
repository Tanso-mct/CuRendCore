#pragma once

#include "CRCConfig.h"

#include <chrono>
#include <ctime>
#include <string>

#include "Math.cuh"

namespace CRC
{

class CRC_API Input
{
private:
    std::string keyText = "";

    double doubleTapTime = 300;

    UINT keyStates[CRC_KEY_MSG_SIZE];
    bool prevKeyExist[CRC_KEY_MSG_SIZE];
    std::chrono::high_resolution_clock::time_point prevKeyTimes[CRC_KEY_MSG_SIZE];
    
    UINT mouseStates[CRC_MOUSE_MSG_SIZE];
    bool prevMouseExist[CRC_MOUSE_MSG_SIZE];
    std::chrono::high_resolution_clock::time_point prevMouseTimes[CRC_MOUSE_MSG_SIZE];

    Vec2d mousePos;
    Vec2d recMousePos;

    Input();

    void SetIsKeyDoubleTap(int currentKey);
    void SetIsMouseDoubleTap(int currentMouse);

public:
    ~Input() = default;

    Input(const Input&) = delete; // Delete copy constructor
    Input& operator=(const Input&) = delete; // Remove copy assignment operator

    Input(Input&&) = delete; // Delete move constructor
    Input& operator=(Input&&) = delete; // Delete move assignment operator

    void Update();
    void SetDoubleTapTime(double time){doubleTapTime = time;};

    void ProcessKeyDown(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    void ProcessKeyUp(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    void ProcessMouse(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

    void GetKeyText(std::string &text){text = keyText;};
    std::string GetKeyText(){return keyText;};

    bool IsKey(CRC_KEY_MSG key){return keyStates[key] == CRC_INPUT_DOWN || keyStates[key] == CRC_INPUT_PUSHING;};
    bool IsKeyDown(CRC_KEY_MSG key){return keyStates[key] == CRC_INPUT_DOWN;};
    bool IsKeyUp(CRC_KEY_MSG key){return keyStates[key] == CRC_INPUT_UP;};
    bool IsKeyDouble(CRC_KEY_MSG key){return keyStates[key] == CRC_INPUT_DOUBLE;};

    bool IsMouse(CRC_MOUSE_MSG button){return mouseStates[button] == CRC_INPUT_DOWN || mouseStates[button] == CRC_INPUT_PUSHING;};
    bool IsMouseDown(CRC_MOUSE_MSG button){return mouseStates[button] == CRC_INPUT_DOWN;};
    bool IsMouseUp(CRC_MOUSE_MSG button){return mouseStates[button] == CRC_INPUT_UP;};
    bool IsMouseDouble(CRC_MOUSE_MSG button){return mouseStates[button] == CRC_INPUT_DOUBLE;};

    void GetMouseWheelDelta(int &delta){delta = mouseStates[CRC_MOUSE_MSG_WHEEL];};
    int GetMouseWheelDelta(){return mouseStates[CRC_MOUSE_MSG_WHEEL];};

    void GetMousePos(float &x, float &y){x = mousePos.x; y = mousePos.y;};
    void GetMousePos(Vec2d &pos){pos = mousePos;};
    Vec2d GetMousePos(){return mousePos;};

    void RecMousePos(){recMousePos = mousePos;};
    void GetMouseRecPos(float &x, float &y){x = recMousePos.x; y = recMousePos.y;};
    void GetMouseRecPos(Vec2d &pos){pos = recMousePos;};
    Vec2d GetMouseRecPos(){return recMousePos;};

    void GetMouseDelta(float &x, float &y){x = mousePos.x - recMousePos.x; y = mousePos.y - recMousePos.y;};
    void GetMouseDelta(Vec2d &delta){delta = mousePos - recMousePos;};
    Vec2d GetMouseDelta(){return mousePos - recMousePos;};

    friend class WindowFactory;
};

} // namespace CRC