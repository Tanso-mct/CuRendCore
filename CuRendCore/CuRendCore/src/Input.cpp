#include "Input.h"

namespace CRC 
{

Input::Input()
{
    // Initialize the key states.
    for (int i = 0; i < CRC_KEY_MSG_SIZE; i++)
    {
        keyStates[i] = CRC_INPUT_NONE;
        prevKeyExist[i] = false;
    }

    // Initialize the mouse states.
    for (int i = 0; i < CRC_MOUSE_MSG_SIZE; i++)
    {
        mouseStates[i] = CRC_INPUT_NONE;
        prevMouseExist[i] = false;
    }
}

void Input::SetIsKeyDoubleTap(int currentKey)
{
    if (currentKey != CRC_KEY_NULL)
    {
        std::chrono::high_resolution_clock::time_point time = std::chrono::high_resolution_clock::now();

        keyStates[currentKey] = CRC_INPUT_UP;
        if (!prevKeyExist[currentKey])
        {
            prevKeyExist[currentKey] = true;
            prevKeyTimes[currentKey] = time;
        }
        else
        {
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time - prevKeyTimes[currentKey]).count();
            if (elapsed < doubleTapTime)
            {
                keyStates[currentKey] = CRC_INPUT_DOUBLE;
            }

            prevKeyTimes[currentKey] = time;
        }
    }
}

void Input::SetIsMouseDoubleTap(int currentMouse)
{
    if (currentMouse != CRC_KEY_NULL)
    {
        std::chrono::high_resolution_clock::time_point time = std::chrono::high_resolution_clock::now();

        mouseStates[currentMouse] = CRC_INPUT_UP;
        if (!prevMouseExist[currentMouse])
        {
            prevMouseExist[currentMouse] = true;
            prevMouseTimes[currentMouse] = time;
        }
        else
        {
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time - prevMouseTimes[currentMouse]).count();
            if (elapsed < doubleTapTime)
            {
                mouseStates[currentMouse] = CRC_INPUT_DOUBLE;
            }

            prevMouseTimes[currentMouse] = time;
        }
    }
}

Input *Input::GetInstance()
{
    // Implementation of the Singleton pattern.
    static Input* instance = nullptr;

    if (instance == nullptr) instance = new Input();

    return instance;
}

void Input::ReleaseInstance()
{
    Input* instance = GetInstance();
    if (instance != nullptr)
    {
        delete instance;
        instance = nullptr;
    }
}

void Input::Update()
{
    keyText = "";

    for (int i = 0; i < CRC_KEY_MSG_SIZE; i++)
    {
        if (keyStates[i] == CRC_INPUT_DOWN)
        {
            keyStates[i] = CRC_INPUT_PUSHING;
        }
        else if (keyStates[i] == CRC_INPUT_UP || keyStates[i] == CRC_INPUT_DOUBLE)
        {
            keyStates[i] = CRC_INPUT_NONE;
        }
    }

    for (int i = 0; i < CRC_MOUSE_MSG_SIZE; i++)
    {
        if (i == CRC_MOUSE_MSG_WHEEL) continue;
        
        if (mouseStates[i] == CRC_INPUT_DOWN)
        {
            mouseStates[i] = CRC_INPUT_PUSHING;
        }
        else if (mouseStates[i] == CRC_INPUT_UP || mouseStates[i] == CRC_INPUT_DOUBLE)
        {
            mouseStates[i] = CRC_INPUT_NONE;
        }
    }

    mouseStates[CRC_MOUSE_MSG_WHEEL] = CRC_INPUT_NONE;

    mousePos.set(0.0f, 0.0f);
}

void Input::ProcessKeyDown(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    int currentKey = CRC_KEY_NULL;
    BOOL isRight = (lParam & (1 << 24)) != 0;
    bool isShift = 
    (
        keyStates[CRC_KEY_MSG_L_SHIFT] == CRC_INPUT_DOWN || keyStates[CRC_KEY_MSG_L_SHIFT] == CRC_INPUT_PUSHING ||
        keyStates[CRC_KEY_MSG_R_SHIFT] == CRC_INPUT_DOWN || keyStates[CRC_KEY_MSG_R_SHIFT] == CRC_INPUT_PUSHING
    ) ? true : false;

    switch (wParam)
    {
    case VK_MENU:
        if (isRight) currentKey = CRC_KEY_MSG_R_ALT;
        else currentKey = CRC_KEY_MSG_L_ALT;
        break;

    case VK_RETURN:
        currentKey = CRC_KEY_MSG_RETURN;
        break;

    case VK_SPACE:
        currentKey = CRC_KEY_MSG_SPACE;
        break;

    case VK_ESCAPE:
        currentKey = CRC_KEY_MSG_ESCAPE;
        break;

    case VK_TAB:
        currentKey = CRC_KEY_MSG_TAB;
        break;

    case VK_BACK:
        currentKey = CRC_KEY_MSG_BACKSPACE;
        break;

    case VK_SHIFT:
        if (isRight) currentKey = CRC_KEY_MSG_R_SHIFT;
        else currentKey = CRC_KEY_MSG_L_SHIFT;
        break;

    case VK_CONTROL:
        if (isRight) currentKey = CRC_KEY_MSG_R_CTRL;
        else currentKey = CRC_KEY_MSG_L_CTRL;
        break;

    case VK_LWIN:
        currentKey = CRC_KEY_MSG_L_WIN;
        break;

    case VK_RWIN:
        currentKey = CRC_KEY_MSG_R_WIN;
        break;

    case VK_F1:
        currentKey = CRC_KEY_MSG_F1;
        break;

    case VK_F2:
        currentKey = CRC_KEY_MSG_F2;
        break;

    case VK_F3:
        currentKey = CRC_KEY_MSG_F3;
        break;

    case VK_F4:
        currentKey = CRC_KEY_MSG_F4;
        break;

    case VK_F5:
        currentKey = CRC_KEY_MSG_F5;
        break;

    case VK_F6:
        currentKey = CRC_KEY_MSG_F6;
        break;

    case VK_F7:
        currentKey = CRC_KEY_MSG_F7;
        break;

    case VK_F8:
        currentKey = CRC_KEY_MSG_F8;
        break;

    case VK_F9:
        currentKey = CRC_KEY_MSG_F9;
        break;

    case VK_F10:
        currentKey = CRC_KEY_MSG_F10;
        break;

    case VK_F11:
        currentKey = CRC_KEY_MSG_F11;
        break;

    case VK_F12:
        currentKey = CRC_KEY_MSG_F12;
        break;

    case '0':
        currentKey = CRC_KEY_MSG_ALPHA_0;
        keyText = "0";
        break;

    case '1':
        currentKey = CRC_KEY_MSG_ALPHA_1;
        keyText = (isShift) ? "!" : "1";
        break;

    case '2':
        currentKey = CRC_KEY_MSG_ALPHA_2;
        keyText = (isShift) ? "\"" : "2";
        break;

    case '3':
        currentKey = CRC_KEY_MSG_ALPHA_3;
        keyText = (isShift) ? "#" : "3";
        break;

    case '4':
        currentKey = CRC_KEY_MSG_ALPHA_4;
        keyText = (isShift) ? "$" : "4";
        break;

    case '5':
        currentKey = CRC_KEY_MSG_ALPHA_5;
        keyText = (isShift) ? "%" : "5";
        break;

    case '6':
        currentKey = CRC_KEY_MSG_ALPHA_6;
        keyText = (isShift) ? "&" : "6";
        break;

    case '7':
        currentKey = CRC_KEY_MSG_ALPHA_7;
        keyText = (isShift) ? "\'" : "7";
        break;

    case '8':
        currentKey = CRC_KEY_MSG_ALPHA_8;
        keyText = (isShift) ? "(" : "8";
        break;

    case '9':
        currentKey = CRC_KEY_MSG_ALPHA_9;
        keyText = (isShift) ? ")" : "9";
        break;

    case VK_OEM_1:
        currentKey = CRC_KEY_MSG_OEM_1;
        keyText = (isShift) ? "*" : ":";
        break;

    case VK_OEM_PLUS:
        currentKey = CRC_KEY_MSG_OEM_PLUS;
        keyText = (isShift) ? "+" : ";";
        break;

    case VK_OEM_COMMA:
        currentKey = CRC_KEY_MSG_OEM_COMMA;
        keyText = (isShift) ? "<" : ",";
        break;

    case VK_OEM_MINUS:
        currentKey = CRC_KEY_MSG_OEM_MINUS;
        keyText = (isShift) ? "=" : "-";
        break;

    case VK_OEM_PERIOD:
        currentKey = CRC_KEY_MSG_OEM_PERIOD;
        keyText = (isShift) ? ">" : ".";
        break;

    case VK_OEM_2:
        currentKey = CRC_KEY_MSG_OEM_2;
        keyText = (isShift) ? "?" : "/";
        break;

    case VK_OEM_3:
        currentKey = CRC_KEY_MSG_OEM_3;
        keyText = (isShift) ? "`" : "@";
        break;

    case VK_OEM_4:
        currentKey = CRC_KEY_MSG_OEM_4;
        keyText = (isShift) ? "{" : "[";
        break;

    case VK_OEM_5:
        currentKey = CRC_KEY_MSG_OEM_5;
        keyText = (isShift) ? "|" : "\\";
        break;

    case VK_OEM_6:
        currentKey = CRC_KEY_MSG_OEM_6;
        keyText = (isShift) ? "}" : "]";
        break;

    case VK_OEM_7:
        currentKey = CRC_KEY_MSG_OEM_7;
        keyText = (isShift) ? "~" : "^";
        break;

    case VK_OEM_102:
        currentKey = CRC_KEY_MSG_OEM_102;
        keyText = (isShift) ? "_" : "\\";
        break;

    case 'A':
        currentKey = CRC_KEY_MSG_A;
        keyText = (isShift) ? "A" : "a";
        break;

    case 'B':
        currentKey = CRC_KEY_MSG_B;
        keyText = (isShift) ? "B" : "b";
        break;

    case 'C':
        currentKey = CRC_KEY_MSG_C;
        keyText = (isShift) ? "C" : "c";
        break;

    case 'D':
        currentKey = CRC_KEY_MSG_D;
        keyText = (isShift) ? "D" : "d";
        break;

    case 'E':
        currentKey = CRC_KEY_MSG_E;
        keyText = (isShift) ? "E" : "e";
        break;

    case 'F':
        currentKey = CRC_KEY_MSG_F;
        keyText = (isShift) ? "F" : "f";
        break;

    case 'G':
        currentKey = CRC_KEY_MSG_G;
        keyText = (isShift) ? "G" : "g";
        break;

    case 'H':
        currentKey = CRC_KEY_MSG_H;
        keyText = (isShift) ? "H" : "h";
        break;

    case 'I':
        currentKey = CRC_KEY_MSG_I;
        keyText = (isShift) ? "I" : "i";
        break;

    case 'J':
        currentKey = CRC_KEY_MSG_J;
        keyText = (isShift) ? "J" : "j";
        break;

    case 'K':
        currentKey = CRC_KEY_MSG_K;
        keyText = (isShift) ? "K" : "k";
        break;

    case 'L':
        currentKey = CRC_KEY_MSG_L;
        keyText = (isShift) ? "L" : "l";
        break;

    case 'M':
        currentKey = CRC_KEY_MSG_M;
        keyText = (isShift) ? "M" : "m";
        break;

    case 'N':
        currentKey = CRC_KEY_MSG_N;
        keyText = (isShift) ? "N" : "n";
        break;

    case 'O':
        currentKey = CRC_KEY_MSG_O;
        keyText = (isShift) ? "O" : "o";
        break;

    case 'P':
        currentKey = CRC_KEY_MSG_P;
        keyText = (isShift) ? "P" : "p";
        break;

    case 'Q':
        currentKey = CRC_KEY_MSG_Q;
        keyText = (isShift) ? "Q" : "q";
        break;

    case 'R':
        currentKey = CRC_KEY_MSG_R;
        keyText = (isShift) ? "R" : "r";
        break;

    case 'S':
        currentKey = CRC_KEY_MSG_S;
        keyText = (isShift) ? "S" : "s";
        break;

    case 'T':
        currentKey = CRC_KEY_MSG_T;
        keyText = (isShift) ? "T" : "t";
        break;

    case 'U':
        currentKey = CRC_KEY_MSG_U;
        keyText = (isShift) ? "U" : "u";
        break;

    case 'V':
        currentKey = CRC_KEY_MSG_V;
        keyText = (isShift) ? "V" : "v";
        break;

    case 'W':
        currentKey = CRC_KEY_MSG_W;
        keyText = (isShift) ? "W" : "w";
        break;

    case 'X':
        currentKey = CRC_KEY_MSG_X;
        keyText = (isShift) ? "X" : "x";
        break;

    case 'Y':
        currentKey = CRC_KEY_MSG_Y;
        keyText = (isShift) ? "Y" : "y";
        break;

    case 'Z':
        currentKey = CRC_KEY_MSG_Z;
        keyText = (isShift) ? "Z" : "z";
        break;
    }

    if (currentKey != CRC_KEY_NULL)
    {
        auto now = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);

        keyStates[currentKey] = CRC_INPUT_DOWN;
    }
    
}

void Input::ProcessKeyUp(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    int currentKey = CRC_KEY_NULL;
    BOOL isRight = (lParam & (1 << 24)) != 0;
    bool isShift = 
    (
        keyStates[CRC_KEY_MSG_L_SHIFT] == CRC_INPUT_DOWN || keyStates[CRC_KEY_MSG_L_SHIFT] == CRC_INPUT_PUSHING ||
        keyStates[CRC_KEY_MSG_R_SHIFT] == CRC_INPUT_DOWN || keyStates[CRC_KEY_MSG_R_SHIFT] == CRC_INPUT_PUSHING
    ) ? true : false;

    switch (wParam)
    {
    case VK_MENU:
        if (isRight) currentKey = CRC_KEY_MSG_R_ALT;
        else currentKey = CRC_KEY_MSG_L_ALT;
        break;

    case VK_RETURN:
        currentKey = CRC_KEY_MSG_RETURN;
        break;

    case VK_SPACE:
        currentKey = CRC_KEY_MSG_SPACE;
        break;

    case VK_ESCAPE:
        currentKey = CRC_KEY_MSG_ESCAPE;
        break;

    case VK_TAB:
        currentKey = CRC_KEY_MSG_TAB;
        break;

    case VK_BACK:
        currentKey = CRC_KEY_MSG_BACKSPACE;
        break;

    case VK_SHIFT:
        if (isRight) currentKey = CRC_KEY_MSG_R_SHIFT;
        else currentKey = CRC_KEY_MSG_L_SHIFT;
        break;

    case VK_CONTROL:
        if (isRight) currentKey = CRC_KEY_MSG_R_CTRL;
        else currentKey = CRC_KEY_MSG_L_CTRL;
        break;

    case VK_LWIN:
        currentKey = CRC_KEY_MSG_L_WIN;
        break;

    case VK_RWIN:
        currentKey = CRC_KEY_MSG_R_WIN;
        break;

    case VK_F1:
        currentKey = CRC_KEY_MSG_F1;
        break;

    case VK_F2:
        currentKey = CRC_KEY_MSG_F2;
        break;

    case VK_F3:
        currentKey = CRC_KEY_MSG_F3;
        break;

    case VK_F4:
        currentKey = CRC_KEY_MSG_F4;
        break;

    case VK_F5:
        currentKey = CRC_KEY_MSG_F5;
        break;

    case VK_F6:
        currentKey = CRC_KEY_MSG_F6;
        break;

    case VK_F7:
        currentKey = CRC_KEY_MSG_F7;
        break;

    case VK_F8:
        currentKey = CRC_KEY_MSG_F8;
        break;

    case VK_F9:
        currentKey = CRC_KEY_MSG_F9;
        break;

    case VK_F10:
        currentKey = CRC_KEY_MSG_F10;
        break;

    case VK_F11:
        currentKey = CRC_KEY_MSG_F11;
        break;

    case VK_F12:
        currentKey = CRC_KEY_MSG_F12;
        break;

    case '0':
        currentKey = CRC_KEY_MSG_ALPHA_0;
        keyText = "0";
        break;

    case '1':
        currentKey = CRC_KEY_MSG_ALPHA_1;
        keyText = (isShift) ? "!" : "1";
        break;

    case '2':
        currentKey = CRC_KEY_MSG_ALPHA_2;
        keyText = (isShift) ? "\"" : "2";
        break;

    case '3':
        currentKey = CRC_KEY_MSG_ALPHA_3;
        keyText = (isShift) ? "#" : "3";
        break;

    case '4':
        currentKey = CRC_KEY_MSG_ALPHA_4;
        keyText = (isShift) ? "$" : "4";
        break;

    case '5':
        currentKey = CRC_KEY_MSG_ALPHA_5;
        keyText = (isShift) ? "%" : "5";
        break;

    case '6':
        currentKey = CRC_KEY_MSG_ALPHA_6;
        keyText = (isShift) ? "&" : "6";
        break;

    case '7':
        currentKey = CRC_KEY_MSG_ALPHA_7;
        keyText = (isShift) ? "\'" : "7";
        break;

    case '8':
        currentKey = CRC_KEY_MSG_ALPHA_8;
        keyText = (isShift) ? "(" : "8";
        break;

    case '9':
        currentKey = CRC_KEY_MSG_ALPHA_9;
        keyText = (isShift) ? ")" : "9";
        break;

    case VK_OEM_1:
        currentKey = CRC_KEY_MSG_OEM_1;
        keyText = (isShift) ? "*" : ":";
        break;

    case VK_OEM_PLUS:
        currentKey = CRC_KEY_MSG_OEM_PLUS;
        keyText = (isShift) ? "+" : ";";
        break;

    case VK_OEM_COMMA:
        currentKey = CRC_KEY_MSG_OEM_COMMA;
        keyText = (isShift) ? "<" : ",";
        break;

    case VK_OEM_MINUS:
        currentKey = CRC_KEY_MSG_OEM_MINUS;
        keyText = (isShift) ? "=" : "-";
        break;

    case VK_OEM_PERIOD:
        currentKey = CRC_KEY_MSG_OEM_PERIOD;
        keyText = (isShift) ? ">" : ".";
        break;

    case VK_OEM_2:
        currentKey = CRC_KEY_MSG_OEM_2;
        keyText = (isShift) ? "?" : "/";
        break;

    case VK_OEM_3:
        currentKey = CRC_KEY_MSG_OEM_3;
        keyText = (isShift) ? "`" : "@";
        break;

    case VK_OEM_4:
        currentKey = CRC_KEY_MSG_OEM_4;
        keyText = (isShift) ? "{" : "[";
        break;

    case VK_OEM_5:
        currentKey = CRC_KEY_MSG_OEM_5;
        keyText = (isShift) ? "|" : "\\";
        break;

    case VK_OEM_6:
        currentKey = CRC_KEY_MSG_OEM_6;
        keyText = (isShift) ? "}" : "]";
        break;

    case VK_OEM_7:
        currentKey = CRC_KEY_MSG_OEM_7;
        keyText = (isShift) ? "~" : "^";
        break;

    case VK_OEM_102:
        currentKey = CRC_KEY_MSG_OEM_102;
        keyText = (isShift) ? "_" : "\\";
        break;

    case 'A':
        currentKey = CRC_KEY_MSG_A;
        keyText = (isShift) ? "A" : "a";
        break;

    case 'B':
        currentKey = CRC_KEY_MSG_B;
        keyText = (isShift) ? "B" : "b";
        break;

    case 'C':
        currentKey = CRC_KEY_MSG_C;
        keyText = (isShift) ? "C" : "c";
        break;

    case 'D':
        currentKey = CRC_KEY_MSG_D;
        keyText = (isShift) ? "D" : "d";
        break;

    case 'E':
        currentKey = CRC_KEY_MSG_E;
        keyText = (isShift) ? "E" : "e";
        break;

    case 'F':
        currentKey = CRC_KEY_MSG_F;
        keyText = (isShift) ? "F" : "f";
        break;

    case 'G':
        currentKey = CRC_KEY_MSG_G;
        keyText = (isShift) ? "G" : "g";
        break;

    case 'H':
        currentKey = CRC_KEY_MSG_H;
        keyText = (isShift) ? "H" : "h";
        break;

    case 'I':
        currentKey = CRC_KEY_MSG_I;
        keyText = (isShift) ? "I" : "i";
        break;

    case 'J':
        currentKey = CRC_KEY_MSG_J;
        keyText = (isShift) ? "J" : "j";
        break;

    case 'K':
        currentKey = CRC_KEY_MSG_K;
        keyText = (isShift) ? "K" : "k";
        break;

    case 'L':
        currentKey = CRC_KEY_MSG_L;
        keyText = (isShift) ? "L" : "l";
        break;

    case 'M':
        currentKey = CRC_KEY_MSG_M;
        keyText = (isShift) ? "M" : "m";
        break;

    case 'N':
        currentKey = CRC_KEY_MSG_N;
        keyText = (isShift) ? "N" : "n";
        break;

    case 'O':
        currentKey = CRC_KEY_MSG_O;
        keyText = (isShift) ? "O" : "o";
        break;

    case 'P':
        currentKey = CRC_KEY_MSG_P;
        keyText = (isShift) ? "P" : "p";
        break;

    case 'Q':
        currentKey = CRC_KEY_MSG_Q;
        keyText = (isShift) ? "Q" : "q";
        break;

    case 'R':
        currentKey = CRC_KEY_MSG_R;
        keyText = (isShift) ? "R" : "r";
        break;

    case 'S':
        currentKey = CRC_KEY_MSG_S;
        keyText = (isShift) ? "S" : "s";
        break;

    case 'T':
        currentKey = CRC_KEY_MSG_T;
        keyText = (isShift) ? "T" : "t";
        break;

    case 'U':
        currentKey = CRC_KEY_MSG_U;
        keyText = (isShift) ? "U" : "u";
        break;

    case 'V':
        currentKey = CRC_KEY_MSG_V;
        keyText = (isShift) ? "V" : "v";
        break;

    case 'W':
        currentKey = CRC_KEY_MSG_W;
        keyText = (isShift) ? "W" : "w";
        break;

    case 'X':
        currentKey = CRC_KEY_MSG_X;
        keyText = (isShift) ? "X" : "x";
        break;

    case 'Y':
        currentKey = CRC_KEY_MSG_Y;
        keyText = (isShift) ? "Y" : "y";
        break;

    case 'Z':
        currentKey = CRC_KEY_MSG_Z;
        keyText = (isShift) ? "Z" : "z";
        break;
    }

    SetIsKeyDoubleTap(currentKey);

    // if (currentKey != CRC_KEY_NULL)
    // {
    //     std::chrono::high_resolution_clock::time_point time = std::chrono::high_resolution_clock::now();

    //     keyStates[currentKey] = CRC_INPUT_UP;
    //     if (!prevKeyExist[currentKey])
    //     {
    //         prevKeyExist[currentKey] = true;
    //         prevKeyTimes[currentKey] = time;
    //     }
    //     else
    //     {
    //         double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time - prevKeyTimes[currentKey]).count();
    //         if (elapsed < doubleTapTime)
    //         {
    //             keyStates[currentKey] = CRC_INPUT_DOUBLE;
    //         }

    //         prevKeyTimes[currentKey] = time;
    //     }
    // }
}

void Input::ProcessMouse(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_RBUTTONDOWN:
        mouseStates[CRC_MOUSE_MSG_RBTN] = CRC_INPUT_DOWN;
        break;

    case WM_RBUTTONUP:
        mouseStates[CRC_MOUSE_MSG_RBTN] = CRC_INPUT_UP;
        SetIsMouseDoubleTap(CRC_MOUSE_MSG_RBTN);
        break;

    case WM_LBUTTONDOWN:
        mouseStates[CRC_MOUSE_MSG_LBTN] = CRC_INPUT_DOWN;
        break;

    case WM_LBUTTONUP:
        mouseStates[CRC_MOUSE_MSG_LBTN] = CRC_INPUT_UP;
        SetIsMouseDoubleTap(CRC_MOUSE_MSG_LBTN);
        break;

    case WM_MBUTTONDOWN:
        mouseStates[CRC_MOUSE_MSG_MBTN] = CRC_INPUT_DOWN;
        break;
        
    case WM_MBUTTONUP:
        mouseStates[CRC_MOUSE_MSG_MBTN] = CRC_INPUT_UP;
        SetIsMouseDoubleTap(CRC_MOUSE_MSG_MBTN);
        break;

    case WM_MOUSEWHEEL:
        mouseStates[CRC_MOUSE_MSG_WHEEL] = GET_WHEEL_DELTA_WPARAM(wParam);
        break;

    case WM_MOUSEMOVE:
        mousePos.x = LOWORD(lParam);
        mousePos.y = HIWORD(lParam);
        break;
    }
}

}