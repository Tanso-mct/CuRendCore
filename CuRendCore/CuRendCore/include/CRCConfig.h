#pragma once

#define BUILDING_CRC_DLL

#ifdef BUILDING_CRC_DLL
#define CRC_API __declspec(dllexport)
#else
#define CRC_API __declspec(dllimport)
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN // Extract only the essentials from the large number of descriptions in Windows.h.
#endif

#include <Windows.h>
#include <string>

// Represents the number of elements in an array, etc.
#define CRC_SLOT unsigned int
#define CRC_SLOT_INVALID -1

#define CRC_INDEX unsigned int
#define CRC_INDEX_INVALID -1

// A macro to return the result of a function on the GPU, equivalent to HRESULT on the CPU.
#define CRC_GRESULT unsigned int
#define CRC_GSUCCEEDED(hr) (((CRC_GRESULT)(hr)) == 0)
#define CRC_GFAILED(hr) (((CRC_GRESULT)(hr)) != 0)

#define CRC_GS_OK 0
#define CRC_GS_FALSE 1

// Macros for branch processing on the GPU.
#define CRC_GBOOL int

#define CRC_GCO(condition, trueValue, falseValue) \
    condition ? trueValue : falseValue;

#define CRC_GIF(condition, branch) \
    for(int branch = 0; branch < (condition) ? TRUE : FALSE; branch++)


// Default window attributes.
#define CRC_WND_DEFAULT_NAME L"CuRendCore Window"
#define CRC_WND_DEFAULT_POS_X CW_USEDEFAULT
#define CRC_WND_DEFAULT_POS_Y CW_USEDEFAULT
#define CRC_WND_DEFAULT_WIDTH 800
#define CRC_WND_DEFAULT_HEIGHT 600
#define CRC_WND_DEFAULT_STYLE WS_OVERLAPPEDWINDOW

// Input process
#define CRC_INPUT_NONE 0
#define CRC_INPUT_DOWN 1
#define CRC_INPUT_PUSHING 2
#define CRC_INPUT_UP 3
#define CRC_INPUT_DOUBLE 4

// Input key codes.
#define CRC_KEY_NULL -1
enum CRC_KEY_MSG
{
    CRC_KEY_MSG_R_ALT = 0,
    CRC_KEY_MSG_L_ALT,
    CRC_KEY_MSG_RETURN,
    CRC_KEY_MSG_ESCAPE,
    CRC_KEY_MSG_SPACE,
    CRC_KEY_MSG_TAB,
    CRC_KEY_MSG_BACKSPACE,
    CRC_KEY_MSG_L_SHIFT,
    CRC_KEY_MSG_R_SHIFT,
    CRC_KEY_MSG_L_CTRL,
    CRC_KEY_MSG_R_CTRL,
    CRC_KEY_MSG_L_WIN,
    CRC_KEY_MSG_R_WIN,
    CRC_KEY_MSG_UP,
    CRC_KEY_MSG_DOWN,
    CRC_KEY_MSG_LEFT,
    CRC_KEY_MSG_RIGHT,
    CRC_KEY_MSG_INSERT,
    CRC_KEY_MSG_DELETE,
    CRC_KEY_MSG_HOME,
    CRC_KEY_MSG_END,
    CRC_KEY_MSG_PAGE_UP,
    CRC_KEY_MSG_PAGE_DOWN,
    CRC_KEY_MSG_CAPS_LOCK,
    CRC_KEY_MSG_F1,
    CRC_KEY_MSG_F2,
    CRC_KEY_MSG_F3,
    CRC_KEY_MSG_F4,
    CRC_KEY_MSG_F5,
    CRC_KEY_MSG_F6,
    CRC_KEY_MSG_F7,
    CRC_KEY_MSG_F8,
    CRC_KEY_MSG_F9,
    CRC_KEY_MSG_F10,
    CRC_KEY_MSG_F11,
    CRC_KEY_MSG_F12,
    CRC_KEY_MSG_F13,
    CRC_KEY_MSG_ALPHA_0,
    CRC_KEY_MSG_ALPHA_1,
    CRC_KEY_MSG_ALPHA_2,
    CRC_KEY_MSG_ALPHA_3,
    CRC_KEY_MSG_ALPHA_4,
    CRC_KEY_MSG_ALPHA_5,
    CRC_KEY_MSG_ALPHA_6,
    CRC_KEY_MSG_ALPHA_7,
    CRC_KEY_MSG_ALPHA_8,
    CRC_KEY_MSG_ALPHA_9,
    CRC_KEY_MSG_NUMPAD_0,
    CRC_KEY_MSG_NUMPAD_1,
    CRC_KEY_MSG_NUMPAD_2,
    CRC_KEY_MSG_NUMPAD_3,
    CRC_KEY_MSG_NUMPAD_4,
    CRC_KEY_MSG_NUMPAD_5,
    CRC_KEY_MSG_NUMPAD_6,
    CRC_KEY_MSG_NUMPAD_7,
    CRC_KEY_MSG_NUMPAD_8,
    CRC_KEY_MSG_NUMPAD_9,
    CRC_KEY_MSG_NUMPAD_ENTER,
    CRC_KEY_MSG_A,
    CRC_KEY_MSG_B,
    CRC_KEY_MSG_C,
    CRC_KEY_MSG_D,
    CRC_KEY_MSG_E,
    CRC_KEY_MSG_F,
    CRC_KEY_MSG_G,
    CRC_KEY_MSG_H,
    CRC_KEY_MSG_I,
    CRC_KEY_MSG_J,
    CRC_KEY_MSG_K,
    CRC_KEY_MSG_L,
    CRC_KEY_MSG_M,
    CRC_KEY_MSG_N,
    CRC_KEY_MSG_O,
    CRC_KEY_MSG_P,
    CRC_KEY_MSG_Q,
    CRC_KEY_MSG_R,
    CRC_KEY_MSG_S,
    CRC_KEY_MSG_T,
    CRC_KEY_MSG_U,
    CRC_KEY_MSG_V,
    CRC_KEY_MSG_W,
    CRC_KEY_MSG_X,
    CRC_KEY_MSG_Y,
    CRC_KEY_MSG_Z,
    CRC_KEY_MSG_OEM_1,
    CRC_KEY_MSG_OEM_PLUS,
    CRC_KEY_MSG_OEM_COMMA,
    CRC_KEY_MSG_OEM_MINUS,
    CRC_KEY_MSG_OEM_PERIOD,
    CRC_KEY_MSG_OEM_2,
    CRC_KEY_MSG_OEM_3,
    CRC_KEY_MSG_OEM_4,
    CRC_KEY_MSG_OEM_5,
    CRC_KEY_MSG_OEM_6,
    CRC_KEY_MSG_OEM_7,
    CRC_KEY_MSG_OEM_102,
    CRC_KEY_MSG_SIZE,
};

#define CRC_MOUSE_NULL -1
enum CRC_MOUSE_MSG
{
    CRC_MOUSE_MSG_LBTN = 0,
    CRC_MOUSE_MSG_RBTN,
    CRC_MOUSE_MSG_MBTN,
    CRC_MOUSE_MSG_WHEEL,
    CRC_MOUSE_MSG_XBTN1,
    CRC_MOUSE_MSG_XBTN2,
    CRC_MOUSE_MSG_SIZE,
};

// Rendering

// Pair hash
// std::pair用のハッシュ関数を定義
struct CRC_PAIR_HASH {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& p) const {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ hash2; // Combine hash values
    }
};


// Component types.
enum CRC_COMPONENT_TYPE
{
    CRC_COMPONENT_TYPE_NULL = 0,
    CRC_COMPONENT_TYPE_OBJECT,
    CRC_COMPONENT_TYPE_UTILITY,
    CRC_COMPONENT_TYPE_UI,
    CRC_COMPONENT_TYPE_SIZE,
};

// Debug Output
static void CRCDebugOutput(std::string filePath, std::string func, int line, std::string element)
{
    std::string file = filePath.substr(filePath.find_last_of("\\") + 1);
    std::string output = "[CRC::FILE] " + file + " [LINE]" + std::to_string(line) + " [FUNC]" + func + " : " + element + "\n";
    OutputDebugStringA(output.c_str());
}

static void CRCErrorOutput(std::string filePath, std::string func, int line, std::string element)
{
    std::string file = filePath.substr(filePath.find_last_of("\\") + 1);
    std::string output = "ERROR [CRC::FILE] " + file + " [LINE]" + std::to_string(line) + " [FUNC]" + func + " : " + element + "\n";
    OutputDebugStringA(output.c_str());
}

static void CRCErrorMsgBox(std::string filePath, std::string func, int line, std::string element)
{
    std::string file = filePath.substr(filePath.find_last_of("\\") + 1);
    std::string output = "[CRC::FILE] " + file + " [LINE]" + std::to_string(line) + "\n[FUNC]" + func + " : " + element + "\n";
    MessageBoxA(NULL, output.c_str(), "Error", MB_OK);
}
