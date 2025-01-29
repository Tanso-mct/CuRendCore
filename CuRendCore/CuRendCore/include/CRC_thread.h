#pragma once

#include <thread>
#include <atomic>

#include "CRC_config.h"
#include "CRC_interface.h"

struct CRCThread
{
    std::thread thread_;
    std::atomic<bool> isRunning_ = false;
    std::atomic<int> isCmd = CRC::THRD_CMD_READY;
    std::atomic<int> didCmd = CRC::THRD_CMD_READY;
    std::atomic<int> error = CRC::THRD_ERROR_OK;
};  