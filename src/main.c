
#include <stdbool.h>
#include <stdint.h>

#include "xsdk/gpu.h"
#include "xsdk/kernel.h"
#include "xsdk/log.h"
#include "xsdk/present.h"
#include "xsdk/splash.h"

enum {
    SPLASH_MS = 2000,
    BUFFERS = 2,
    SWAP_INTERVAL = 1,
    BEGIN_TIMEOUT_MS = 5000,
    CLEAR_ARGB = 0xFF183060
};

static void game_loop(uint32_t width, uint32_t height)
{
    xsdk_log_write("RocketLeague: entering game loop");

    for (;;)
    {
        INPUT_STATE input = {0};

        uint32_t result = XamInputGetState(0, 0, &input);

        if (result != ERROR_SUCCESS)
            input = (INPUT_STATE){0};

        if (input.gamepad.buttons & INPUT_GAMEPAD_BACK)
            break;

        uint32_t address;

        NTSTATUS status =
            xsdk_present_begin(&address, BEGIN_TIMEOUT_MS);

        if (!NT_SUCCESS(status))
        {
            xsdk_log_write(
                "present_begin failed: 0x%08X",
                (unsigned)status
            );
            break;
        }

        xsdk_gpu_resolve(
            address,
            width,
            height,
            CLEAR_ARGB
        );

        xsdk_present_end();
    }
}

int main(void)
{
    xsdk_log_write("RocketLeague: START");

    NTSTATUS status = xsdk_splash_show(SPLASH_MS);

    if (!NT_SUCCESS(status))
        xsdk_log_write("Splash failed: 0x%08X", (unsigned)status);

    status = xsdk_gpu_open();

    if (!NT_SUCCESS(status))
    {
        xsdk_log_write("GPU open failed: 0x%08X", (unsigned)status);
        return 1;
    }

    uint32_t width = 0;
    uint32_t height = 0;

    status = xsdk_present_open(
        XSDK_DISPLAY_720P,
        BUFFERS,
        SWAP_INTERVAL,
        &width,
        &height
    );

    if (!NT_SUCCESS(status))
    {
        xsdk_log_write("Display open failed: 0x%08X", (unsigned)status);
        return 1;
    }

    xsdk_log_write(
        "Display ready: %ux%u",
        (unsigned)width,
        (unsigned)height
    );

    game_loop(width, height);

    xsdk_present_close();

    status = xsdk_gpu_close();

    if (!NT_SUCCESS(status))
        xsdk_log_write("GPU close failed: 0x%08X", (unsigned)status);

    xsdk_log_write("RocketLeague: EXIT");

    return 0;
}
