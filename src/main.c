#include <stdint.h>
#include "xsdk/gpu.h"
#include "xsdk/kernel.h"
#include "xsdk/log.h"
#include "xsdk/present.h"
#include "xsdk/splash.h"

#define CLEAR_ARGB 0xFF87CEEB

static void game_loop(uint32_t width, uint32_t height)
{
    while (1)
    {
        INPUT_STATE input = {0};

        if (XamInputGetState(0, 0, &input) == ERROR_SUCCESS &&
            (input.gamepad.buttons & INPUT_GAMEPAD_BACK))
            break;

        uint32_t address;

        NTSTATUS status = xsdk_present_begin(&address, 5000);

        if (!NT_SUCCESS(status))
            break;

        // Futuramente: renderização da arena e do carro.
        xsdk_gpu_resolve(address, width, height, CLEAR_ARGB);

        xsdk_present_end();
    }
}

int main(void)
{
    xsdk_log_write("RocketLeague X360: starting");

    xsdk_splash_show(1500);

    NTSTATUS status = xsdk_gpu_open();

    if (!NT_SUCCESS(status))
        return 1;

    uint32_t width = 0;
    uint32_t height = 0;

    status = xsdk_present_open(
        XSDK_DISPLAY_720P,
        2,
        1,
        &width,
        &height
    );

    if (!NT_SUCCESS(status))
    {
        xsdk_gpu_close();
        return 1;
    }

    game_loop(width, height);

    xsdk_present_close();
    xsdk_gpu_close();

    xsdk_log_write("RocketLeague X360: finished");

    return 0;
}
