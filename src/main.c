#include <stdio.h>
#include <xtl.h>

int main(void)
{
    XVideoSetMode(
        1280,
        720,
        1,
        REFRESH_DEFAULT
    );

    for (;;)
    {
        Sleep(16);
    }

    return 0;
}
