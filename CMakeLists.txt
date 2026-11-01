cmake_minimum_required(VERSION 3.25)

project(RocketLeague C)

add_executable(RocketLeague
    src/main.c
)

set_target_properties(RocketLeague PROPERTIES
    C_STANDARD 17
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS ON
)

target_compile_options(RocketLeague PRIVATE
    -Wall
    -Wextra
)

xsdk_add_xex(RocketLeague)

xsdk_add_shaders(RocketLeague
    src/cube.vert
    src/cube.frag
)
