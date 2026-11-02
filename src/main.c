
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "RocketLeague_shaders.h"
#include "xsdk/gpu.h"
#include "xsdk/kernel.h"
#include "xsdk/log.h"
#include "xsdk/present.h"
#include "xsdk/shader.h"
#include "xsdk/splash.h"
#include "xsdk/texture.h"
#include <stdlib.h>

enum {
    SPLASH_MS = 2000,
    BUFFERS = 2,
    SWAP_INTERVAL = 1,
    BEGIN_TIMEOUT_MS = 5000,

    CLEAR_ARGB = 0xFF183060,

    VERTEX_COUNT = 24,
    INDEX_COUNT = 36,
    VERTEX_FLOATS = 8,

    VERTEX_STRIDE = VERTEX_FLOATS * 4,
    VERTEX_BYTES = VERTEX_COUNT * VERTEX_STRIDE,
    INDEX_BYTES = INDEX_COUNT * 2,

    TEXTURE_SIZE = 2
};

typedef struct {
    float m[16];
} Matrix;

typedef struct {
    XsdkShader vertex;
    XsdkShader pixel;
    XsdkTexture texture;
    XsdkSamplerState sampler;
    XsdkDepthStencilState depth;

    volatile uint32_t *vertices;
    volatile uint32_t *indices;
} Renderer;

static const XsdkVertexElement vertex_layout[] = {
    {
        .usage = XSDK_USAGE_POSITION,
        .format = XSDK_VERTEX_FLOAT3,
        .offset_bytes = 0
    },
    {
        .usage = XSDK_USAGE_COLOR,
        .format = XSDK_VERTEX_FLOAT3,
        .offset_bytes = 12
    },
    {
        .usage = XSDK_USAGE_TEXCOORD,
        .format = XSDK_VERTEX_FLOAT2,
        .offset_bytes = 24
    }
};

static void write_gpu(
    volatile uint32_t *memory,
    const void *data,
    uint32_t size
) {
    const uint8_t *source = data;

    for (uint32_t i = 0; i < size / 4; i++) {
        uint32_t word;
        memcpy(&word, source + i * 4, 4);
        memory[i] = word;
    }

    xsdk_gpu_invalidate(
        xsdk_gpu_address(memory),
        size
    );
}

static bool create_mesh(Renderer *renderer)
{
    static const float vertices[VERTEX_COUNT][VERTEX_FLOATS] = {
        /* Frente */
        {-0.5f,-0.5f, 0.5f, 1,0,0, 0,1},
        { 0.5f,-0.5f, 0.5f, 1,0,0, 1,1},
        {-0.5f, 0.5f, 0.5f, 1,0,0, 0,0},
        { 0.5f, 0.5f, 0.5f, 1,0,0, 1,0},

        /* Trás */
        { 0.5f,-0.5f,-0.5f, 0,0,1, 0,1},
        {-0.5f,-0.5f,-0.5f, 0,0,1, 1,1},
        { 0.5f, 0.5f,-0.5f, 0,0,1, 0,0},
        {-0.5f, 0.5f,-0.5f, 0,0,1, 1,0},

        /* Direita */
        { 0.5f,-0.5f, 0.5f, 0,1,0, 0,1},
        { 0.5f,-0.5f,-0.5f, 0,1,0, 1,1},
        { 0.5f, 0.5f, 0.5f, 0,1,0, 0,0},
        { 0.5f, 0.5f,-0.5f, 0,1,0, 1,0},

        /* Esquerda */
        {-0.5f,-0.5f,-0.5f, 1,1,0, 0,1},
        {-0.5f,-0.5f, 0.5f, 1,1,0, 1,1},
        {-0.5f, 0.5f,-0.5f, 1,1,0, 0,0},
        {-0.5f, 0.5f, 0.5f, 1,1,0, 1,0},

        /* Cima */
        {-0.5f, 0.5f, 0.5f, 1,0,1, 0,1},
        { 0.5f, 0.5f, 0.5f, 1,0,1, 1,1},
        {-0.5f, 0.5f,-0.5f, 1,0,1, 0,0},
        { 0.5f, 0.5f,-0.5f, 1,0,1, 1,0},

        /* Baixo */
        {-0.5f,-0.5f,-0.5f, 0,1,1, 0,1},
        { 0.5f,-0.5f,-0.5f, 0,1,1, 1,1},
        {-0.5f,-0.5f, 0.5f, 0,1,1, 0,0},
        { 0.5f,-0.5f, 0.5f, 0,1,1, 1,0}
    };

    static const uint16_t indices[INDEX_COUNT] = {
         0, 1, 2,  2, 1, 3,
         4, 5, 6,  6, 5, 7,
         8, 9,10, 10, 9,11,
        12,13,14, 14,13,15,
        16,17,18, 18,17,19,
        20,21,22, 22,21,23
    };

    renderer->vertices = xsdk_gpu_alloc(VERTEX_BYTES);
    renderer->indices = xsdk_gpu_alloc(INDEX_BYTES);

    if (!renderer->vertices || !renderer->indices) {
        xsdk_log_write("RocketLeague: GPU allocation failed");
        return false;
    }

    write_gpu(renderer->vertices, vertices, VERTEX_BYTES);
    write_gpu(renderer->indices, indices, INDEX_BYTES);

    return true;
}

static bool create_texture(Renderer *renderer)
{
    NTSTATUS status = xsdk_texture_create(
        &renderer->texture,
        XSDK_FORMAT_A8R8G8B8,
        TEXTURE_SIZE,
        TEXTURE_SIZE,
        1,
        true
    );

    if (!NT_SUCCESS(status)) {
        xsdk_log_write("RocketLeague: texture creation failed");
        return false;
    }

    static const uint32_t texels[TEXTURE_SIZE][TEXTURE_SIZE] = {
        {0xFFFFFFFF, 0xFF808080},
        {0xFF808080, 0xFFFFFFFF}
    };

    xsdk_texture_upload(
        &renderer->texture,
        0,
        texels,
        sizeof(texels[0])
    );

    xsdk_gpu_invalidate(
        renderer->texture.address,
        renderer->texture.size_bytes
    );

    const XsdkSampler sampler = {
        .mag_filter = XSDK_FILTER_LINEAR,
        .min_filter = XSDK_FILTER_LINEAR,
        .mip_filter = XSDK_FILTER_POINT,
        .address_u = XSDK_ADDRESS_WRAP,
        .address_v = XSDK_ADDRESS_WRAP
    };

    xsdk_texture_sampler_state(&renderer->sampler, &sampler);

    return true;
}

static bool create_renderer(Renderer *renderer)
{
    NTSTATUS status = xsdk_shader_load(
        &renderer->vertex,
        cube_vert.container,
        cube_vert.container_bytes
    );

    if (NT_SUCCESS(status)) {
        status = xsdk_shader_load(
            &renderer->pixel,
            cube_frag.container,
            cube_frag.container_bytes
        );
    }

    if (NT_SUCCESS(status)) {
        status = xsdk_shader_set_vertex_layout(
            &renderer->vertex,
            vertex_layout,
            sizeof(vertex_layout) / sizeof(vertex_layout[0]),
            (const uint32_t[]){VERTEX_STRIDE}
        );
    }

    if (!NT_SUCCESS(status)) {
        xsdk_log_write(
            "RocketLeague: shader loading failed: 0x%08X",
            (unsigned)status
        );
        return false;
    }

    const XsdkDepthStencil depth = {
        .depth_test = true,
        .depth_write = true,
        .depth_compare = XSDK_COMPARE_LESS_EQUAL
    };

    xsdk_gpu_depth_stencil_state(&renderer->depth, &depth);

    return create_mesh(renderer) && create_texture(renderer);
}

static Matrix multiply_matrix(const Matrix *a, const Matrix *b)
{
    Matrix result;

    for (uint32_t column = 0; column < 4; column++) {
        for (uint32_t row = 0; row < 4; row++) {
            float sum = 0.0f;

            for (uint32_t k = 0; k < 4; k++) {
                sum += a->m[k * 4 + row] *
                       b->m[column * 4 + k];
            }

            result.m[column * 4 + row] = sum;
        }
    }

    return result;
}

static Matrix camera_matrix(float angle, float aspect)
{
    const float fov = 1.0f;
    const float near_plane = 0.1f;
    const float far_plane = 10.0f;

    const float f = 1.0f / tanf(fov / 2.0f);
    const float fx = f / aspect;
    const float fz = far_plane / (near_plane - far_plane);

    const Matrix projection = {{
        fx, 0,  0,              0,
        0,  f,  0,              0,
        0,  0,  fz,            -1,
        0,  0,  near_plane*fz,  0
    }};

    const Matrix camera = {{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, -3, 1
    }};

    const float c = cosf(angle);
    const float s = sinf(angle);

    const Matrix rotation = {{
         c, 0, -s, 0,
         0, 1,  0, 0,
         s, 0,  c, 0,
         0, 0,  0, 1
    }};

    const Matrix transformed = multiply_matrix(&camera, &rotation);

    return multiply_matrix(&projection, &transformed);
}

static void draw_frame(
    const Renderer *renderer,
    const Matrix *matrix,
    uint32_t address,
    uint32_t width,
    uint32_t height
)
{
    xsdk_gpu_set_target(width, height);

    xsdk_gpu_set_depth_stencil(&renderer->depth);

    xsdk_gpu_set_shaders(
        &renderer->vertex,
        &renderer->pixel
    );

    xsdk_built_shader_set_literals(&cube_vert);
    xsdk_built_shader_set_literals(&cube_frag);

    xsdk_gpu_set_vertex_constants(
        CUBE_VERT_PUSH,
        matrix->m,
        CUBE_VERT_PUSH_REGISTERS
    );

    xsdk_gpu_set_texture(
        CUBE_FRAG_SAMPLER_0_0,
        &renderer->texture,
        &renderer->sampler
    );

    xsdk_gpu_set_vertices(
        0,
        xsdk_gpu_address(renderer->vertices),
        VERTEX_BYTES
    );

    xsdk_gpu_draw_indexed(
        XSDK_PRIMITIVE_TRIANGLE_LIST,
        xsdk_gpu_address(renderer->indices),
        INDEX_COUNT,
        false
    );

    xsdk_gpu_resolve(address, width, height, CLEAR_ARGB);
}

static float read_stick(int16_t value)
{
    const int32_t dead_zone = 8000;

    if (value > dead_zone || value < -dead_zone)
        return (float)value / 32768.0f;

    return 0.0f;
}

static void game_loop(
    const Renderer *renderer,
    uint32_t width,
    uint32_t height
)
{
    float angle = 0.0f;

    for (;;) {
        INPUT_STATE input = {0};

        if (XamInputGetState(0, 0, &input) == ERROR_SUCCESS) {
            if (input.gamepad.buttons & INPUT_GAMEPAD_BACK)
                break;

            angle += 0.05f * read_stick(input.gamepad.thumb_lx);
        }

        uint32_t address;

        NTSTATUS status = xsdk_present_begin(
            &address,
            BEGIN_TIMEOUT_MS
        );

        if (!NT_SUCCESS(status)) {
            xsdk_log_write(
                "RocketLeague: present failed: 0x%08X",
                (unsigned)status
            );
            break;
        }

        Matrix matrix = camera_matrix(
            angle,
            (float)width / (float)height
        );

        draw_frame(
            renderer,
            &matrix,
            address,
            width,
            height
        );

        xsdk_present_end();
    }
}

static void destroy_renderer(Renderer *renderer)
{
    xsdk_texture_free(&renderer->texture);

    if (renderer->indices)
        xsdk_gpu_free(renderer->indices);

    if (renderer->vertices)
        xsdk_gpu_free(renderer->vertices);

    xsdk_shader_free(&renderer->pixel);
    xsdk_shader_free(&renderer->vertex);
}

static uint32_t read_u32_le(const uint8_t *p)
{
    return (uint32_t)p[0]
        | ((uint32_t)p[1] << 8)
        | ((uint32_t)p[2] << 16)
        | ((uint32_t)p[3] << 24);
}

static bool read_exact(HANDLE file, void *buffer, uint32_t size)
{
    IO_STATUS_BLOCK io = {0};

    NTSTATUS status = NtReadFile(
        file,
        0,
        NULL,
        NULL,
        &io,
        buffer,
        size,
        NULL
    );

    return NT_SUCCESS(status) && io.information == size;
}

static void check_stadium_file(void)
{
    const char path[] = "game:\\arena\\champions_field.rlar";

    ANSI_STRING name = {
        (uint16_t)(sizeof(path) - 1),
        (uint16_t)(sizeof(path) - 1),
        path
    };

    OBJECT_ATTRIBUTES attributes = {
        OB_DOS_DEVICES,
        &name,
        0
    };

    HANDLE file = 0;
    IO_STATUS_BLOCK io = {0};

    NTSTATUS status = NtCreateFile(
        &file,
        GENERIC_READ | SYNCHRONIZE,
        &attributes,
        &io,
        NULL,
        FILE_ATTRIBUTE_NORMAL,
        FILE_SHARE_READ,
        FILE_OPEN,
        FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT
    );

    if (!NT_SUCCESS(status)) {
        xsdk_log_write(
            "ARENA: erro ao abrir arquivo: 0x%08X\n",
            (uint32_t)status
        );
        return;
    }

    FILE_NETWORK_OPEN_INFORMATION info = {0};

    status = NtQueryInformationFile(
        file,
        &io,
        &info,
        sizeof(info),
        FILE_NETWORK_OPEN_INFORMATION_CLASS
    );

    if (!NT_SUCCESS(status)) {
        xsdk_log_write("ARENA: erro ao consultar tamanho: 0x%08X\n",
                       (uint32_t)status);
        NtClose(file);
        return;
    }

    xsdk_log_write("ARENA: arquivo encontrado, tamanho: %llu bytes\n",
                   (unsigned long long)info.end_of_file);

    uint8_t header[12] = {0};

    status = NtReadFile(
        file,
        0,
        NULL,
        NULL,
        &io,
        header,
        sizeof(header),
        NULL
    );

    NtClose(file);

    if (!NT_SUCCESS(status) || io.information != sizeof(header)) {
        xsdk_log_write("ARENA: erro ao ler cabecalho: 0x%08X\n",
                       (uint32_t)status);
        return;
    }

    uint32_t version = read_u32_le(&header[4]);
    uint32_t chunks = read_u32_le(&header[8]);

    if (header[0] != 'R' || header[1] != 'L' ||
        header[2] != 'A' || header[3] != 'R') {
        xsdk_log_write("ARENA: arquivo com magic invalido\n");
        return;
    }

    xsdk_log_write("ARENA: versao %u, blocos %u\n",
                   version, chunks);

    if (version == 2 && chunks > 0) {
        xsdk_log_write("ARENA: arquivo RLAR valido!\n");
    } else {
        xsdk_log_write("ARENA: versao ou quantidade de blocos invalida\n");
    }
}

static void scan_stadium_file(void)
{
    const char path[] = "game:\\arena\\champions_field.rlar";

    ANSI_STRING name = {
        (uint16_t)(sizeof(path) - 1),
        (uint16_t)(sizeof(path) - 1),
        path
    };

    OBJECT_ATTRIBUTES attributes = {
        OB_DOS_DEVICES,
        &name,
        0
    };

    HANDLE file = 0;
    IO_STATUS_BLOCK io = {0};

    NTSTATUS status = NtCreateFile(
        &file,
        GENERIC_READ | SYNCHRONIZE,
        &attributes,
        &io,
        NULL,
        FILE_ATTRIBUTE_NORMAL,
        FILE_SHARE_READ,
        FILE_OPEN,
        FILE_NON_DIRECTORY_FILE |
        FILE_SYNCHRONOUS_IO_NONALERT
    );

    if (!NT_SUCCESS(status)) {
        xsdk_log_write("SCAN: erro ao abrir estadio");
        return;
    }

    uint8_t header[12];

    if (!read_exact(file, header, 12)) {
        xsdk_log_write("SCAN: erro no cabecalho");
        NtClose(file);
        return;
    }

    if (header[0] != 'R' || header[1] != 'L' ||
        header[2] != 'A' || header[3] != 'R' ||
        read_u32_le(&header[4]) != 2) {
        xsdk_log_write("SCAN: formato invalido");
        NtClose(file);
        return;
    }

    uint32_t chunks = read_u32_le(&header[8]);

    uint8_t *vertices = malloc(60000 * 32);

    if (!vertices) {
        xsdk_log_write("SCAN: memoria insuficiente");
        NtClose(file);
        return;
    }

    uint8_t skip_buffer[4096];

    float min_x =  1.0e30f;
    float min_y =  1.0e30f;
    float min_z =  1.0e30f;

    float max_x = -1.0e30f;
    float max_y = -1.0e30f;
    float max_z = -1.0e30f;

    uint32_t total_vertices = 0;
    uint32_t total_indices = 0;
    bool valid = true;

    for (uint32_t chunk = 0; chunk < chunks; chunk++) {
        uint8_t chunk_header[8];

        if (!read_exact(file, chunk_header, 8)) {
            valid = false;
            break;
        }

        uint32_t vertex_count = read_u32_le(chunk_header);
        uint32_t index_count = read_u32_le(&chunk_header[4]);

        if (vertex_count == 0 || vertex_count > 60000 ||
            index_count == 0 || index_count % 3 != 0) {
            valid = false;
            break;
        }

        uint32_t vertex_bytes = vertex_count * 32;
        uint32_t index_bytes = index_count * 2;

        if (!read_exact(file, vertices, vertex_bytes)) {
            valid = false;
            break;
        }

        for (uint32_t i = 0; i < vertex_count; i++) {
            uint8_t *v = &vertices[i * 32];

            uint32_t bx = read_u32_le(&v[0]);
            uint32_t by = read_u32_le(&v[4]);
            uint32_t bz = read_u32_le(&v[8]);

            float x, y, z;

            memcpy(&x, &bx, 4);
            memcpy(&y, &by, 4);
            memcpy(&z, &bz, 4);

            if (x < min_x) min_x = x;
            if (y < min_y) min_y = y;
            if (z < min_z) min_z = z;

            if (x > max_x) max_x = x;
            if (y > max_y) max_y = y;
            if (z > max_z) max_z = z;
        }

        uint32_t remaining = index_bytes;

        while (remaining > 0) {
            uint32_t amount = remaining;

            if (amount > sizeof(skip_buffer))
                amount = sizeof(skip_buffer);

            if (!read_exact(file, skip_buffer, amount)) {
                valid = false;
                break;
            }

            remaining -= amount;
        }

        if (!valid)
            break;

        total_vertices += vertex_count;
        total_indices += index_count;
    }

    if (valid) {
        xsdk_log_write("SCAN: blocos lidos: %u", chunks);
        xsdk_log_write("SCAN: vertices: %u", total_vertices);
        xsdk_log_write("SCAN: indices: %u", total_indices);

        xsdk_log_write(
            "SCAN: X %d ate %d",
            (int)(min_x * 1000.0f),
            (int)(max_x * 1000.0f)
        );

        xsdk_log_write(
            "SCAN: Y %d ate %d",
            (int)(min_y * 1000.0f),
            (int)(max_y * 1000.0f)
        );

        xsdk_log_write(
            "SCAN: Z %d ate %d",
            (int)(min_z * 1000.0f),
            (int)(max_z * 1000.0f)
        );

        xsdk_log_write("SCAN: leitura concluida!");
    } else {
        xsdk_log_write("SCAN: erro ao percorrer os blocos");
    }

    free(vertices);
    NtClose(file);
}

int main(void)
{
    xsdk_log_write("RocketLeague: START");

    NTSTATUS status = xsdk_splash_show(SPLASH_MS);

    if (!NT_SUCCESS(status))
        xsdk_log_write("RocketLeague: splash failed");

    status = xsdk_gpu_open();

             check_stadium_file();
             scan_stadium_file();

    if (!NT_SUCCESS(status)) {
        xsdk_log_write("RocketLeague: GPU open failed");
        return 1;
    }

    static Renderer renderer = {0};

    if (!create_renderer(&renderer)) {
        xsdk_log_write("RocketLeague: renderer creation failed");
        xsdk_gpu_close();
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

    if (!NT_SUCCESS(status)) {
        xsdk_log_write("RocketLeague: display open failed");
        destroy_renderer(&renderer);
        xsdk_gpu_close();
        return 1;
    }

    xsdk_log_write(
        "RocketLeague: display ready %ux%u",
        (unsigned)width,
        (unsigned)height
    );

    game_loop(&renderer, width, height);

    xsdk_present_close();
    destroy_renderer(&renderer);

    status = xsdk_gpu_close();

    if (!NT_SUCCESS(status))
        xsdk_log_write("RocketLeague: GPU close failed");

    xsdk_log_write("RocketLeague: EXIT");

    return 0;
}
