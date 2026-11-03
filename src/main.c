#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "RocketLeague_shaders.h"
#include "xsdk/gpu.h"
#include "xsdk/kernel.h"
#include "xsdk/log.h"
#include "xsdk/present.h"
#include "xsdk/shader.h"
#include "xsdk/splash.h"
#include "xsdk/texture.h"

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
    TEXTURE_SIZE = 2,

    MAX_STADIUM_CHUNKS = 3000,
    MAX_STADIUM_VERTICES_PER_CHUNK = 60000,
    MAX_DRAW_INDICES = 65532
};

#define STADIUM_PATH "game:\\arena\\champions_field.rlar"

typedef struct {
    float m[16];
} Matrix;

typedef struct {
    uint32_t vertex_offset;
    uint32_t index_offset;
    uint32_t vertex_count;
    uint32_t index_count;
} StadiumChunk;

typedef struct {
    XsdkShader vertex;
    XsdkShader pixel;
    XsdkTexture texture;
    XsdkSamplerState sampler;
    XsdkDepthStencilState depth;

    volatile uint32_t *vertices;
    volatile uint32_t *indices;

    volatile uint32_t *stadium_vertices;
    volatile uint32_t *stadium_indices;

    StadiumChunk *stadium_chunks;
    uint32_t stadium_chunk_count;
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

static uint32_t read_u32_le(const uint8_t *p)
{
    return (uint32_t)p[0]
        | ((uint32_t)p[1] << 8)
        | ((uint32_t)p[2] << 16)
        | ((uint32_t)p[3] << 24);
}

static uint16_t read_u16_le(const uint8_t *p)
{
    return (uint16_t)(
        (uint16_t)p[0] |
        ((uint16_t)p[1] << 8)
    );
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

static HANDLE open_stadium(void)
{
    const char path[] = STADIUM_PATH;

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
        xsdk_log_write(
            "ARENA: erro ao abrir arquivo: 0x%08X",
            (uint32_t)status
        );
        return 0;
    }

    return file;
}

static bool skip_file_bytes(HANDLE file, uint32_t bytes)
{
    uint8_t buffer[4096];

    while (bytes > 0) {
        uint32_t amount = bytes;

        if (amount > sizeof(buffer))
            amount = sizeof(buffer);

        if (!read_exact(file, buffer, amount))
            return false;

        bytes -= amount;
    }

    return true;
}

static void write_gpu_words(
    volatile uint32_t *destination,
    const uint8_t *source,
    uint32_t word_count
)
{
    for (uint32_t i = 0; i < word_count; i++) {
        destination[i] = read_u32_le(source + i * 4);
    }
}

static void write_stadium_indices(
    volatile uint32_t *destination,
    const uint8_t *source,
    uint32_t index_count
)
{
    uint32_t word_count = (index_count + 1) / 2;

    for (uint32_t i = 0; i < word_count; i++) {
        uint32_t index = i * 2;

        uint32_t first = read_u16_le(
            source + index * 2
        );

        uint32_t second = 0;

        if (index + 1 < index_count) {
            second = read_u16_le(
                source + (index + 1) * 2
            );
        }

        destination[i] = (first << 16) | second;
    }
}

static bool load_stadium(Renderer *renderer)
{
    HANDLE file = open_stadium();

    if (!file)
        return false;

    uint8_t header[12];

    if (!read_exact(file, header, sizeof(header)) ||
        header[0] != 'R' ||
        header[1] != 'L' ||
        header[2] != 'A' ||
        header[3] != 'R' ||
        read_u32_le(&header[4]) != 2) {

        xsdk_log_write("ARENA: cabecalho RLAR invalido");
        NtClose(file);
        return false;
    }

    uint32_t chunk_count = read_u32_le(&header[8]);

    if (chunk_count == 0 ||
        chunk_count > MAX_STADIUM_CHUNKS) {

        xsdk_log_write(
            "ARENA: quantidade de blocos invalida: %u",
            chunk_count
        );

        NtClose(file);
        return false;
    }

    uint32_t total_vertices = 0;
    uint32_t total_indices = 0;
    bool valid = true;

    /* Primeira leitura: validar e calcular a memoria necessária. */
    for (uint32_t i = 0; i < chunk_count; i++) {
        uint8_t chunk_header[8];

        if (!read_exact(file, chunk_header, 8)) {
            valid = false;
            break;
        }

        uint32_t vertex_count = read_u32_le(chunk_header);
        uint32_t index_count = read_u32_le(&chunk_header[4]);

        if (vertex_count == 0 ||
            vertex_count > MAX_STADIUM_VERTICES_PER_CHUNK ||
            index_count == 0 ||
            index_count % 3 != 0 ||
            total_vertices > UINT32_MAX - vertex_count ||
            total_indices > UINT32_MAX - index_count) {

            valid = false;
            break;
        }

        if (!skip_file_bytes(file, vertex_count * 32u) ||
            !skip_file_bytes(file, index_count * 2u)) {

            valid = false;
            break;
        }

        total_vertices += vertex_count;
        total_indices += index_count;
    }

    NtClose(file);

    if (!valid || total_vertices == 0 || total_indices == 0) {
        xsdk_log_write("ARENA: falha na validacao dos blocos");
        return false;
    }

    uint32_t vertex_bytes = total_vertices * 32u;
    uint32_t index_bytes = total_indices * 2u;
    uint32_t index_allocation = (index_bytes + 3u) & ~3u;

    if (vertex_bytes / 32u != total_vertices ||
        index_bytes / 2u != total_indices) {

        xsdk_log_write("ARENA: tamanho excede o limite");
        return false;
    }

    renderer->stadium_vertices = xsdk_gpu_alloc(vertex_bytes);
    renderer->stadium_indices = xsdk_gpu_alloc(index_allocation);
    renderer->stadium_chunks = calloc(
        chunk_count,
        sizeof(StadiumChunk)
    );

    uint8_t *vertex_raw = malloc(
        MAX_STADIUM_VERTICES_PER_CHUNK * 32u
    );

    uint8_t *index_raw = malloc(index_bytes);

    if (!renderer->stadium_vertices ||
        !renderer->stadium_indices ||
        !renderer->stadium_chunks ||
        !vertex_raw ||
        !index_raw) {

        xsdk_log_write("ARENA: memoria insuficiente");

        free(vertex_raw);
        free(index_raw);
        return false;
    }

    file = open_stadium();

    if (!file) {
        free(vertex_raw);
        free(index_raw);
        return false;
    }

    if (!read_exact(file, header, sizeof(header)) ||
        header[0] != 'R' ||
        header[1] != 'L' ||
        header[2] != 'A' ||
        header[3] != 'R' ||
        read_u32_le(&header[4]) != 2 ||
        read_u32_le(&header[8]) != chunk_count) {

        xsdk_log_write("ARENA: arquivo mudou durante carregamento");

        NtClose(file);
        free(vertex_raw);
        free(index_raw);
        return false;
    }

    uint32_t vertex_offset = 0;
    uint32_t index_offset = 0;

    for (uint32_t i = 0; i < chunk_count; i++) {
        uint8_t chunk_header[8];

        if (!read_exact(file, chunk_header, 8)) {
            valid = false;
            break;
        }

        uint32_t vertex_count = read_u32_le(chunk_header);
        uint32_t index_count = read_u32_le(&chunk_header[4]);

        if (vertex_count == 0 ||
            vertex_count > MAX_STADIUM_VERTICES_PER_CHUNK ||
            index_count == 0 ||
            index_count % 3 != 0 ||
            vertex_offset + vertex_count * 32u > vertex_bytes ||
            index_offset + index_count * 2u > index_bytes) {

            valid = false;
            break;
        }

        uint32_t current_vertex_bytes = vertex_count * 32u;
        uint32_t current_index_bytes = index_count * 2u;

        if (!read_exact(file, vertex_raw, current_vertex_bytes) ||
            !read_exact(
                file,
                index_raw + index_offset,
                current_index_bytes
            )) {

            valid = false;
            break;
        }

        renderer->stadium_chunks[i].vertex_offset = vertex_offset;
        renderer->stadium_chunks[i].index_offset = index_offset;
        renderer->stadium_chunks[i].vertex_count = vertex_count;
        renderer->stadium_chunks[i].index_count = index_count;

        write_gpu_words(
            renderer->stadium_vertices + vertex_offset / 4u,
            vertex_raw,
            current_vertex_bytes / 4u
        );

        vertex_offset += current_vertex_bytes;
        index_offset += current_index_bytes;
    }

    NtClose(file);

    if (!valid ||
        vertex_offset != vertex_bytes ||
        index_offset != index_bytes) {

        xsdk_log_write("ARENA: erro durante leitura dos dados");

        free(vertex_raw);
        free(index_raw);
        return false;
    }

    write_stadium_indices(
        renderer->stadium_indices,
        index_raw,
        total_indices
    );

    xsdk_gpu_invalidate(
        xsdk_gpu_address(renderer->stadium_vertices),
        vertex_bytes
    );

    xsdk_gpu_invalidate(
        xsdk_gpu_address(renderer->stadium_indices),
        index_allocation
    );

    renderer->stadium_chunk_count = chunk_count;

    free(vertex_raw);
    free(index_raw);

    xsdk_log_write("ARENA: estadio carregado!");
    xsdk_log_write("ARENA: blocos: %u", chunk_count);
    xsdk_log_write("ARENA: vertices: %u", total_vertices);
    xsdk_log_write("ARENA: indices: %u", total_indices);

    return true;
}

static void write_gpu(
    volatile uint32_t *memory,
    const void *data,
    uint32_t size
)
{
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
        {-0.5f,-0.5f, 0.5f, 1,0,0, 0,1},
        { 0.5f,-0.5f, 0.5f, 1,0,0, 1,1},
        {-0.5f, 0.5f, 0.5f, 1,0,0, 0,0},
        { 0.5f, 0.5f, 0.5f, 1,0,0, 1,0},

        { 0.5f,-0.5f,-0.5f, 0,0,1, 0,1},
        {-0.5f,-0.5f,-0.5f, 0,0,1, 1,1},
        { 0.5f, 0.5f,-0.5f, 0,0,1, 0,0},
        {-0.5f, 0.5f,-0.5f, 0,0,1, 1,0},

        { 0.5f,-0.5f, 0.5f, 0,1,0, 0,1},
        { 0.5f,-0.5f,-0.5f, 0,1,0, 1,1},
        { 0.5f, 0.5f, 0.5f, 0,1,0, 0,0},
        { 0.5f, 0.5f,-0.5f, 0,1,0, 1,0},

        {-0.5f,-0.5f,-0.5f, 1,1,0, 0,1},
        {-0.5f,-0.5f, 0.5f, 1,1,0, 1,1},
        {-0.5f, 0.5f,-0.5f, 1,1,0, 0,0},
        {-0.5f, 0.5f, 0.5f, 1,1,0, 1,0},

        {-0.5f, 0.5f, 0.5f, 1,0,1, 0,1},
        { 0.5f, 0.5f, 0.5f, 1,0,1, 1,1},
        {-0.5f, 0.5f,-0.5f, 1,0,1, 0,0},
        { 0.5f, 0.5f,-0.5f, 1,0,1, 1,0},

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
        {0xFFFFFFFF, 0xFFFFFFFF},
        {0xFFFFFFFF, 0xFFFFFFFF}
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

    xsdk_texture_sampler_state(
        &renderer->sampler,
        &sampler
    );

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

    xsdk_gpu_depth_stencil_state(
        &renderer->depth,
        &depth
    );

    return create_mesh(renderer) &&
           create_texture(renderer);
}

static Matrix multiply_matrix(
    const Matrix *a,
    const Matrix *b
)
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
    const float far_plane = 100.0f;

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
        0, 0, -3.0f, 1
    }};

    const float c = cosf(angle);
    const float s = sinf(angle);

    const Matrix rotation_y = {{
         c, 0, -s, 0,
         0, 1,  0, 0,
         s, 0,  c, 0,
         0, 0,  0, 1
    }};

    const float tilt = -0.42f;
    const float ct = cosf(tilt);
    const float st = sinf(tilt);

    const Matrix rotation_x = {{
        1,  0,  0, 0,
        0, ct, st, 0,
        0,-st, ct, 0,
        0,  0,  0, 1
    }};

    /*
     * Escala inicial para enquadrar o estádio.
     * O deslocamento vertical centraliza aproximadamente
     * os limites encontrados na leitura do arquivo.
     */
    const Matrix model = {{
        0.035f, 0,      0,      0,
        0,      0.035f, 0,      0,
        0,      0,      0.035f, 0,
        0,          0.14194f,  0,          1
    }};

    Matrix transformed = multiply_matrix(&camera, &rotation_y);
    transformed = multiply_matrix(&transformed, &rotation_x);
    transformed = multiply_matrix(&transformed, &model);

    return multiply_matrix(&projection, &transformed);
}

static void draw_cube(const Renderer *renderer)
{
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
}

static void draw_stadium(const Renderer *renderer)
{
    for (uint32_t i = 0; i < renderer->stadium_chunk_count; i++) {
        const StadiumChunk *chunk = &renderer->stadium_chunks[i];

        uint32_t vertex_address =
            xsdk_gpu_address(renderer->stadium_vertices) +
            chunk->vertex_offset;

        xsdk_gpu_set_vertices(
            0,
            vertex_address,
            chunk->vertex_count * 32u
        );

        uint32_t drawn = 0;

        while (drawn < chunk->index_count) {
            uint32_t amount = chunk->index_count - drawn;

            if (amount > MAX_DRAW_INDICES)
                amount = MAX_DRAW_INDICES;

            amount -= amount % 3u;

            if (amount == 0)
                break;

            uint32_t index_address =
                xsdk_gpu_address(renderer->stadium_indices) +
                chunk->index_offset +
                drawn * 2u;

            xsdk_gpu_draw_indexed(
                XSDK_PRIMITIVE_TRIANGLE_LIST,
                index_address,
                amount,
                false
            );

            drawn += amount;
        }
    }
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

    if (renderer->stadium_chunk_count > 0)
        draw_stadium(renderer);
    else
        draw_cube(renderer);

    xsdk_gpu_resolve(
        address,
        width,
        height,
        CLEAR_ARGB
    );
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

            angle += 0.025f *
                read_stick(input.gamepad.thumb_lx);
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

    if (renderer->stadium_indices)
        xsdk_gpu_free(renderer->stadium_indices);

    if (renderer->stadium_vertices)
        xsdk_gpu_free(renderer->stadium_vertices);

    free(renderer->stadium_chunks);

    if (renderer->indices)
        xsdk_gpu_free(renderer->indices);

    if (renderer->vertices)
        xsdk_gpu_free(renderer->vertices);

    xsdk_shader_free(&renderer->pixel);
    xsdk_shader_free(&renderer->vertex);

    memset(renderer, 0, sizeof(*renderer));
}

int main(void)
{
    xsdk_log_write("RocketLeague: START");

    NTSTATUS status = xsdk_splash_show(SPLASH_MS);

    if (!NT_SUCCESS(status))
        xsdk_log_write("RocketLeague: splash failed");

    status = xsdk_gpu_open();

    if (!NT_SUCCESS(status)) {
        xsdk_log_write("RocketLeague: GPU open failed");
        return 1;
    }

    static Renderer renderer = {0};

    if (!create_renderer(&renderer)) {
        xsdk_log_write("RocketLeague: renderer creation failed");

        destroy_renderer(&renderer);
        xsdk_gpu_close();

        return 1;
    }

    if (!load_stadium(&renderer)) {
        xsdk_log_write(
            "RocketLeague: estadio nao carregado; usando cubo"
        );
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
