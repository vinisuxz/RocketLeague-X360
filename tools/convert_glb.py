import argparse
import os
import struct
import sys

import numpy as np
import trimesh

MAX_VERTICES = 60000


def get_material(mesh):
    try:
        return mesh.visual.material
    except Exception:
        return None


def get_material_factor(material):
    factor = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    if material is None:
        return factor

    try:
        value = getattr(material, "baseColorFactor", None)

        if value is not None:
            value = np.asarray(value, dtype=np.float32)[:3]

            if value.max() > 1.0:
                value /= 255.0

            return np.clip(value, 0.0, 1.0)
    except Exception:
        pass

    try:
        value = np.asarray(material.diffuse, dtype=np.float32)[:3]

        if value.max() > 1.0:
            value /= 255.0

        return np.clip(value, 0.0, 1.0)
    except Exception:
        pass

    return factor


def get_texture_image(material):
    if material is None:
        return None

    for attribute in ("baseColorTexture", "image"):
        try:
            image = getattr(material, attribute, None)

            if image is not None:
                image = np.asarray(image.convert("RGB"), dtype=np.float32)
                return image / 255.0
        except Exception:
            pass

    return None


def sample_texture(texture, uv):
    height, width = texture.shape[:2]

    uv = np.asarray(uv, dtype=np.float32)

    u = np.mod(uv[:, 0], 1.0)
    v = np.mod(uv[:, 1], 1.0)

    x = u * (width - 1)
    y = (1.0 - v) * (height - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)

    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]

    c00 = texture[y0, x0]
    c10 = texture[y0, x1]
    c01 = texture[y1, x0]
    c11 = texture[y1, x1]

    top = c00 * (1.0 - fx) + c10 * fx
    bottom = c01 * (1.0 - fx) + c11 * fx

    return top * (1.0 - fy) + bottom * fy


def get_colors(mesh, uv):
    count = len(mesh.vertices)
    material = get_material(mesh)
    factor = get_material_factor(material)

    colors = None

    if getattr(mesh.visual, "kind", None) == "vertex":
        try:
            vertex_colors = np.asarray(
                mesh.visual.vertex_colors,
                dtype=np.float32
            )

            if len(vertex_colors) == count:
                colors = vertex_colors[:, :3]

                if colors.max() > 1.0:
                    colors /= 255.0
        except Exception:
            pass

    if colors is None:
        colors = np.ones((count, 3), dtype=np.float32)

    texture = get_texture_image(material)

    if texture is not None and uv is not None:
        try:
            texture_colors = sample_texture(texture, uv)
            colors *= texture_colors
            print("Textura aplicada aos vertices:", texture.shape)
        except Exception as error:
            print("Aviso: falha ao aplicar textura:", error)

    colors *= factor

    return np.clip(colors, 0.0, 1.0)


def get_uv(mesh):
    try:
        uv = getattr(mesh.visual, "uv", None)

        if uv is not None:
            uv = np.asarray(uv, dtype=np.float32)

            if len(uv) == len(mesh.vertices):
                return uv[:, :2]
    except Exception:
        pass

    return np.zeros(
        (len(mesh.vertices), 2),
        dtype=np.float32
    )


def write_chunk(file, vertices, indices):
    if not indices:
        return 0

    vertex_data = np.asarray(vertices, dtype="<f4")
    index_data = np.asarray(indices, dtype="<u2")

    file.write(struct.pack(
        "<II",
        len(vertex_data),
        len(index_data)
    ))

    file.write(vertex_data.tobytes())
    file.write(index_data.tobytes())

    return 1


def convert(source, output):
    print("Abrindo Champions Field...")

    scene = trimesh.load(
        source,
        force="scene",
        process=False
    )

    os.makedirs(
        os.path.dirname(output) or ".",
        exist_ok=True
    )

    chunk_count = 0
    mesh_count = 0

    with open(output, "wb") as file:
        file.write(b"RLAR")
        file.write(struct.pack("<I", 2))
        file.write(struct.pack("<I", 0))

        for node in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node]

            mesh = scene.geometry[geometry_name].copy()

            if not isinstance(mesh, trimesh.Trimesh):
                continue

            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                continue

            mesh.apply_transform(transform)

            positions = np.asarray(
                mesh.vertices,
                dtype=np.float32
            )

            uv = get_uv(mesh)
            colors = get_colors(mesh, uv)

            vertices = np.concatenate(
                [positions, colors, uv],
                axis=1
            )

            local_vertices = []
            local_indices = []
            vertex_map = {}

            for face in mesh.faces:
                new_vertices = sum(
                    1 for index in face
                    if int(index) not in vertex_map
                )

                if (
                    len(vertex_map) + new_vertices > MAX_VERTICES
                    and local_indices
                ):
                    chunk_count += write_chunk(
                        file,
                        local_vertices,
                        local_indices
                    )

                    local_vertices = []
                    local_indices = []
                    vertex_map = {}

                for original_index in face:
                    original_index = int(original_index)

                    if original_index not in vertex_map:
                        local_index = len(local_vertices)

                        vertex_map[original_index] = local_index

                        local_vertices.append(
                            vertices[original_index]
                        )

                    local_indices.append(
                        vertex_map[original_index]
                    )

            chunk_count += write_chunk(
                file,
                local_vertices,
                local_indices
            )

            mesh_count += 1

            print(
                "Malha convertida:",
                geometry_name,
                "vertices:",
                len(mesh.vertices)
            )

        file.seek(8)
        file.write(struct.pack("<I", chunk_count))

    print("Conversao concluida!")
    print("Malhas:", mesh_count)
    print("Blocos:", chunk_count)
    print("Arquivo:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("source")
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    try:
        convert(args.source, args.out)
    except Exception as error:
        print("ERRO:", error)
        sys.exit(1)
