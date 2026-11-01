
import argparse
import os
import struct
import sys

import numpy as np
import trimesh

MAX_VERTICES = 60000


def get_colors(mesh):
    count = len(mesh.vertices)

    try:
        colors = np.asarray(mesh.visual.vertex_colors)

        if len(colors) == count:
            colors = colors[:, :3].astype(np.float32) / 255.0
            return colors
    except Exception:
        pass

    color = np.array([0.65, 0.65, 0.65], dtype=np.float32)

    try:
        material = mesh.visual.material
        factor = np.asarray(
            material.baseColorFactor[:3],
            dtype=np.float32
        )

        if factor.max() > 1.0:
            factor /= 255.0

        color = factor
    except Exception:
        pass

    return np.tile(color, (count, 1))


def write_chunk(file, vertices, indices):
    if not indices:
        return 0

    vertex_data = np.asarray(
        vertices,
        dtype="<f4"
    )

    index_data = np.asarray(
        indices,
        dtype="<u2"
    )

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

            colors = get_colors(mesh)

            uv = getattr(mesh.visual, "uv", None)

            if uv is None:
                uv = np.zeros(
                    (len(positions), 2),
                    dtype=np.float32
                )
            else:
                uv = np.asarray(uv, dtype=np.float32)

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

        file.seek(8)
        file.write(struct.pack("<I", chunk_count))

    print("Conversao concluida!")
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
    
