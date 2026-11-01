#!/usr/bin/env python3
import argparse
import os
import struct
import sys

import numpy as np
import trimesh


def convert(source, output):
    print("Abrindo modelo GLB...")

    scene = trimesh.load(
        source,
        force="scene",
        process=False
    )

    meshes = []

    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()

        if not isinstance(mesh, trimesh.Trimesh):
            continue

        mesh.apply_transform(transform)

        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue

        positions = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

        uv = getattr(mesh.visual, "uv", None)

        if uv is None:
            uv = np.zeros((len(positions), 2), dtype=np.float32)
        else:
            uv = np.asarray(uv, dtype=np.float32)

        vertices = np.zeros(
            (len(positions), 8),
            dtype=np.float32
        )

        vertices[:, 0:3] = positions
        vertices[:, 3:6] = normals
        vertices[:, 6:8] = uv

        indices = np.asarray(
            mesh.faces,
            dtype=np.uint32
        ).flatten()

        meshes.append((vertices, indices))

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    with open(output, "wb") as file:
        file.write(b"RLAR")
        file.write(struct.pack("<I", len(meshes)))

        for vertices, indices in meshes:
            file.write(struct.pack(
                "<II",
                len(vertices),
                len(indices)
            ))

            file.write(vertices.astype("<f4").tobytes())
            file.write(indices.astype("<u4").tobytes())

    print("Conversao concluida!")
    print("Malhas:", len(meshes))
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
