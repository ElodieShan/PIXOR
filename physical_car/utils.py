import torch

def physical_transforms(mesh, vector=None, angle=None, center=(0, 0, 0), scale=None):
    mesh.verts_packed()

    if angle is not None:
        angle = mesh._verts_packed.new([1])
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)
        b = angle.new(center)
        mesh._verts_packed = b + (mesh._verts_packed - b).matmul(mesh._verts_packed.new([cos, sin, zero, -sin, cos, zero, zero, zero, one]).view(3, 3))

    if vector is not None:
        mesh._verts_packed += vector

    if scale is not None:
        mesh._verts_packed *= scale

    new_verts_list = list(
            mesh._verts_packed.split(mesh.num_verts_per_mesh().tolist(), 0)
        )
    mesh._verts_list = new_verts_list

    # update verts padded
    if mesh._verts_padded is not None:
        for i, verts in enumerate(new_verts_list):
            if len(verts) > 0:
                mesh._verts_padded[i, : verts.shape[0], :] = verts

    # update face areas and normals and vertex normals
    # only if the original attributes are computed
    if any(
        v is not None
        for v in [mesh._faces_areas_packed, mesh._faces_normals_packed]
    ):
        mesh._compute_face_areas_normals(refresh=True)
    if mesh._verts_normals_packed is not None:
        mesh._compute_vertex_normals(refresh=True)


