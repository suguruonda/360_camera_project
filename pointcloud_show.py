import open3d as o3d
import glob
import os 

species_number = 3

base_path = "/data2/suguru/datasets/360camera/butterfly/"
original_folder_list = glob.glob(base_path + "correspondence/*")
original_folder_list.sort()

species_index = species_number - 1
name = os.path.basename(original_folder_list[species_index])

f_name = original_folder_list[species_index] + "/" + name + ".pcd"
#f_name = original_folder_list[species_index] + "/" + name + "_ba_xparam.pcd"


pcd = o3d.io.read_point_cloud(f_name)
alpha = 0.03
print(f"alpha={alpha:.3f}")
"""
tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
"""


o3d.visualization.draw_geometries([pcd])