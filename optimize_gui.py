bl_info = {
    "name": "Rubik's Cube Add-on",
    "description": "Creating and dealing with Rubik's cube",
    "author": "mic19",
    "version": (0, 0, 1),
    "blender": (2, 80, 0),
    "location": "3D View > Tools",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"
}


import bpy, math, mathutils, copy, bmesh, time
import numpy as np
import pyopencl as cl

from enum import Enum
from math import radians
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from bpy.props import (StringProperty,
                       IntProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       EnumProperty,
                       PointerProperty,
                       )
from bpy.types import (Panel,
                       Operator,
                       PropertyGroup,
                       )
                       

# Finding Contours ################################################################################################################
def get_unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)


def get_neigbours(vert):
    """ Returns neigbouring faces """
    return vert.link_faces


def get_link_verts(vert):
    return [edge.other_vert(vert) for edge in vert.link_edges]


def view3d_find( return_area = False ):
    # returns first 3d view, normally we get from context
    for area in bpy.context.window.screen.areas:
        if area.type == 'VIEW_3D':
            v3d = area.spaces[0]
            rv3d = v3d.region_3d
            for region in area.regions:
                if region.type == 'WINDOW':
                    if return_area: return region, rv3d, v3d, area
                    return region, rv3d, v3d
    return None, None

region, rv3d, v3d, area = view3d_find(True)
override = {
    'scene'  : bpy.context.scene,
    'region' : region,
    'area'   : area,
    'space'  : v3d
}


def get_geodesic_points(point1, point2, steps=10):
    step = (point2 - point1)/steps
    return [point1 + i * step for i in range(steps + 1)]


def get_distance(point1, point2):
    vec = point2 - point1
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


def get_geodesic_distance(point1, point2):
    steps = 10
    points = get_geodesic_points(point1, point2, steps)
    bv = BVHTree.FromBMesh(VertTensor.target_bm)

    for i in range(len(points)):
        origin = points[i]
        location, normal, index, distance = bv.find_nearest(origin)
        points[i] = location

    length = 0
    for i in range(len(points) - 1):
        length += get_distance(points[i], points[i + 1])

    return length


def get_close(vert, depth=3):
    verts_to_visit = [vert]
    close_verts = set(verts_to_visit)
    iter = 0
    
    while iter < depth:
        new_verts_to_visit = []
        for v in verts_to_visit:
            close_verts.update(get_link_verts(v))
            new_verts_to_visit.extend(get_link_verts(v))
        verts_to_visit = new_verts_to_visit
        iter += 1

    return close_verts


def get_close_faces(verts):
    """ Returns unique close faces based on list of verts """
    close_faces = set()
    for v in verts:
        close_faces.update(v.link_faces)
    return list(close_faces)


def get_covariance_matrix(tensor_face):
    """ Returns covariance matrix based on face's normal """
    normal = np.array([get_unit_vector(tensor_face.normal)])
    normal_transpose = normal.reshape(3, 1)
    output = normal_transpose @ normal
    return output


def get_num_similar(vector, tolerance, diff_ratio):
    """ Returns number of similar values from vector based
        on tolerance (similarity) and difference ratios """
    a, b, c = vector
    
    if abs(b - c) < tolerance and b > diff_ratio * a and c > diff_ratio * a:
        return 2
    
    if abs(a - c) < tolerance and a > diff_ratio * b and c > diff_ratio * b:
        return 2
    
    if abs(b - a) < tolerance and b > diff_ratio * c and a > diff_ratio * c:
        return 2
    
    return None


class VertClass(Enum):
    SURFACE = 0
    CONTOUR = 1
    CORNER = 2


class FaceTensor:
    def __init__(self, index, center, normal):
        self.index = index
        self.center = center
        self.normal = normal


class VertTensor:
    obj = None
    matrix_world = None
    target_bm = None
    def __init__(self, index, co, normal):
        self.index = index
        self.co = co
        self.normal = normal
        self.close_faces = None
        self.selected_close_faces = []
        self.tensor = None
        self.classification = None

    def set_close_faces(self, close_faces):
        self.close_faces = close_faces
   
    def select_close_faces(self, proximity):
        self.selected_close_faces = []
        
        for tensor_face in self.close_faces:
            length = get_geodesic_distance(self.co, tensor_face.center)
            if length < proximity:
                self.selected_close_faces.append(tensor_face)

    def get_points_to_measure(self):
        froms = [[self.co[0], self.co[1], self.co[2]] for i in range(len(self.close_faces))]
        tos = [[tf.center[0], tf.center[1], tf.center[2]] for tf in self.close_faces]
        return froms, tos

    def calculate_tensor(self):
        tensor = np.zeros((3, 3))
        for face in self.selected_close_faces:
            tensor = tensor + get_covariance_matrix(face)
        self.tensor = tensor
    
    def classify(self):
        eigvals = np.linalg.eigvals(self.tensor)
        corner_ratio = 7
        contour_ratio = 5
        
        max_val = max(eigvals)
        others = np.delete(eigvals, np.where(eigvals == max_val))
        
        if len(others) < 2:
            self.classification = VertClass.CORNER
            return VertClass.CORNER
        
        if max_val > corner_ratio * others[0] and max_val > corner_ratio * others[1]:
            self.classification = VertClass.SURFACE
            #print("surface")
            return VertClass.SURFACE
        
        if get_num_similar(eigvals, contour_ratio, corner_ratio) is 2:
            self.classification = VertClass.CONTOUR
            #print("contour")
            return VertClass.CONTOUR
        
        self.classification = VertClass.CORNER
        #print("corner")
        return VertClass.CORNER
    
    
def select_contours_main(adjacency_depth, proximity):
    obj = bpy.context.edit_object
    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)
    
    VertTensor.obj = obj
    VertTensor.matrix_world = obj.matrix_world
    VertTensor.target_bm = bm
    index_to_vert_tensor = {v.index : VertTensor(v.index, v.co.copy(), v.normal.copy()) for v in bm.verts}
    iter = 0

    # find close verts for calculating tensors
    for index in index_to_vert_tensor:
        bmvert = bm.verts[index]
        close_bm_verts = [v for v in get_close(bmvert, adjacency_depth)]
        close_faces = [FaceTensor(f.index, f.calc_center_median(), f.normal.copy()) for f in get_close_faces(close_bm_verts)]
        index_to_vert_tensor[index].set_close_faces(close_faces)
    
    select_time = 0
    calculate_time = 0
    classify_time = 0
    
    length = 0
    vert_froms_arrays = []
    vert_tos_arrays = []
    next_indexes = []
    
    # Get point from which and to which measure distance
    for index in index_to_vert_tensor:
        vert_froms, vert_tos = index_to_vert_tensor[index].get_points_to_measure()
        
        vert_froms_arrays.append([])
        vert_tos_arrays.append([])
        for i in range(len(vert_froms)):
            vert_froms_arrays[index].append(vert_froms[i])
            vert_tos_arrays[index].append(vert_tos[i])
        
        next_indexes.append(length)
        length += len(vert_tos)
    
    next_indexes.append(length)
    froms = np.zeros(shape=(length, 3), dtype=np.float32)
    tos = np.zeros(shape=(length, 3), dtype=np.float32)
    
    # Copy these points to numpy array to be able to use in opencl
    array_iter = 0
    for index in index_to_vert_tensor:
        for i in range(len(vert_froms_arrays[index])):
            froms[array_iter + i] = vert_froms_arrays[index][i]
            tos[array_iter + i] = vert_tos_arrays[index][i]
        
        array_iter += len(vert_froms_arrays[index])
    
    print('- From points -')
    print(froms)
    print('- To points -')
    print(tos)
    
    # PREPARING INPUT
    faces_points1_world = [VertTensor.matrix_world @ face.verts[0].co for face in VertTensor.target_bm.faces]
    faces_points2_world = [VertTensor.matrix_world @ face.verts[1].co for face in VertTensor.target_bm.faces]
    faces_points3_world = [VertTensor.matrix_world @ face.verts[2].co for face in VertTensor.target_bm.faces]
    
    points1 = np.array([[p[0], p[1], p[2]] for p in faces_points1_world], dtype=np.float32)
    points2 = np.array([[p[0], p[1], p[2]] for p in faces_points2_world], dtype=np.float32)
    points3 = np.array([[p[0], p[1], p[2]] for p in faces_points3_world], dtype=np.float32)
    
    normals = np.array([[face.normal.x, face.normal.y, face.normal.z] for face in VertTensor.target_bm.faces], dtype=np.float32)
    length = np.array([len(VertTensor.target_bm.faces)], dtype=np.int32)
    
    output_distances = np.zeros(shape=(len(froms)), dtype=np.float32)
    
    # OPENCL
    cntxt = cl.create_some_context()
    queue = cl.CommandQueue(cntxt)

    points1_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=points1)
    points2_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=points2)
    points3_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=points3)
    
    froms_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=froms)
    tos_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=tos)
    
    normals_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=normals)
    length_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=length)

    output_distances_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY, output_distances.nbytes)

    # Kernel Program
    code = """
    float dot_operator(float* vec1, float* vec2)
    {
        float output = 0;
        for (int i = 0; i < 3; ++i)
            output += vec1[i] * vec2[i];
        return output;
    }
    
    float* cross_operator(float* vec1, float* vec2, float* output)
    {
        output[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
        output[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
        output[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];
        return output;
    }
    
    float get_length(float* vec)
    {
        return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    }
    
    float* substract(float* vec1, float* vec2, float* output)
    {
        for (int i = 0; i < 3; ++i)
            output[i] = vec1[i] - vec2[i];
        return output;
    }
    
    float* add(float* vec1, float* vec2, float* output)
    {
        for (int i = 0; i < 3; ++i)
            output[i] = vec1[i] + vec2[i];
        return output;
    }
    
    float* multiply(float* vec, float val, float* output)
    {
        for (int i = 0; i < 3; ++i)
            output[i] = vec[i] * val;
        return output;
    }
    
    void copy(float* from, float* to)
    {
        for (int i = 0; i < 3; ++i)
            to[i] = from[i];
    }
    
    void get_barycentric(float* a, float* b, float* c, float* p, float* coors)
    {
        float v0[3], v1[3], v2[3];
        substract(b, a, v0);
        substract(c, a, v1);
        substract(p, a, v2);
        
        float d00 = dot_operator(v0, v0);
        float d01 = dot_operator(v0, v1);
        float d11 = dot_operator(v1, v1);
        float d20 = dot_operator(v2, v0);
        float d21 = dot_operator(v2, v1);
        float denom = d00 * d11 - d01 * d01;
        
        coors[0] = (d11 * d20 - d01 * d21) / denom; //v
        coors[1] = (d00 * d21 - d01 * d20) / denom; //w
        coors[2] = 1.0 - coors[0] - coors[1]; //u
    }
    
    bool is_inside(float* point1, float* point2, float* point3, float* p)
    {
        float coors[3];
        get_barycentric(point1, point2, point3, p, coors);
        
        for (int i = 0; i < 3; ++i)
        {
            if (coors[i] < 0 || coors[i] > 1)
                return false;
        }
        
        return true;
    }
    
    bool is_point_projection_on_line_segment(float coor1, float coor2, float projection_coor)
    {
        float delta = 0.02;
        bool flag = false;
        if (coor1 < coor2)
            flag = (coor1 < (projection_coor + delta) && (projection_coor - delta) < coor2);
        else
            flag = (coor2 < (projection_coor + delta) && (projection_coor - delta) < coor1);
        return flag;
    }
    
    bool is_point_projection_on_line(float* point1, float* point2, float* p, float* output_projection)
    {
        float ap[3], ab[3];
        substract(p, point1, ap);
        substract(point2, point1, ab);
        float mul = dot_operator(ap, ab) / dot_operator(ab, ab);
        float projection[3];
        float temp[3] = {mul * ab[0], mul * ab[1], mul * ab[2]};
        add(point1, temp, projection);
        bool flag_x = false, flag_y = false, flag_z = false;
        
        copy(projection, output_projection);
        
        flag_x = is_point_projection_on_line_segment(point1[0], point2[0], projection[0]);
        flag_y = is_point_projection_on_line_segment(point1[1], point2[1], projection[1]);
        flag_z = is_point_projection_on_line_segment(point1[2], point2[2], projection[2]);
        
        return flag_x && flag_y && flag_z;
    }
    
    float get_distance_from_line(float* point1, float* point2, float* p, float* output_point)
    {
        float min_distance = 10000;
        float step[3], d[3], temp[3];
        
        substract(point2, point1, step);
        substract(point1, p, d);
        float output_projection[3];
        
        if (is_point_projection_on_line(point1, point2, p, output_projection))
        {
            float distance = fabs(get_length(cross_operator(step, d, temp)))/(fabs(get_length(step)));
            if (distance < min_distance)
            {
                min_distance = distance;
                copy(output_projection, output_point);
            }
        } else
        {
            float temp[3];
            substract(point1, p, temp);
            float distance1 = get_length(temp);
            substract(point2, p, temp);
            float distance2 = get_length(temp);
            
            if (distance1 < min_distance)
            {
                min_distance = distance1;
                copy(point1, output_point);
            }
                
            if (distance2 < min_distance)
            {
                min_distance = distance2;
                copy(point2, output_point);
            }
        }
        
        return min_distance;
    }
    
    float get_min_distance_outside(float* point1, float* point2, float* point3, float* p, float* output_point)
    {
        float min_distance = 10000;
        float temp1[3], temp2[3], temp3[3];
        float distance1 = get_distance_from_line(point1, point2, p, temp1);
        float distance2 = get_distance_from_line(point2, point3, p, temp2);
        float distance3 = get_distance_from_line(point1, point3, p, temp3);
        
        if (distance1 < min_distance)
        {
            min_distance = distance1;
            copy(temp1, output_point);
        }
        if (distance2 < min_distance)
        {
            min_distance = distance2;
            copy(temp2, output_point);
        }
        if (distance3 < min_distance)
        {
            min_distance = distance3;
            copy(temp3, output_point);
        }
        
        return min_distance;
    }
    
    void get_point_on_mesh(float* points1, float* points2, float* points3,
                               float* normals, int length, float* p,
                               float* output_point
                               )
    {
        float min_distance = INFINITY;
        
        for (int i = 0; i < length; ++i)
        {
            int iter = 3 * i;
            float point1[3] = {points1[iter], points1[iter + 1], points1[iter + 2]};
            float point2[3] = {points2[iter], points2[iter + 1], points2[iter + 2]};
            float point3[3] = {points3[iter], points3[iter + 1], points3[iter + 2]};
            
            bool is_inside_flag = is_inside(point1, point2, point3, p);
            if (is_inside_flag)
            {
                float A = normals[iter], B = normals[iter + 1], C = normals[iter + 2];
                float D = -A * points1[iter] - B * points1[iter + 1] - C * points1[iter + 2];
                float distance = (A * p[0] + B * p[1] + C * p[2] + D)/sqrt(A * A + B * B + C * C);
                if (fabs(distance) < min_distance)
                {
                    min_distance = fabs(distance);
                    output_point[0] = p[0] - distance * normals[iter];
                    output_point[1] = p[1] - distance * normals[iter + 1];
                    output_point[2] = p[2] - distance * normals[iter + 2];
                }
            }else
            {
                float temp[3];
                float distance = get_min_distance_outside(point1, point2, point3, p, temp);
                if (distance < min_distance)
                {
                    min_distance = distance;
                    output_point[0] = temp[0];
                    output_point[1] = temp[1];
                    output_point[2] = temp[2];
                }
            }
        }
        
        
    }
    
    void get_middle_points(float* from, float* to, float* output_middle_points, int length)
    {
        float difference[3], step[3];
        substract(to, from, difference);
        multiply(difference, 1/(float)(length - 1), step);
        
        //printf("|%f, %f, %f|", step[0], step[1], step[2]);
        
        for (int i = 0; i < length; ++i)
        {
            float middle_point[3], temp[3];
            add(from, multiply(step, i, temp), middle_point);
            copy(middle_point, output_middle_points + 3 * i);
        }
    }
    
    #define STEP_IN_GEODESIC 10
    
    __kernel void get_geodesic_distances(__global float* points1, __global float* points2, __global float* points3,
                                        __global float* normals, __global int* length,
                                        __global float* froms, __global float* tos, __global float* output_distances)
    {
        int gid = get_global_id(0);
        
        //printf("|%d", gid);
        
        float middle_points[3 * STEP_IN_GEODESIC], points_projections[3 * STEP_IN_GEODESIC];
        get_middle_points(froms + gid * 3, tos + gid * 3, middle_points, STEP_IN_GEODESIC);
        
        for (int i = 0; i < STEP_IN_GEODESIC; i+=3)
        {
            get_point_on_mesh(points1, points2, points3, normals, length[0], middle_points + i, points_projections + i);
        }
        
        float distance = 0;
        for (int i = 0; i < STEP_IN_GEODESIC - 3; i+=3)
        {
            float temp[3];
            substract(points_projections + i + 3, points_projections + i, temp);
            distance += get_length(temp);
        }
        
        output_distances[gid] = distance;
        //printf("|%f|", distance);
    }
    """
    
    start = time.perf_counter()

    bld = cl.Program(cntxt, code).build()
    launch = bld.get_geodesic_distances(queue, [len(froms)], None,
                                   points1_buf, points2_buf, points3_buf,
                                   normals_buf, length_buf,
                                   froms_buf, tos_buf,
                                   output_distances_buf)

    launch.wait()

    cl.enqueue_copy(queue, output_distances, output_distances_buf)
    
    end = time.perf_counter()

    print("Elepsed time OpenCl: " + str(end - start))
    print(len(output_distances))
    print(output_distances)
    
    close_counter = 0
    not_close_counter = 0
    
    tensor_index = 0
    for i in range(len(next_indexes) - 1):
        index = next_indexes[i]
        next_index = next_indexes[i + 1]
        for j in range(index, next_index):
            if(output_distances[j] < proximity):
                index_to_vert_tensor[tensor_index].selected_close_faces.append(index_to_vert_tensor[tensor_index].close_faces[j - index])
                close_counter += 1
            else:
                not_close_counter += 1
        tensor_index += 1
    
    print("close: " + str(close_counter))
    print("not close: " + str(not_close_counter))
    
    for index in index_to_vert_tensor:
        start = time.time()
        index_to_vert_tensor[index].calculate_tensor()
        end = time.time()
        calculate_time = end - start
        #print("calculate tensor: " + str(end - start))
        
        start = time.time()
        index_to_vert_tensor[index].classify()
        end = time.time()
        classify_time += end - start
        #print("classify: " + str(end - start))
    
    print("select close face: " + str(select_time))
    print("calculate tensor: " + str(calculate_time))
    print("classify: " + str(classify_time))

    obj = bpy.context.edit_object
    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)
    bm.verts.ensure_lookup_table()
    
    for index in index_to_vert_tensor:
        if index_to_vert_tensor[index].classification is VertClass.CONTOUR:
            bm.verts[index].select = True
        if index_to_vert_tensor[index].classification is VertClass.CORNER:
            bm.verts[index].select = True
            
    bmesh.update_edit_mesh(mesh)


# Divide Surfaces #################################################################################################################
def get_connected_verts(vert):
    output = []
    for e in vert.link_edges:
        other = e.other_vert(vert)
        if other.select:
            output.append(other)
    return output


def remove_verts(verts, verts_to_remove):
    for v in verts_to_remove:
        if v in verts:
            verts.remove(v)


def get_cluster(selected_verts, vert):
    cluster = {vert}
    to_visit = {vert}
    visited = set()
    
    while len(to_visit) > 0:
        current_vert = to_visit.pop()
        visited.add(current_vert)
        
        connected = get_connected_verts(current_vert)
        next = [v for v in connected if not v in visited]
        
        to_visit.update(next)
        cluster.update(connected)
    
    return cluster


def divide_surfaces_main():
    obj = bpy.context.edit_object
    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)
    
    clusters = []
    current_cluster = 0
    selected_verts = [v for v in bm.verts if v.select]

    while len(selected_verts) > 0:
        current_vert = selected_verts[0]
        cluster = get_cluster(selected_verts, current_vert)
        clusters.append([])
        clusters[current_cluster].extend(cluster)
        current_cluster += 1
        remove_verts(selected_verts, cluster)
    
    bpy.ops.mesh.select_all(action='DESELECT')
    
    contour_cluster = clusters[0]
    for cluster in clusters:
        print(len(cluster))
        if len(cluster) > len(contour_cluster):
            contour_cluster = cluster
    
    for v in contour_cluster:
        v.select = True
    
    indices = [v.index for v in clusters[2]]
    group = obj.vertex_groups.new(name = 'Contours')
    bpy.ops.object.mode_set(mode='OBJECT')
    group.add(indices, 0, 'REPLACE')
    bpy.ops.object.mode_set(mode='EDIT')


# Optimize Surfaces ###############################################################################################################
def select_surfaces():
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.object.vertex_group_set_active(group='Contours')
    bpy.ops.object.vertex_group_deselect()


def remesh_surfaces(resolution, iterations):
    bpy.ops.mesh.delete(type='FACE')

    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.object.vertex_group_set_active(group='Contours')
    bpy.ops.object.vertex_group_deselect()
    
    bpy.ops.mesh.edge_face_add()
    bpy.ops.mesh.split()
    bpy.ops.remesh.boundary_aligned_remesh(resolition=resolution, iterations=iterations)
    
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles()
    bpy.ops.mesh.select_all(action='DESELECT')


def optimize_surfaces_main(resolution, iterations):
    select_surfaces()
    remesh_surfaces(resolution, iterations)


# Operators - Select Contours #####################################################################################################
class OperatorSelectContoursProperties(bpy.types.PropertyGroup):
    AdjacencyDepth: IntProperty(
        name="Adjacency Depth",
        description="Adjacency depth of the algorithm",
        default=1,
        min=0,
        max=4
    )
    Proximity: FloatProperty(
        name="Proximity",
        description="Proximity of verts taken into account",
        default=3,
        min=0,
        max=10
    )
    CornerRatio: FloatProperty(
        name="Corner Ratio",
        description="Corner ratio of the algorithm",
        default=7,
        min=0,
        max=2
    )
    ContourRatio: FloatProperty(
        name="Contour Ratio",
        description="Contoru ratio depth of the algorithm",
        default=5,
        min=0,
        max=20
    )


class RC_OT_SelectContours(Operator):
    bl_label = "Select Contours"
    bl_idname = "optimize.select_contours"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        adjacency_depth = context.scene.select_contours_props.AdjacencyDepth
        proximity = context.scene.select_contours_props.Proximity
        corner_ratio = context.scene.select_contours_props.CornerRatio
        contour_ratio = context.scene.select_contours_props.ContourRatio

        select_contours_main(adjacency_depth, proximity)

        bpy.ops.wm.tool_set_by_id(name='builtin.select_box', space_type='VIEW_3D')
        return {'FINISHED'}


# Operators - Divide Surfaces #####################################################################################################
class OperatorDivideSurfacesProperties(bpy.types.PropertyGroup):
    AdjacencyDepth: IntProperty(
        name="Adjacency Depth",
        description="Adjacency depth of the algorithm",
        default=1,
        min=0,
        max=4
    )


class RC_OT_DivideSurfaces(Operator):
    bl_label = "Confirm"
    bl_idname = "optimize.divide_surfaces"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        divide_surfaces_main()
        bpy.ops.wm.tool_set_by_id(name='builtin.select_box', space_type='VIEW_3D')
        return {'FINISHED'}


# Operators - Optimize Surface ####################################################################################################
class OperatorOptimizeSurfacesProperties(bpy.types.PropertyGroup):
    Resolution: FloatProperty(
        name="Resolution",
        description="Resolution of the remesh algorithm",
        default=4,
        min=1,
        max=10
    )
    Iterations: IntProperty(
        name="Iterations",
        description="Iterations of the remesh algorithm",
        default=30,
        min=1,
        max=100
    )


class RC_OT_OptimizeSurfaces(Operator):
    bl_label = "Optimize Surfaces"
    bl_idname = "optimize.optimize_surfaces"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        resolution = context.scene.optimize_surfaces_props.Resolution
        iterations = context.scene.optimize_surfaces_props.Iterations
        optimize_surfaces_main(resolution, iterations)
        bpy.ops.wm.tool_set_by_id(name='builtin.select_box', space_type='VIEW_3D')
        return {'FINISHED'}


# Panels ##########################################################################################################################
class View3DPanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return True


class SelectContoursPanel(View3DPanel, bpy.types.Panel):
    bl_idname = "VIEW3D_PT_select_contours"
    bl_label = "Select Contours"

    def draw(self, context):
        scene = context.scene

        self.layout.prop(scene.select_contours_props, "AdjacencyDepth")
        self.layout.prop(scene.select_contours_props, "Proximity")
        self.layout.prop(scene.select_contours_props, "CornerRatio")
        self.layout.prop(scene.select_contours_props, "ContourRatio")
        self.layout.operator("optimize.select_contours")

        self.layout.label(text="", icon_value=custom_icons["cube_icon"].icon_id)
        
        
class DivideSurfacesPanel(View3DPanel, bpy.types.Panel):
    bl_idname = "VIEW3D_PT_divide_surfaces"
    bl_label = "Divide Surfaces"

    def draw(self, context):
        scene = context.scene

        #self.layout.prop(scene.divide_surfaces_props, "AdjacencyDepth")
        self.layout.operator("optimize.divide_surfaces")

        self.layout.label(text="", icon_value=custom_icons["cube_icon"].icon_id) 
        
        
class OptimizeSurfacesPanel(View3DPanel, bpy.types.Panel):
    bl_idname = "VIEW3D_PT_optimize_surfaces"
    bl_label = "Optimize Surfaces"

    def draw(self, context):
        scene = context.scene

        self.layout.prop(scene.optimize_surfaces_props, "Resolution")
        self.layout.prop(scene.optimize_surfaces_props, "Iterations")
        self.layout.operator("optimize.optimize_surfaces")

        self.layout.label(text="", icon_value=custom_icons["cube_icon"].icon_id) 


classes = (
    RC_OT_SelectContours,
    RC_OT_DivideSurfaces,
    RC_OT_OptimizeSurfaces,
    OperatorSelectContoursProperties,
    OperatorDivideSurfacesProperties,
    OperatorOptimizeSurfacesProperties,
    SelectContoursPanel,
    DivideSurfacesPanel,
    OptimizeSurfacesPanel
)

custom_icons = None


def register():
    from bpy.utils import register_class, register_tool, previews
    global custom_icons
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))

    custom_icons = bpy.utils.previews.new()
    icons_dir = dir_path + "icons"
    custom_icons.load("cube_icon", os.path.join(icons_dir, "icon.png"), 'IMAGE')

    for cls in classes:
        register_class(cls)

    bpy.types.Scene.select_contours_props = bpy.props.PointerProperty(type=OperatorSelectContoursProperties)
    bpy.types.Scene.divide_surfaces_props = bpy.props.PointerProperty(type=OperatorDivideSurfacesProperties)
    bpy.types.Scene.optimize_surfaces_props = bpy.props.PointerProperty(type=OperatorOptimizeSurfacesProperties)


def unregister():
    from bpy.utils import unregister_class, unregister_tool, previews
    global custom_icons
    bpy.utils.previews.remove(custom_icons)

    for cls in reversed(classes):
        unregister_class(cls)

    del bpy.types.Scene.select_contours_props
    del bpy.types.Scene.divide_surfaces_props
    del bpy.types.Scene.optimize_surfaces_props


if __name__ == "__main__":
    register()

    
            
            
                    
            
            
            
            

