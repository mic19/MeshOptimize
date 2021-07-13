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


import bpy, math, mathutils, copy, bmesh
import numpy as np

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


def clear_geodesics(start_index, end_index):
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.select_all(action='DESELECT')
    
    for i in range(start_index, end_index):
        print('geodesic' + str(i))
        bpy.data.objects['geodesic' + str(i)].select_set(True)
        bpy.ops.object.delete()
    get_geodesic_distance.counter = 0
    
    bpy.ops.object.editmode_toggle()

    bpy.data.objects[VertTensor.obj.name].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[VertTensor.obj.name]
    # After changing edit mode bm data is lost
    mesh = VertTensor.obj.data
    VertTensor.target_bm = bmesh.from_edit_mesh(mesh)


def get_geodesic_distance(point1, point2):
    get_geodesic_distance.counter += 1
    steps = 10
    vertices = get_geodesic_points(point1, point2, steps)
    edges = [(i, i + 1) for i in range(steps)]
    faces = []
    
    name = 'geodesic' + str(get_geodesic_distance.counter)
    new_mesh = bpy.data.meshes.new(name)
    new_mesh.from_pydata(vertices, edges, faces)
    new_mesh.update()
    geodesic = bpy.data.objects.new(name, new_mesh)
    bpy.context.collection.objects.link(geodesic)

    bv = BVHTree.FromBMesh(VertTensor.target_bm)
    bm = bmesh.new()
    bm.from_mesh(new_mesh)
    bm.verts.ensure_lookup_table()
    matrix_world = geodesic.matrix_world

    for vert in bm.verts:
        origin = matrix_world @ vert.co
        location, normal, index, distance = bv.find_nearest(origin)
        vert.co = location

    bm.to_mesh(new_mesh)
    length = 0
    for edge in bm.edges:
        length += edge.calc_length()
    bm.free()

    if get_geodesic_distance.counter % 500 == 0:
        print("Removing temp objects")
        clear_geodesics(get_geodesic_distance.counter - 499, get_geodesic_distance.counter + 1)
    
    return length
get_geodesic_distance.counter = 0


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
        self.selected_close_faces = None
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
        
    # select points from selected faces and calculate tensors
    for index in index_to_vert_tensor:
        index_to_vert_tensor[index].select_close_faces(proximity)
        index_to_vert_tensor[index].calculate_tensor()
        index_to_vert_tensor[index].classify()
    
    clear_geodesics(1, get_geodesic_distance.counter + 1)

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
        # TODO: checking context like context.object is not None
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

    
            
            
                    
            
            
            
            
