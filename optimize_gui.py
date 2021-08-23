bl_info = {
    "name": "Feature Selection",
    "description": "Selecting features based on normal vector voting",
    "author": "mic19",
    "version": (0, 0, 1),
    "blender": (2, 80, 0),
    "location": "3D View > Tools",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"
}


import bpy, math, mathutils, copy, bmesh, time, sys, re
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


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def are_similar_directions(vec1, vec2, degrees=15):
    angle = angle_between(vec1, vec2)
    if angle < math.radians(degrees) and angle > math.radians(-degrees):
        return True
    if angle < math.radians(180 + degrees) and angle > math.radians(180 - degrees):
        return True
    return False


class VertClass(Enum):
    SURFACE = 0
    CONTOUR = 1
    CORNER = 2


class FaceTensor:
    def __init__(self, index, center, normal):
        self.index = index
        self.center = center
        self.normal = normal
        self.covariance_matrix = self.get_covariance_matrix()
    
    def get_covariance_matrix(self):
        """ Returns covariance matrix based on face's normal """
        normal = np.array([get_unit_vector(self.normal)])
        normal_transpose = normal.reshape(3, 1)
        output = normal_transpose @ normal
        return output


class VertTensor:
    obj = None
    matrix_world = None
    target_bm = None
    corner_threshold = 0.35
    contour_threshold = 0.2
    def __init__(self, index, co, normal):
        self.index = index
        self.co = co
        self.normal = normal
        self.close_faces = None
        self.selected_close_faces = None
        self.tensor = None
        self.classification = None
        self.norm_eigvals = None
        self.eigvectors = None
        self.direction = None
        self.initial_measure = None
        self.final_measure = 0

    def set_close_faces(self, close_faces):
        self.close_faces = close_faces
   
    def select_close_faces(self, proximity):
        #self.selected_close_faces = self.close_faces.copy()
        self.selected_close_faces = []
        for tensor_face in self.close_faces:
            length = get_distance(self.co, tensor_face.center) #get_geodesic_distance
            if length < proximity:
                self.selected_close_faces.append(tensor_face)
    
    time1 = 0
    time2 = 0
    time3 = 0

    def calculate_tensor(self):
        tensor = np.zeros((3, 3))
        
        for tensor_face in self.selected_close_faces:
            start = time.time()
            A = VertTensor.target_bm.faces[tensor_face.index].calc_area()
            end = time.time()
            VertTensor.time1 += end - start
            
            Amax, distance_max = 0, 0
            
            start = time.time()
            for face in VertTensor.target_bm.verts[self.index].link_faces:
                surface = face.calc_area()
                if surface > Amax:
                    Amax = surface
                d = get_distance(face.calc_center_median(), self.co)
                if d > distance_max:
                    distance_max = d
            end = time.time()
            VertTensor.time2 += end - start
            
            start = time.time()
            distance = get_distance(tensor_face.center, self.co)
            coeff = A/Amax * math.exp(-distance/distance_max)
            tensor = tensor + coeff * tensor_face.covariance_matrix
            end = time.time()
            VertTensor.time3 += end - start
        
        self.tensor = tensor
        #print(tensor)
    
    def classify(self):
        eigvals = np.linalg.eigvals(self.tensor)
        
        max_val = max(eigvals)
        others = np.delete(eigvals, np.where(eigvals == max_val))
        
        if len(others) < 2:
            self.classification = VertClass.CORNER
            return VertClass.CORNER
        
        if max_val > VertTensor.corner_ratio * others[0] and max_val > VertTensor.corner_ratio * others[1]:
            self.classification = VertClass.SURFACE
            return VertClass.SURFACE
        
        if get_num_similar(eigvals, VertTensor.contour_ratio, VertTensor.corner_ratio) is 2:
            self.classification = VertClass.CONTOUR
            return VertClass.CONTOUR
        
        self.classification = VertClass.CORNER
        return VertClass.CORNER
    
    def prepare_curve_classification(self):
        eigvals, self.eigvectors = np.linalg.eig(self.tensor)
        self.norm_eigvals = eigvals / np.sqrt(np.sum(eigvals**2))
        self.initial_measure = self.norm_eigvals.sum()/2 - 1/2
        self.direction = self.eigvectors[self.norm_eigvals.argmin()]


def select_contours_main(adjacency_depth, proximity):
    obj = bpy.context.edit_object
    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)
    
    all_start = time.time()
    starting = time.time()
    
    VertTensor.obj = obj
    VertTensor.matrix_world = obj.matrix_world
    VertTensor.target_bm = bm
    index_to_vert_tensor = {v.index : VertTensor(v.index, v.co.copy(), v.normal.copy()) for v in bm.verts}

    # Find close verts for calculating tensors, from them closest verts are selected
    for index in index_to_vert_tensor:
        bmvert = bm.verts[index]
        close_bm_verts = [v for v in get_close(bmvert, adjacency_depth)]
        close_faces = [FaceTensor(f.index, f.calc_center_median(), f.normal.copy()) for f in get_close_faces(close_bm_verts)]
        index_to_vert_tensor[index].set_close_faces(close_faces)
    
    max_measure = 0
    min_measure = sys.float_info.max
    max_final_measure = 0
    min_final_measure = sys.float_info.max
    
    
    # Select points from selected faces and calculate tensors
    for index in index_to_vert_tensor:
        index_to_vert_tensor[index].select_close_faces(proximity)
        index_to_vert_tensor[index].calculate_tensor()
        index_to_vert_tensor[index].prepare_curve_classification()
        
        m = index_to_vert_tensor[index].initial_measure
        if m > max_measure:
            max_measure = m
        if m < min_measure:
            min_measure = m

    print('calculating tensors: ' + str(time.time() - starting))
    starting = time.time()
    
    # Normalize initial measure
    for index in index_to_vert_tensor:
        index_to_vert_tensor[index].initial_measure = (index_to_vert_tensor[index].initial_measure - min_measure)/(max_measure - min_measure)
    
    # Mean length of edge in the model
    mean_edge_length = 0
    for edge in VertTensor.target_bm.edges:
        mean_edge_length += edge.calc_length()
    mean_edge_length /= len(VertTensor.target_bm.edges)
    
    contours = 0
    corners = 0
    
    print('normalizing and finding mean length: ' + str(time.time() - starting))
    starting = time.time()
    
    start = time.time()
    for index in index_to_vert_tensor:
        
        if index_to_vert_tensor[index].initial_measure < VertTensor.contour_threshold:
            continue
        if index_to_vert_tensor[index].initial_measure > VertTensor.corner_threshold:
            #index_to_vert_tensor[index].classification = VertClass.CORNER
            index_to_vert_tensor[index].final_measure = index_to_vert_tensor[index].initial_measure
            corners += 1
            continue
        
        contours += 1
        
        # Prepare supporting neigbours
        sn = []
        visited = [False for i in range(len(index_to_vert_tensor))]
        curr_index = index
        verts_to_visit = [VertTensor.target_bm.verts[curr_index]]
        
        counter = 0
        while(len(sn) < 5 and counter < 10 and len(verts_to_visit) > 0):
            curr_index = verts_to_visit[-1].index
            verts_to_visit.pop()
            candidates = [v for v in get_close(VertTensor.target_bm.verts[curr_index], 1) if visited[v.index] is False]
            if VertTensor.target_bm.verts[curr_index] in candidates:
                candidates.remove(VertTensor.target_bm.verts[curr_index])
            visited[curr_index] = True
            
            vec1 = index_to_vert_tensor[curr_index].direction
            for v in candidates:
                # In neigbouring verts looking for these with similar direction
                vec2 = index_to_vert_tensor[v.index].direction
                temp = are_similar_directions(vec1, vec2, 20)
                
                if visited[v.index] is False and temp:
                    sn.append(index_to_vert_tensor[v.index])
                    # Close verts of this vert may be visited as they support this vert
                    verts_to_visit.append(v)
                visited[v.index] = True
            
            counter += 1
            
        final_measure = 0
        if len(sn) > 3:
            for v in sn:
                weight = math.exp(-get_distance(index_to_vert_tensor[index].co, v.co)/(2*(1.5*mean_edge_length)**2))
                final_measure += weight * v.initial_measure

        index_to_vert_tensor[index].final_measure = final_measure
        
        if final_measure < min_final_measure:
            min_final_measure = final_measure
        if final_measure > max_final_measure:
            max_final_measure = final_measure

    # 20% of vertices with greatest measure are considered corners
    s = {k: v for k, v in sorted(index_to_vert_tensor.items(), key=lambda item: item[1].final_measure, reverse=True) if v.final_measure is not 0}
    corners_threshold = 0.2 * len(s)
    iter = 0
    for i in s:
        if iter < corners_threshold:
            s[i].classification = VertClass.CORNER
        else:
            if s[i].final_measure == 0:
                s[i].classification = VertClass.SURFACE
            else:
                s[i].classification = VertClass.CONTOUR
        iter += 1

    print('classfing time: ' + str(time.time() - starting))
    starting = time.time()

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
    bpy.types.Scene.index_to_vert_tensor = index_to_vert_tensor
    
    bm = bmesh.from_edit_mesh(mesh)
    for vert in bm.verts:
        connected = get_connected_verts(vert)
        selected = 0
        for v in connected:
            if v.select == True:
                selected += 1
        if selected == len(connected):
            vert.select = True
    
#    all_end = time.time()
#    print('all time: ' + str(all_end - all_start))
    print('selecting time: ' + str(time.time() - starting))
    starting = time.time()


# Divide Surfaces #################################################################################################################
def get_connected_verts_selected(vert):
    output = []
    for e in vert.link_edges:
        other = e.other_vert(vert)
        if other.select:
            output.append(other)
    return output


def get_connected_verts(vert):
    output = []
    for e in vert.link_edges:
        other = e.other_vert(vert)
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
        
        connected = get_connected_verts_selected(current_vert)
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

    # Separate groups of selected vertices
    while len(selected_verts) > 0:
        current_vert = selected_verts[0]
        cluster = get_cluster(selected_verts, current_vert)
        clusters.append([])
        clusters[current_cluster].extend(cluster)
        current_cluster += 1
        remove_verts(selected_verts, cluster)
    
    # Prepare indexes of vertices to add vertex groups
    indexes_groups = []
    for cluster in clusters:
        print(len(cluster))
        indexes_group = []
        for v in cluster:
            v.select = True
            indexes_group.append(v.index)
        indexes_groups.append(indexes_group)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    iter = 1
    for indexes_group in indexes_groups:
        group = obj.vertex_groups.new(name = 'Element' + str(iter))
        group.add(indexes_group, 0, 'REPLACE')
        iter += 1
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    
    bpy.context.active_object.iter = 1
    bpy.context.active_object.num_elements = len(clusters)



# Optimize Surfaces ###############################################################################################################
def expand_selection():
    obj = bpy.context.edit_object
    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)
    
    selected = [bv for bv in bm.verts if bv.select == True]
    print(len(selected))
    
    connected = set()
    for bv in selected:
        connected.update(get_link_verts(bv))
    
    for bv in connected:
        bv.select = True


def select_element(index):
    obj = bpy.context.edit_object
    #group = obj.vertex_groups['Element' + str(index)]
    group_name = 'Element' + str(index)
    print(group_name)
    
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode="OBJECT")
    verts = [vert for vert in obj.data.vertices if obj.vertex_groups[group_name].index in [i.group for i in vert.groups]]
    print(len(verts))
    
    for vert in verts:
        obj.data.vertices[vert.index].select = True
        vert.select = True
    
    bpy.ops.object.mode_set(mode="EDIT")


def optimize_select_prev_main():
    print('select prev')
    bpy.context.active_object.iter -= 1
    if bpy.context.active_object.iter < 1:
        bpy.context.active_object.iter = bpy.context.active_object.num_elements
    
    print('iter: ' + str(bpy.context.active_object.iter))
    select_element(bpy.context.active_object.iter)


def optimize_select_next_main():
    print('select next')
    bpy.context.active_object.iter += 1
    if bpy.context.active_object.iter > bpy.context.active_object.num_elements:
        bpy.context.active_object.iter = 1

    print('iter: ' + str(bpy.context.active_object.iter))
    print('num_elements: ' + str(bpy.context.active_object.num_elements))
    select_element(bpy.context.active_object.iter)


def optimize_focus_main():
    print('focus')

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            ctx = bpy.context.copy()
            ctx['area'] = area
            ctx['region'] = area.regions[-1]
            bpy.ops.view3d.view_selected(ctx) # points view
            # bpy.ops.view3d.camera_to_view_selected(ctx) # points camera


def optimize_surfaces_main():
    expand_selection()
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.mesh.select_non_manifold()
    bpy.ops.mesh.fill()
    bpy.ops.mesh.select_all(action='DESELECT')
    
    bpy.ops.mesh.print3d_clean_non_manifold()
    bpy.ops.mesh.select_all(action='DESELECT')
    
#    bpy.ops.object.modifier_add(type='DECIMATE')
#    bpy.context.object.modifiers['Decimate'].decimate_type = 'COLLAPSE'


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
    CornerThreshold: FloatProperty(
        name="Corner Threshold",
        description="Corner threshold of the algorithm",
        default=0.35,
        min=0,
        max=1
    )
    ContourThreshold: FloatProperty(
        name="Contour Threshold",
        description="Contour threshold depth of the algorithm",
        default=0.2,
        min=0,
        max=1
    )


class RC_OT_SelectContours(Operator):
    bl_label = "Select Features"
    bl_idname = "optimize.select_features"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        adjacency_depth = context.scene.select_contours_props.AdjacencyDepth
        proximity = context.scene.select_contours_props.Proximity
        VertTensor.corner_threshold = context.scene.select_contours_props.CornerThreshold
        VertTensor.contour_threshold = context.scene.select_contours_props.ContourThreshold

        select_contours_main(adjacency_depth, proximity)

        bpy.ops.wm.tool_set_by_id(name='builtin.select_box', space_type='VIEW_3D')
        return {'FINISHED'}


# Operators - Divide Surfaces #####################################################################################################
class RC_OT_DivideSurfaces(Operator):
    bl_label = "Confirm"
    bl_idname = "optimize.divide_model"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        divide_surfaces_main()
        bpy.ops.wm.tool_set_by_id(name='builtin.select_box', space_type='VIEW_3D')
        return {'FINISHED'}


# Operators - Optimize Surfaces ####################################################################################################
class RC_OT_SelectPrevious(Operator):
    bl_label = "Previous"
    bl_idname = "optimize.select_prev"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        optimize_select_prev_main()
        bpy.ops.wm.tool_set_by_id(name='builtin.select_box', space_type='VIEW_3D')
        return {'FINISHED'}


class RC_OT_SelectNext(Operator):
    bl_label = "Next"
    bl_idname = "optimize.select_next"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        optimize_select_next_main()
        bpy.ops.wm.tool_set_by_id(name='builtin.select_box', space_type='VIEW_3D')
        return {'FINISHED'}


class RC_OT_Focus(Operator):
    bl_label = "Focus"
    bl_idname = "optimize.focus"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        optimize_focus_main()
        bpy.ops.wm.tool_set_by_id(name='builtin.select_box', space_type='VIEW_3D')
        return {'FINISHED'}


class RC_OT_OptimizeSurfaces(Operator):
    bl_label = "Optimize Feature"
    bl_idname = "optimize.optimize_model"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        optimize_surfaces_main()
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
    bl_label = "Select Features"

    def draw(self, context):
        scene = context.scene

        self.layout.label(text='Detection Parameters:')
        self.layout.prop(scene.select_contours_props, "AdjacencyDepth")
        self.layout.prop(scene.select_contours_props, "Proximity")
        self.layout.prop(scene.select_contours_props, "CornerThreshold")
        self.layout.prop(scene.select_contours_props, "ContourThreshold")
        self.layout.separator()
        
        self.layout.operator("optimize.select_features")
        self.layout.separator()


class DivideSurfacesPanel(View3DPanel, bpy.types.Panel):
    bl_idname = "VIEW3D_PT_divide_surfaces"
    bl_label = "Divide Model"

    def draw(self, context):
        scene = context.scene

        self.layout.operator("optimize.divide_model")
        self.layout.separator()


class OptimizeSurfacesPanel(View3DPanel, bpy.types.Panel):
    bl_idname = "VIEW3D_PT_optimize_surfaces"
    bl_label = "Optimize Model"

    def draw(self, context):
        scene = context.scene
        layout = self.layout

        layout.label(text='Features Selection:')
        row = layout.row(align=True)
        row.operator("optimize.select_prev")
        row.operator("optimize.select_next")
        
        self.layout.operator("optimize.focus")
        self.layout.separator()

        self.layout.operator("optimize.optimize_model")
        self.layout.separator()


classes = (
    RC_OT_SelectContours,
    RC_OT_DivideSurfaces,
    RC_OT_OptimizeSurfaces,
    RC_OT_SelectPrevious,
    RC_OT_SelectNext,
    RC_OT_Focus,
    OperatorSelectContoursProperties,
    SelectContoursPanel,
    DivideSurfacesPanel,
    OptimizeSurfacesPanel
)

custom_icons = None


def register():
    from bpy.utils import register_class, register_tool, previews

    for cls in classes:
        register_class(cls)

    bpy.types.Scene.select_contours_props = bpy.props.PointerProperty(type=OperatorSelectContoursProperties)
    bpy.types.Object.iter = bpy.props.IntProperty(name='iter')
    bpy.types.Object.num_elements = bpy.props.IntProperty(name='num_elements')


def unregister():
    from bpy.utils import unregister_class, unregister_tool, previews

    for cls in reversed(classes):
        unregister_class(cls)

    del bpy.types.Scene.select_contours_props
    del bpy.types.Scene.divide_surfaces_props
    del bpy.types.Scene.optimize_surfaces_props
    del bpy.types.Scene.select_elements_props


if __name__ == "__main__":
    register()

    
            
            
                    
            
            
            
            

