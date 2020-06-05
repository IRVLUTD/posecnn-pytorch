import os
import random
import torch
import numpy as np
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender

from pyrender import RenderFlags
from scipy import linalg

import three
from utils import meshutils

CAM_REF_POSE = torch.tensor((
    (1, 0, 0, 0),
    (0, -1, 0, 0),
    (0, 0, -1, 0),
    (0, 0, 0, 1),
), dtype=torch.float32)


OBJ_DEFAULT_POSE = torch.tensor((
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
))


def object_to_camera_pose(object_pose):
    """
    Take an object pose and converts it to a camera pose.

    Takes a matrix that transforms object-space points to camera-space points and converts it
    to a matrix that takes OpenGL camera-space points and converts it into object-space points.
    """
    camera_transform = three.inverse_transform(object_pose)

    # We must flip the z-axis before performing our transformation so that the z-direction is
    # pointing in the correct direction when we feed this as OpenGL coordinates.
    return np.matmul(CAM_REF_POSE.t(), np.matmul(camera_transform, CAM_REF_POSE))


def load_object(path, scale=1.0, size=1.0, recenter=True, resize=True,
                bound_type='diameter', load_materials=False):
    """
    Loads an object model as an Object3D instance.

    Args:
        path: the path to the 3D model
        scale: a scaling factor to apply after all transformations
        size: the reference 'size' of the object if `resize` is True
        recenter: if True the object will be recentered at the centroid
        resize: if True the object will be resized to fit insize a cube of size `size`
        bound_type: how to compute size for resizing. Either 'diameter' or 'extents'

    Returns:
        (meshutils.Object3D): the loaded object model
    """
    obj = meshutils.Object3D(path, load_materials=load_materials)

    if recenter:
        obj.recenter('bounds')

    if resize:
        if bound_type == 'diameter':
            object_scale = size / obj.bounding_diameter
        elif bound_type == 'extents':
            object_scale = size / obj.bounding_size
        else:
            raise ValueError("Unkown size_type {bound_type!r}")

        obj.rescale(object_scale)
    else:
        object_scale = 1.0

    if scale != 1.0:
        obj.rescale(scale)

    return obj, object_scale


def _create_object_node(obj, pose=np.eye(4)):
    smooth = True
    # Turn smooth shading off if vertex normals are unreliable.
    if obj.are_normals_corrupt():
        smooth = False

    mesh = pyrender.Mesh.from_trimesh(obj.meshes, smooth=smooth)
    node = pyrender.Node(mesh=mesh, matrix=pose)
    return node


def get_zbound(distance, scale, eps=0.01):
    znear = max(eps, distance - scale / 2.0 - eps)
    zfar = distance + scale / 2.0 + eps
    return znear, zfar


class SceneContext(object):
    """
    A wrapper class containing all contextual information needed for rendering.
    """

    # intrinsic: torch.Tensor
    def __init__(self, intrinsic):
        self.intrinsic = intrinsic
        self.scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(0.1, 0.1, 0.1))

        fx = self.intrinsic[0, 0].item()
        fy = self.intrinsic[1, 1].item()
        cx = self.intrinsic[0, 2].item()
        cy = self.intrinsic[1, 2].item()
        self.camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        self.camera_pose = np.eye(4)

        # Create lights.
        self.light_nodes = []
        self.extrinsic = None
        self.camera_node = self.scene.add(self.camera, name='camera')

        # objects
        self.objects = []
        self.object_nodes = []

    def add_obj(self, obj, pose=np.eye(4)):
        self.objects.append(obj)
        object_node = _create_object_node(obj, pose)
        self.object_nodes.append(object_node)
        self.scene.add_node(object_node)

    @property
    def object_quaternion(self, ind=0):
        x, y, z, w = self.object_nodes[ind].rotation
        return torch.tensor((w, x, y, z), dtype=torch.float32)

    @property
    def object_translation(self, ind=0):
        return torch.tensor(self.object_nodes[ind].translation, dtype=torch.float32)

    def _update_light_nodes(self, num_lights):
        delta = num_lights - len(self.light_nodes)
        if delta < 0:
            for _ in range(abs(delta)):
                self.scene.remove_node(self.light_nodes.pop())
        elif delta > 0:
            for _ in range(delta):
                light_node = self.scene.add(pyrender.PointLight(color=np.ones(3), intensity=0.0),
                                            pose=np.eye(4), name='point_light')
                self.light_nodes.append(light_node)

    def set_lighting(self, intensity):
        self._update_light_nodes(num_lights=1)
        for node in self.light_nodes:
            node.light.intensity = intensity
            node.light.color = np.ones(3)
            self.scene.set_pose(node, self.camera_pose)

    def randomize_lights(self, min_lights, max_lights, min_dist=1.5, max_dist=3.0,
                         min_intensity=1.0, max_intensity=30.0, random_color=True):
        num_lights = random.randint(min_lights, max_lights)
        self._update_light_nodes(num_lights)

        for node in self.light_nodes:
            # Set light intensity and color.
            intensity = random.uniform(min_intensity, max_intensity)
            node.light.intensity = intensity
            if random_color:
                node.light.color = np.random.uniform(0.2, 1.0, 3)
            else:
                node.light.color = np.ones(3)

            # Set light pose.
            light_dist = random.uniform(min_dist, max_dist)
            light_pose = np.eye(4)
            position = np.random.randn(3)
            light_pose[:3, 3] = light_dist * position / linalg.norm(position)
            self.scene.set_pose(node, light_pose)

    def set_pose(self, translation, quaternion, frame='default'):
        if frame == 'realsense':
            canon_quat = three.quaternion.mat_to_quat(OBJ_DEFAULT_POSE)
            quaternion = quaternion.clone().squeeze()
            quaternion = three.quaternion.qmul(quaternion, canon_quat)

        extrinsic = three.rigid.to_extrinsic_matrix(translation, quaternion)
        self.set_pose_from_extrinsic(extrinsic)

    def set_pose_from_extrinsic(self, extrinsic):
        self.extrinsic = extrinsic
        self.camera_pose = object_to_camera_pose(extrinsic).numpy()
        self.scene.set_pose(self.camera_node, self.camera_pose)

    def set_intrinsic(self, intrinsic):
        self.intrinsic = intrinsic
        self.camera.fx = intrinsic[0, 0].item()
        self.camera.fy = intrinsic[1, 1].item()
        self.camera.cx = intrinsic[0, 2].item()
        self.camera.cy = intrinsic[1, 2].item()


class Renderer:
    """
    A thin wrapper around the PyRender renderer.
    """

    def __init__(self, width, height):
        self._renderer = pyrender.OffscreenRenderer(width, height)
        self._render_flags = RenderFlags.SKIP_CULL_FACES | RenderFlags.RGBA

    @property
    def width(self):
        return self._renderer.viewport_width

    @property
    def height(self):
        return self._renderer.viewport_height

    def __del__(self):
        self._renderer.delete()

    def render(self, context, flags=None):
        if flags is None:
            flags = self._render_flags
        color, depth = self._renderer.render(context.scene, flags=flags)
        color = color.copy().astype(np.float32) / 255.0
        color = torch.tensor(color)
        depth = torch.tensor(depth)
        mask = color[..., 3]
        color = color[..., :3]
        return color.flip(0), depth.flip(0), mask.flip(0)
