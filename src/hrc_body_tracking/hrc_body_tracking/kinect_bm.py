import numpy as np
import torch

from patched_smplx.lbs import (
    batch_rigid_transform,
    batch_rodrigues,
    blend_shapes,
    vertices2joints,
)


def to_t(d, device, dtype=torch.float32):
    return torch.tensor(d, device=device, dtype=dtype)


def to_batch_t(x, device, size=None):
    if x is None:
        if size is None:
            raise Exception()
        x = np.zeros(size)
    return to_t(x, device=device).reshape(1, -1)


def transform_kinect_landmarks(joints, pose, parents, shaped_vertices,
                               skinning_weights, dtype=torch.float32):
    """Apply joint transforms to the 32 Kinect landmark vertices.

    The expanded body model stores the Kinect landmarks as its final 32
    vertices. Pose-corrective blend shapes are intentionally not applied here;
    this fast path only performs forward kinematics followed by weighted rigid
    transformation of those landmark vertices.
    """
    batch_size = pose.shape[0]
    joint_count = parents.numel()

    rotations = batch_rodrigues(
        pose.reshape(-1, 3), dtype=dtype
    ).reshape(batch_size, joint_count, 3, 3)
    posed_joints, joint_transforms = batch_rigid_transform(
        rotations, joints, parents, dtype=dtype
    )

    landmark_vertices = shaped_vertices[:, -32:, :]
    landmark_weights = skinning_weights[-32:, :].unsqueeze(0).expand(
        batch_size, -1, -1
    )
    landmark_transforms = torch.einsum(
        'bvk,bkij->bvij', landmark_weights, joint_transforms
    )

    homogeneous_vertices = torch.cat(
        [landmark_vertices, torch.ones_like(landmark_vertices[..., :1])],
        dim=-1,
    )
    posed_landmarks = torch.einsum(
        'bvij,bvj->bvi', landmark_transforms, homogeneous_vertices
    )[..., :3]

    return posed_landmarks, posed_joints, joint_transforms


class KinectJointsBM:
    def __init__(self, exp_bm_wrapper, gender, betas):
        self.bm = exp_bm_wrapper.pykinect_data[gender]['exp_bm']
        self.device = exp_bm_wrapper.device

        self.betas = to_batch_t(betas, self.device, 10)
        expression = to_batch_t(None, self.device, 10)
        shape_components = torch.cat(
            [self.betas, self.bm.fe_scale * expression], dim=-1)
        shapedirs = torch.cat([self.bm.shapedirs, self.bm.exprdirs], dim=-1)
        self.v_shaped = self.bm.v_template + blend_shapes(
            shape_components, shapedirs)
        self.J = vertices2joints(self.bm.J_regressor, self.v_shaped)

        self.pose_jaw = to_batch_t(None, self.device, 3)
        self.pose_eye = to_batch_t(None, self.device, 6)

        left_hand_pose = self.bm.left_hand_mean.reshape(1, -1).repeat(1, 1)
        right_hand_pose = self.bm.right_hand_mean.reshape(1, -1).repeat(1, 1)
        self.pose_hand = torch.cat([left_hand_pose, right_hand_pose], dim=1)

    def inf(self, pose_body, global_rot, global_trans):
        pose_body = to_batch_t(pose_body, self.device)
        global_rot = to_batch_t(global_rot, self.device)
        global_trans = to_batch_t(global_trans, self.device)

        full_pose = torch.cat(
            [global_rot, pose_body, self.pose_jaw, self.pose_eye, self.pose_hand],
            dim=1)

        verts, joints, A = transform_kinect_landmarks(
            joints=self.J,
            pose=full_pose,
            parents=self.bm.kintree_table[0].long(),
            shaped_vertices=self.v_shaped,
            skinning_weights=self.bm.weights,
            dtype=self.bm.dtype
        )

        verts = verts + global_trans.unsqueeze(dim=1)

        joints = joints + global_trans.unsqueeze(dim=1)

        return verts, joints, A


def kinect_bm_out2kinect_joints(verts, joints, substract_pelvis=False):
    kinect_joints = verts[0, -32:].detach().cpu().numpy()

    if substract_pelvis:
        kinect_joints -= kinect_joints[[0], :]

    smplx_pelvis = joints[0, 0].detach().cpu().numpy()
    return kinect_joints, smplx_pelvis
