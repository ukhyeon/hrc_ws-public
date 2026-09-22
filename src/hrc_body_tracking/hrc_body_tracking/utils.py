
import torch
import torch.nn.functional as F

from . import constants
import numpy as np
import json
from scipy.signal import savgol_filter

AXIS_TO_IDX = {"x": 0, "y": 1, "z": 2}
ALL_AXES = ("x", "y", "z")

SMPL_TO_Q_AXIS_MAP_19 = {
    0:  {"x": "z", "y": "x", "z": "y"},  # pelvis

    1:  {"x": "z", "y": "y", "z": "x"},  # L_hip
    2:  {"x": "z", "y": "y", "z": "x"},  # R_hip
    3:  {"x": "z", "y": "y", "z": "x"},  # spine1
    4:  {"x": "z", "y": "y", "z": "x"},  # L_knee
    5:  {"x": "z", "y": "y", "z": "x"},  # R_knee
    6:  {"x": "z", "y": "y", "z": "x"},  # L_ankle
    7:  {"x": "z", "y": "y", "z": "x"},  # R_ankle
    8:  {"x": "z", "y": "y", "z": "x"},  # spine2
    9:  {"x": "z", "y": "y", "z": "x"},  # neck
    10: {"x": "z", "y": "y", "z": "x"},  # L_collar
    11: {"x": "z", "y": "y", "z": "x"},  # R_collar
    12: {"x": "z", "y": "y", "z": "x"},  # head

    13: {"x": "y", "y": "z", "z": "x"},  # L_shoulder
    14: {"x": "y", "y": "z", "z": "x"},  # R_shoulder

    15: {"x": "z", "y": "x", "z": "y"},  # L_elbow
    16: {"x": "z", "y": "x", "z": "y"},  # R_elbow

    17: {"x": "z", "y": "y", "z": "x"},  # L_wrist
    18: {"x": "z", "y": "y", "z": "x"},  # R_wrist
}
def Rx(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    one = torch.ones_like(c)
    zero = torch.zeros_like(c)
    return torch.stack([
        torch.stack([one,  zero, zero]),
        torch.stack([zero, c,   -s ]),
        torch.stack([zero, s,    c ])
    ], dim=0)

def Ry(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    one = torch.ones_like(c)
    zero = torch.zeros_like(c)
    return torch.stack([
        torch.stack([ c,   zero, s   ]),
        torch.stack([ zero, one, zero]),
        torch.stack([-s,   zero, c   ])
    ], dim=0)

def Rz(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    one = torch.ones_like(c)
    zero = torch.zeros_like(c)
    return torch.stack([
        torch.stack([c,   -s,   zero]),
        torch.stack([s,    c,   zero]),
        torch.stack([zero, zero, one ])
    ], dim=0)

def camera_to_world_rotmat(roll, pitch, yaw, device=None):
    """
    roll, pitch, yaw : degree or tensor (degree)
    Camera → World
    """

    # 🔑 int / float → tensor 변환
    roll  = torch.as_tensor(roll,  dtype=torch.float32, device=device)
    pitch = torch.as_tensor(pitch, dtype=torch.float32, device=device)
    yaw   = torch.as_tensor(yaw,   dtype=torch.float32, device=device)

    # degree → radian
    roll_rad  = torch.deg2rad(roll)
    pitch_rad = torch.deg2rad(pitch)
    yaw_rad   = torch.deg2rad(yaw)

    R_cw = Rx(roll_rad) @ Ry(pitch_rad) @ Rz(yaw_rad)
    return R_cw

def normalize_axes_dict(euler_xyz):
    """
    JSON의 euler_xyz를 현재 코드가 기대하는 형식으로 통일
    기대 출력:
    {
        "x": {"min_deg": ..., "max_deg": ...},
        ...
    }
    """
    out = {}
    for ax, lim in euler_xyz.items():
        if ax not in ALL_AXES:
            continue

        if isinstance(lim, dict):
            if "min_deg" in lim and "max_deg" in lim:
                out[ax] = {
                    "min_deg": float(lim["min_deg"]),
                    "max_deg": float(lim["max_deg"]),
                }
            elif "min" in lim and "max" in lim:
                out[ax] = {
                    "min_deg": float(lim["min"]),
                    "max_deg": float(lim["max"]),
                }
            else:
                raise ValueError(f"Axis '{ax}' must contain min/max or min_deg/max_deg: {lim}")
        else:
            raise ValueError(f"Axis '{ax}' entry must be dict, got: {type(lim)}")

    return out

def load_rom_config_19(
    json_path,
    n_reduced_joints=19,
):
    with open(json_path, "r") as f:
        rom_json = json.load(f)["rom_limits"]

    cfg = {
        "single_joint_rules": {},      # q-type axis로 저장
        "shoulder_sum_rules": {},      # smpl/local axis 그대로 저장
        "structural_lock_axes": {}     # q-type axis 기준
    }

    # structural lock 계산용 free axis는 q-type 기준으로 추적
    free_axes_q = {j: set() for j in range(n_reduced_joints)}

    for name, spec in rom_json.items():
        ctype = spec["constraint_type"]
        axes_smpl = normalize_axes_dict(spec["euler_xyz"])

        if ctype == "single_joint_limit":
            j = int(spec["reduced_smpl_joint_index"])

            # single joint는 q-type axis로 remap해서 저장
            axis_map = SMPL_TO_Q_AXIS_MAP_19[j]
            axes_q = {}
            for smpl_axis, lim in axes_smpl.items():
                q_axis = axis_map[smpl_axis]
                axes_q[q_axis] = lim

            cfg["single_joint_rules"][j] = {
                "name": name,
                "axes": axes_q,           # q-type axis
                "axes_smpl": axes_smpl,   # 디버그용으로 남겨도 좋음
            }
            free_axes_q[j].update(axes_q.keys())

        elif ctype == "combined_sum_limit":
            # 현재 구현은 shoulder special case만 지원
            if name == "left_shoulder":
                sum_joints = [10, 13]   # L_collar + L_shoulder
                j_lock = 13
            elif name == "right_shoulder":
                sum_joints = [11, 14]   # R_collar + R_shoulder
                j_lock = 14
            else:
                continue

            cfg["shoulder_sum_rules"][j_lock] = {
                "name": name,
                "sum_joints": sum_joints,
                "lock_joint": j_lock,
                "axes_smpl": axes_smpl,   # smpl/local axis 그대로 유지
            }

            # structural lock 계산을 위해 lock_joint 입장에서 q-axis 후보를 추가
            axis_map_lock = SMPL_TO_Q_AXIS_MAP_19[j_lock]
            q_axes_for_lock = [axis_map_lock[smpl_axis] for smpl_axis in axes_smpl.keys()]
            free_axes_q[j_lock].update(q_axes_for_lock)

    # 정의 안 된 축은 structural lock (q-type axis 기준)
    for j in range(n_reduced_joints):
        missing_axes = [ax for ax in ALL_AXES if ax not in free_axes_q[j]]
        cfg["structural_lock_axes"][j] = missing_axes

    return cfg

def wrap_to_pi_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))

# 각도를 -pi~pi 정규화
def remove_singlular_batch(pred_q):
    pred_q_updated = pred_q.clone()
    pred_q_updated[:, :, 3:] = torch.remainder(pred_q_updated[:, :, 3:], 2*np.pi)
    pred_q_updated[:, :, 3:][pred_q_updated[:, :, 3:]>np.pi] = pred_q_updated[:, :, 3:][pred_q_updated[:, :, 3:]>np.pi] - 2*np.pi
    return pred_q_updated

# 속도를 구함 차분
def compute_velocity_batch(pred_q, dt):
    pred_q_dot = pred_q[:, 1:] - pred_q[:, :-1]
    pred_q_dot[pred_q_dot>np.pi] = pred_q_dot[pred_q_dot>np.pi] - 2*np.pi
    pred_q_dot[pred_q_dot<-np.pi] = pred_q_dot[pred_q_dot<-np.pi] + 2*np.pi
    return pred_q_dot / dt

def compute_acceleration_batch(pred_q_dot, dt):
    return (pred_q_dot[:, 1:] - pred_q_dot[:, :-1]) / dt


# 포즈 예측에서 속도와 가속도를 기반으로 매끄러움 제약을 계산; 가속도의 변화량을 기준으로 손실을 구함 
def smoothness_constraint(pred_x, dt):
    pred_v = compute_velocity_batch(pred_x, dt)
    pred_a = compute_acceleration_batch(pred_v, dt)
    loss_smoothness = ((pred_a[:, 1:] - pred_a[:, :-1])**2).mean()
    return pred_v, pred_a, loss_smoothness

# 관절의 회전 행렬과 초기 위치, 관절관계를 기반으로 전역 3D 위치를 계산
def batch_global_rigid_transformation(Rs, Js, parent):
    """
    Computes 3D joint locations given pose. J_child = A_parent * A_child[:, :, :3, 3]
    Args:
      Rs: N x 24 x 3 x 3, rotation matrix of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24, holding the parent id for each joint
    Returns
      J_transformed : N x 24 x 3 location of absolute joints
      A_relative: N x 24 4 x 4 relative transformation matrix for LBS.
    """
    def make_A(R, t, N):
        """
        construct transformation matrix for a joint
            Args: 
                R: N x 3 x 3, rotation matrix 
                t: N x 3 x 1, bone vector (child-parent)
            Returns:
                A: N x 4 x 4, transformation matrix
        """
        # N x 4 x 3
        R_homo = F.pad(R, (0,0,0,1))
        # N x 4 x 1
        t_homo = torch.cat([t, torch.ones([N, 1, 1]).type(torch.float32).to(R.device)], 1)
        # N x 4 x 4
        return torch.cat([R_homo, t_homo], 2)
    
    # obtain the batch size
    N = Rs.size()[0]
    # unsqueeze Js to N x 24 x 3 x 1
    Js = Js.unsqueeze(-1)
    
    root_rotation = Rs[:, 0, :, :]
    # transformation matrix of the root
    A0 = make_A(root_rotation, Js[:, 0], N)
    A = [A0]
    # caculate transformed matrix of each joint
    for i in range(1, parent.shape[0]):
        # transformation matrix
        t_here = Js[:,i] - Js[:,parent[i]]
        A_here = make_A(Rs[:,i], t_here, N)
        # transformation given parent matrix
        A_here_tran = torch.matmul(A[parent[i]], A_here)
        A.append(A_here_tran)

    # N x 24 x 4 x 4, transformation matrix for each joint
    A = torch.stack(A, dim=1)
    # recover transformed joints from the transformed transformation matrix
    J_transformed = A[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    # N x 24 x 3 x 1 to N x 24 x 4 x 1, homo with zeros
    Js_homo = torch.cat([Js, torch.zeros([N, 24, 1, 1]).type(torch.float32).to(Rs.device)], 2)
    # N x 24 x 4 x 1
    init_bone = torch.matmul(A, Js_homo)
    # N x 24 x 4 x 4, For each 4 x 4, last column is the joints position, and otherwise 0. 
    init_bone = F.pad(init_bone, (3,0))
    A_relative = A - init_bone
    return J_transformed, A_relative

# 로드리게스를 사용하여 theta를 회전행렬로 변환 

def batch_rodrigues(theta):
    """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.
    Args:
        theta: [N, 1, 3]
    Return
        R: [N, 3, 3]
    """
    N = theta.size()[0]
    
    # obtain the angle by taking the norm and expand dimension to [N,1,1]
    angle = torch.norm(theta + 1e-8, dim=2, keepdim=True)
    # obtain the rotation axis [N,1,3]
    r_hat = torch.div(theta, angle)

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    K2 = torch.matmul(r_hat.transpose(2,1), r_hat)
    I = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(angle.device)
    # obtain Skew_sym matrix of r_hat [N,3,3]
    zeros = torch.zeros(N, dtype=torch.float32).to(angle.device)
    batch_skew = torch.stack(
      (zeros, -r_hat[:,0,2], r_hat[:,0,1], r_hat[:,0,2], zeros,
       -r_hat[:,0,0], -r_hat[:, 0, 1], r_hat[:, 0, 0], zeros), dim=1).view((-1, 3, 3))
        
    R = cos * I + (1 - cos) * K2 + sin * batch_skew
    return R

# 오일러 각도를 다양한 회전 순서에 따라 회전 행렬로 변환 
def batch_euler2mat(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    rotMat_individual = torch.stack([xmat, ymat, zmat], dim=1)
    return rotMat, rotMat_individual


def batch_euler2matyxz(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = zmat @ xmat @ ymat
    rotMat_individual = torch.stack([xmat, ymat, zmat], dim=1)
    return rotMat, rotMat_individual

def batch_euler2matzxy(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = ymat @ xmat @ zmat
    rotMat_individual = torch.stack([zmat, xmat, ymat], dim=1)
    return rotMat, rotMat_individual

def batch_euler2matzyx(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    rotMat_individual = torch.stack([zmat, ymat, xmat], dim=1)
    return rotMat, rotMat_individual


def batch_euler2matyzx(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ zmat @ ymat
    rotMat_individual = torch.stack([ymat, zmat, xmat], dim=1)
    return rotMat, rotMat_individual

def batch_roteulerSMPL(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 72], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)

    rotMat_root, rotMat_root_individual = batch_euler2matyzx(angle[:,:3].reshape(-1, 3))
    rotMat_s, rotMat_s_individual = batch_euler2matzyx(angle[:,3:48].reshape(-1, 3))
    rotMat_shoulder, rotMat_shoulder_individual = batch_euler2matzxy(angle[:,48:54].reshape(-1, 3))
    rotMat_elbow, rotMat_elbow_individual = batch_euler2matyzx(angle[:,54:60].reshape(-1, 3))
    rotMat_e, rotMat_e_individual = batch_euler2matzyx(angle[:,60:].reshape(-1, 3))

    rotMat_root = rotMat_root.reshape(B, 1, 3, 3)
    rotMat_s = rotMat_s.reshape(B, 15, 3, 3)
    rotMat_shoulder = rotMat_shoulder.reshape(B, 2, 3, 3)
    rotMat_elbow = rotMat_elbow.reshape(B, 2, 3, 3)
    rotMat_e = rotMat_e.reshape(B, 4, 3, 3)
    rotMat = torch.cat((rotMat_root, rotMat_s, rotMat_shoulder, rotMat_elbow, rotMat_e), dim=1)

    rotMat_root_individual = rotMat_root_individual.reshape(B, 1, 3, 3, 3)
    rotMat_s_individual = rotMat_s_individual.reshape(B, 15, 3, 3, 3)
    rotMat_shoulder_individual = rotMat_shoulder_individual.reshape(B, 2, 3, 3, 3)
    rotMat_elbow_individual = rotMat_elbow_individual.reshape(B, 2, 3, 3, 3)
    rotMat_e_individual = rotMat_e_individual.reshape(B, 4, 3, 3, 3)
    rotMat_individual = torch.cat(( rotMat_root_individual, 
                                    rotMat_s_individual, 
                                    rotMat_shoulder_individual,
                                    rotMat_elbow_individual,
                                    rotMat_e_individual), 
                                dim=1).reshape(B, -1, 3, 3)
    return rotMat, rotMat_individual

def batch_mat2euler(rotmat):
    ''' Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (N,3,3)
    Returns
    -------
    angle : (N,3)
       Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)
    Problems arise when cos(y) is close to zero, because both of::
       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    N = rotmat.size()[0]
    cy_thresh = torch.ones(N).type(torch.float32).to(rotmat.device) * 1e-8

    eulerangle = torch.zeros(N,3).type(torch.float32).to(rotmat.device)
    
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = torch.flatten(rotmat, start_dim=1)
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = torch.sqrt(r33*r33 + r23*r23)

    eulerangle[cy>cy_thresh,2] = torch.atan2(-r12[cy>cy_thresh], r11[cy>cy_thresh])
    eulerangle[cy>cy_thresh,1] = torch.atan2( r13[cy>cy_thresh],  cy[cy>cy_thresh])
    eulerangle[cy>cy_thresh,0] = torch.atan2(-r23[cy>cy_thresh], r33[cy>cy_thresh])

    eulerangle[cy<=cy_thresh,2] = torch.atan2( r21[cy<=cy_thresh], r22[cy<=cy_thresh])
    eulerangle[cy<=cy_thresh,1] = torch.atan2( r13[cy<=cy_thresh],  cy[cy<=cy_thresh])

    return eulerangle

def keypoint_3d_loss(criterion, pred_keypoints_3d, gt_keypoints_3d):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    gt_keypoints_3d = gt_keypoints_3d - gt_keypoints_3d[:, 0:1, :]
    pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, 0:1, :]
    
    return criterion(pred_keypoints_3d*100, gt_keypoints_3d*100).mean()

def batch_global_rigid_transformation(Rs, Js, parent):
    """
    Computes 3D joint locations given pose. J_child = A_parent * A_child[:, :, :3, 3]
    Args:
      Rs: N x 24 x 3 x 3, rotation matrix of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24, holding the parent id for each joint
    Returns
      J_transformed : N x 24 x 3 location of absolute joints
      A_relative: N x 24 4 x 4 relative transformation matrix for LBS.
    """
    def make_A(R, t, N):
        """
        construct transformation matrix for a joint
            Args: 
                R: N x 3 x 3, rotation matrix 
                t: N x 3 x 1, bone vector (child-parent)
            Returns:
                A: N x 4 x 4, transformation matrix
        """
        # N x 4 x 3
        R_homo = F.pad(R, (0,0,0,1))
        # N x 4 x 1
        t_homo = torch.cat([t, torch.ones([N, 1, 1]).type(torch.float32).to(R.device)], 1)
        # N x 4 x 4
        return torch.cat([R_homo, t_homo], 2)
    
    # obtain the batch size
    N = Rs.size()[0]
    # unsqueeze Js to N x 24 x 3 x 1
    Js = Js.unsqueeze(-1)
    
    root_rotation = Rs[:, 0, :, :]
    # transformation matrix of the root
    A0 = make_A(root_rotation, Js[:, 0], N)
    A = [A0]
    # caculate transformed matrix of each joint
    for i in range(1, parent.shape[0]):
        # transformation matrix
        t_here = Js[:,i] - Js[:,parent[i]]
        A_here = make_A(Rs[:,i], t_here, N)
        # transformation given parent matrix
        A_here_tran = torch.matmul(A[parent[i]], A_here)
        A.append(A_here_tran)

    # N x 24 x 4 x 4, transformation matrix for each joint
    A = torch.stack(A, dim=1)
    # recover transformed joints from the transformed transformation matrix
    J_transformed = A[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    # N x 24 x 3 x 1 to N x 24 x 4 x 1, homo with zeros
    Js_homo = torch.cat([Js, torch.zeros([N, 24, 1, 1]).type(torch.float32).to(Rs.device)], 2)
    # N x 24 x 4 x 1
    init_bone = torch.matmul(A, Js_homo)
    # N x 24 x 4 x 4, For each 4 x 4, last column is the joints position, and otherwise 0. 
    init_bone = F.pad(init_bone, (3,0))
    A_relative = A - init_bone
    return J_transformed, A_relative

def write_obj(model, verts, output_filename):
    """
    Save verts to obj file given the define in model object
    Args:
      model: SMPL class object
      verts: 6890 x 3, 3D vertices position
      output_filename: output file path and name
    """    
    with open(output_filename, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in model.faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def align_by_pelvis(joints, root):
    """
    Root alignments, i.e. subtracts the root.
    Args:
        joints: is N x 3
        roots: index of root joints
    """
    hip_id = 0 # 루트 관절
    pelvis = joints[hip_id, :] # 루트 관절의 3D 좌표
    
    return joints - np.expand_dims(pelvis, axis=0)

def compute_errors(gt3ds, preds, root):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 17 common joints.
    3번 관절부터 21번째 관절까지 
    Inputs:
      - gt3ds: N x J x 3
      - preds: N x J x 3
      - root: root index for alignment
    """
    perjerrors, errors, perjerrors_pa, errors_pa = [], [], [], []
    for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
        gt3d = gt3d.reshape(-1, 3)
        # Root align.
        gt3d = align_by_pelvis(gt3d, root)
        pred3d = align_by_pelvis(pred, root)


        joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
        
        errors.append(np.mean(joint_error) * 1000)
        perjerrors.append(joint_error * 1000)

    return perjerrors, errors


def compute_error_accel(joints_gt, joints_pred, root=0):
    """
    Computes acceleration error:
        1/(n-2) sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (NxJx3).
        joints_pred (NxJx3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)xJx3
    joints_gt = align_by_pelvis(joints_gt, 0)
    joints_pred = align_by_pelvis(joints_pred, 0)


    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    return np.linalg.norm(accel_pred - accel_gt, axis=2)
    
def compute_error_accel_T(joints_gt, joints_pred, T, root=0):
    """
    Computes acceleration error:
        1/(n-2) sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (NxTxJx3).;  N*50, 19 ,3 
        joints_pred (NxTxJx3).
        vis (N).
        T: Total Length
    Returns:
        error_accel (N-2).
    """
    # (N-2)xJx3
    _, J, _ = joints_gt.shape
    joints_gt = align_by_pelvis(joints_gt, root)
    joints_pred = align_by_pelvis(joints_pred, root)

    joints_gt = joints_gt.reshape([-1, T, J, 3])  # N, T, J, 3 
    joints_pred = joints_pred.reshape([-1, T, J, 3])

    accel_gt = joints_gt[:, :-2] - 2 * joints_gt[:, 1:-1] + joints_gt[:, 2:]
    accel_pred = joints_pred[:, :-2] - 2 * joints_pred[:, 1:-1] + joints_pred[:, 2:]

    accel_error = np.mean(np.linalg.norm(accel_pred - accel_gt, axis=3), axis=2) # x,y,z 좌표의 유클리디안 거리 계산 -> 1,50,19; 관절에 대한 평균 -> 1,50 = 각 프레임에 대해 관절의 평균 가속도
    accel_gt = np.mean(np.mean(accel_gt,axis=3),2) # x,y,z 에 대한 평균 -> 1, 50, 19 -> 관절에 대한 평균 -> 1, 50 
    accel_pred = np.mean(np.mean(accel_pred,axis=3), 2)

    return accel_error # N, T-2 ; 각 샘플에 대해 프레임별 가속도 오차

def compute_error_accel_T_gt_pred(joints_gt, joints_pred, T, root=0):
    """
    Computes acceleration error:
        1/(n-2) sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (NxTxJx3).;  N*50, 19 ,3 
        joints_pred (NxTxJx3).
        vis (N).
        T: Total Length
    Returns:
        error_accel (N-2).
    """
    # (N-2)xJx3
    _, J, _ = joints_gt.shape
    joints_gt = align_by_pelvis(joints_gt, root)
    joints_pred = align_by_pelvis(joints_pred, root)

    joints_gt = joints_gt.reshape([-1, T, J, 3])  # N, T, J, 3 
    joints_pred = joints_pred.reshape([-1, T, J, 3])

    accel_gt = joints_gt[:, :-2] - 2 * joints_gt[:, 1:-1] + joints_gt[:, 2:]
    accel_pred = joints_pred[:, :-2] - 2 * joints_pred[:, 1:-1] + joints_pred[:, 2:]

    accel_error = np.mean(np.linalg.norm(accel_pred - accel_gt, axis=3), axis=2) # x,y,z 좌표의 유클리디안 거리 계산 -> 1,50,19; 관절에 대한 평균 -> 1,50 = 각 프레임에 대해 관절의 평균 가속도
    accel_gt = np.mean(np.mean(accel_gt,axis=3), 2) # x,y,z 에 대한 평균 -> 1, 50, 19 -> 관절에 대한 평균 -> 1, 50 
    accel_pred = np.mean(np.mean(accel_pred,axis=3), 2)

    return accel_error, accel_gt, accel_pred  # N, T-2 ; 각 샘플에 대해 프레임별 가속도 오차

def compute_q_vel(q):
    N,J,_ =q.shape
    q_vel = np.zeros((N,J,3))

    q_vel[1:-1] = (q[2:] - q[:-2]) / (2* constants.dt)

    # 개선된 전방 차분법 (첫 프레임)
    q_vel[0] = (-q[3] + 6*q[2] - 18*q[1] + 10*q[0]) / 12  # 4계 전방 차분법

    # 개선된 후방 차분법 (마지막 프레임)
    q_vel[-1] = (10*q[-1] - 18*q[-2] + 6*q[-3] - q[-4]) / 12  # 4계 후방 차분법


    q_vel = q_vel * constants.m2mm
    return q_vel

def runge_kutta_velocity(accel,dt=1):
    """
    4차 룽게-쿠타(RK4) 방법을 사용하여 속도 계산
    accel: (N, J, 3) 형태의 가속도 데이터 (프레임, 관절 개수, 3D 좌표)
    dt: 시간 간격
    """
    N, J, _ = accel.shape
    print(N)
    velocity = np.zeros((N, J, 3))

    for i in range(1, N):
        k1 = accel[i-1] * dt
        k2 = (accel[i-1] + k1 / 2) * dt
        k3 = (accel[i-1] + k2 / 2) * dt
        k4 = (accel[i-1] + k3) * dt
        velocity[i] = velocity[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6  # RK4 적용

    return velocity


def compute_q_vel_savgol(q):
    """
    프레임 기반 속도(q')를 Savitzky-Golay 필터를 사용하여 계산하는 함수.

    Args:
        q (numpy.ndarray): (N, J, 3) 형태의 모션 데이터
                           N: 프레임 개수, J: 조인트 개수, 3: 각도 값(x, y, z)

    Returns:
        q_vel (numpy.ndarray): (N, J, 3) 형태의 속도 데이터
    """
    N, J, _ = q.shape
    q_vel = np.zeros((N, J, 3))

    # 각 조인트마다 Savitzky-Golay 필터 적용하여 미분
    for j in range(J):
        for axis in range(3):
            q_vel[:, j, axis] = savgol_filter(q[:, j, axis], window_length=5, polyorder=2, deriv=1, mode='nearest')

    return q_vel*constants.m2mm

def compute_accel(joints_gt, T, root=0):
    """
    Computes acceleration error:
        1/(n-2) sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (NxTxJx3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)xJx3
    _, J, _ = joints_gt.shape
    joints_gt = align_by_pelvis(joints_gt, root)
    joints_gt = joints_gt.reshape([-1, T, J, 3])
    

    accel_gt = joints_gt[:, :-2] - 2 * joints_gt[:, 1:-1] + joints_gt[:, 2:]
   

    accel_error = np.array(accel_gt)

    return accel_error

def rotmat2expmap(R):
    """
    :param R: Rotation matrix, Nx3x3
    :return: r: Rotation vector, Nx3
    """
    assert R.shape[1] == R.shape[2] == 3
    theta = torch.acos(torch.clamp((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2, min=-1., max=1.)).view(-1, 1)
    r = torch.stack((R[:, 2, 1]-R[:, 1, 2], R[:, 0, 2]-R[:, 2, 0], R[:, 1, 0]-R[:, 0, 1]), 1) / (2*torch.sin(theta))
    r_norm = r / torch.sqrt(torch.sum(torch.pow(r, 2), 1, keepdim=True))
    return theta * r_norm

def rotmat2eulerzyx(rotmat):
    ''' Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (N,3,3)
    Returns
    -------
    angle : (N,3)
       Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)
    Problems arise when cos(y) is close to zero, because both of::
       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    N = rotmat.size()[0]
    cy_thresh = torch.ones(N).type(torch.float32).to(rotmat.device) * 1e-8

    eulerangle = torch.zeros(N,3).type(torch.float32).to(rotmat.device)
    
    r11, r12, r13 = rotmat[:,0,0],rotmat[:,0,1],rotmat[:,0,2]
    r21, r22, r23 = rotmat[:,1,0],rotmat[:,1,1],rotmat[:,1,2]
    r31, r32, r33 = rotmat[:,2,0],rotmat[:,2,1],rotmat[:,2,2]
     
    # cy: sqrt((cos(y)*sin(x))**2 + (cos(x)*cos(y))**2) = cos(y)
    cy = torch.sqrt(r33*r33 + r23*r23)
    # y = atan(sin(y),con(y))
    eulerangle[:,1] = torch.atan2( r13,  cy) # [-pi,pi]

    # c>cy_thresh
    # -cos(y)*sin(z) / cos(y)*cos(z) = tanz, z = atan(sin(z),con(z))
    eulerangle[cy>cy_thresh,2] = torch.atan2(-r12[cy>cy_thresh], r11[cy>cy_thresh])
    # -cos(y)*sin(x)] / cos(y)*cos(x) = tanx, x = atan(sin(x),con(x))
    eulerangle[cy>cy_thresh,0] = torch.atan2(-r23[cy>cy_thresh], r33[cy>cy_thresh])

    # cy<=cy_thresh
    # r21 = sin(z), r22 = cos(z)
    eulerangle[cy<=cy_thresh,2] = torch.atan2( r21[cy<=cy_thresh], r22[cy<=cy_thresh])

    return eulerangle

def rotmat2euleryzx(rotmat):
    ''' Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (N,3,3)
    Returns
    -------
    angle : (N,3)
       Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
      [                       cos(y)*cos(z),      - sin(z),                         cos(z)sin(y)],
      [sin(x)*sin(y) + cos(x)*cos(y)*sin(z), cos(x)*cos(z), cos(x)*sin(y)*sin(z) - cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x), cos(x)*cos(y) + sin(x)*sin(y)*sin(z)]
    with the obvious derivations for z, y, and x
       z = asin(r12)
       y = atan2(r13, r11)
       x = atan2(r32, r22)
    Problems arise when cos(z) is close to zero, because both of::
    '''
    N = rotmat.size()[0]
    cz_thresh = torch.ones(N).type(torch.float32).to(rotmat.device) * 1e-8

    eulerangle = torch.zeros(N,3).type(torch.float32).to(rotmat.device)
    
    r11, r12, r13 = rotmat[:,0,0],rotmat[:,0,1],rotmat[:,0,2]
    r21, r22, r23 = rotmat[:,1,0],rotmat[:,1,1],rotmat[:,1,2]
    r31, r32, r33 = rotmat[:,2,0],rotmat[:,2,1],rotmat[:,2,2]
     
    # cz: sqrt((cos(y)*cos(z))**2 + (cos(z)sin(y))**2) = cos(y)
    cz = torch.sqrt(r11*r11 + r13*r13)
    # z = atan(sin(z),con(z))
    eulerangle[:,2] = torch.atan2( -r12,  cz) # [-pi,pi]

    # c>cz_thresh
    eulerangle[cz>cz_thresh,1] = torch.atan2(r13[cz>cz_thresh], r11[cz>cz_thresh])
    eulerangle[cz>cz_thresh,0] = torch.atan2(r32[cz>cz_thresh], r22[cz>cz_thresh])

    # cy<=cy_thresh
    eulerangle[cz<=cz_thresh,0] = torch.atan2(-r23[cz<=cz_thresh], r33[cz<=cz_thresh])

    return eulerangle

def rotmat2eulerzxy(rotmat):
    ''' Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (N,3,3)
    Returns
    -------
    angle : (N,3)
       Rotations in radians around z, x, y axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
      [cos(y)*cos(z) + sin(x)*sin(y)*sin(z), cos(z)*sin(x)*sin(y) - cos(y)*sin(z), cos(x)*sin(y)],
      [                       cos(x)*sin(z),                        cos(x)*cos(z),      - sin(x)],
      [cos(y)*sin(x)*sin(z) - cos(z)*sin(y), sin(y)*sin(z) + cos(y)*cos(z)*sin(x), cos(x)*cos(y)]
    '''
    N = rotmat.size()[0]
    cx_thresh = torch.ones(N).type(torch.float32).to(rotmat.device) * 1e-8

    eulerangle = torch.zeros(N,3).type(torch.float32).to(rotmat.device)
    
    r11, r12, r13 = rotmat[:,0,0],rotmat[:,0,1],rotmat[:,0,2]
    r21, r22, r23 = rotmat[:,1,0],rotmat[:,1,1],rotmat[:,1,2]
    r31, r32, r33 = rotmat[:,2,0],rotmat[:,2,1],rotmat[:,2,2]
     
    # cx
    cx = torch.sqrt(r21*r21 + r22*r22)
    # x = atan(sin(x),con(x))
    eulerangle[:,0] = torch.atan2( -r23,  cx) # [-pi,pi]

    # c>cx_thresh
    eulerangle[cx>cx_thresh,2] = torch.atan2(r21[cx>cx_thresh], r22[cx>cx_thresh])
    eulerangle[cx>cx_thresh,1] = torch.atan2(r13[cx>cx_thresh], r33[cx>cx_thresh])

    # cy<=cy_thresh
    eulerangle[cx<=cx_thresh,2] = torch.atan2(-r12[cx<=cx_thresh], r11[cx<=cx_thresh])

    return eulerangle