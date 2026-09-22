from __future__ import division

import pickle
from collections import namedtuple

import numpy as np
import torch




from .utils import batch_global_rigid_transformation



ModelOutput = namedtuple('ModelOutput',
                         ['v_shaped', 'joint_smpl_shaped', 'vertices', 'joints', 'joints_smpl', 'physics_parameters'
                          ])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)

# SMPL_MODEL_PATH = '/home/robotics/hrc_ws/src/body_models/smpl/SMPL_NEUTRAL.pkl'
SMPL_MODEL_PATH = '/home/robotics/python/physpt/assets/data/smpl/neutral.pkl'
SMPLH_M_PATH = "/home/robotics/hrc_ws/src/body_models/smplh/male.npz"
SMPLH_F_PATH = "/home/robotics/hrc_ws/src/body_models/smplh/female.npz"
SMPLH_N_PATH = "/home/robotics/hrc_ws/src/body_models/smplh/neutral.npz"



AXIS_TO_IDX = {"x": 0, "y": 1, "z": 2}
ALL_AXES = ("x", "y", "z")
IDX_TO_AXIS = {0: "x", 1: "y", 2: "z"}
CHAIN_GROUPS_19 = [
    [0, 3, 8, 9, 12],           # torso / neck / head chain
    [0, 3, 8, 10, 13, 15, 17],  # left arm chain
    [0, 3, 8, 11, 14, 16, 18],  # right arm chain
    [0, 1, 4, 6],               # left leg chain
    [0, 2, 5, 7],               # right leg chain
]
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


def apply_sequential_rom_projector_qddot_contact_free(
    Minv_contact,
    J_p,
    u_target,
    pose_19_euler_deg,
    rom_cfg,
    chain_joints,
    structural_locked_cols=None,
    base_dofs=3,
    dof_per_joint=3,
    tol_deg=10.0,
    lam=1e-5,
    j_eval=None,
):
    """
    Contact-constrained free-acceleration 기반 sequential ROM lock.

    이론:
        qddot_dir = Minv_current @ J_c^T @ u

    여기서 J_c는 실제 충돌 target j_eval의 선형 Jacobian이다.
    ROM 후보 관절은 chain_joints를 부모->자식 순서로 검사한다.

    Parameters
    ----------
    Minv_contact : torch.Tensor
        (B, D, D). structural lock + foot contact가 이미 반영된 inverse mass.
    J_p : torch.Tensor
        (B, J, 3, D)
    u_target : torch.Tensor
        (B, 3). j_eval에 대한 충돌 방향 단위벡터.
    pose_19_euler_deg : torch.Tensor
        (B, 19, 3)
    rom_cfg : dict
        기존 ROM config.
    chain_joints : list[int]
        부모->자식 순서의 chain.
    structural_locked_cols : list[int] or None
        이미 structural lock으로 잠긴 col. ROM lock 중복 방지용.
    j_eval : int
        실제 충돌 target index. qddot 계산에 J_p[:, j_eval] 사용.

    Returns
    -------
    Minv_rom : torch.Tensor
        ROM sequential lock까지 반영된 inverse mass.
    rom_locked_cols_per_batch : list[list[int]]
        batch별 ROM lock col.
    rom_debug_steps : list[dict]
        각 chain joint별 debug 정보.
    """
    B, D, _ = Minv_contact.shape
    _, J, _, Dj = J_p.shape
    assert D == Dj

    device = Minv_contact.device
    dtype = Minv_contact.dtype

    if structural_locked_cols is None:
        structural_locked_set = set()
    else:
        structural_locked_set = set(int(c) for c in structural_locked_cols)

    if j_eval is None:
        raise ValueError("j_eval must be provided because qddot_dir uses J_p[:, j_eval].")

    j_eval = int(j_eval)
    if j_eval < 0 or j_eval >= J:
        raise ValueError(f"Invalid j_eval={j_eval}, J={J}")

    # 현재 inverse mass. 부모 관절 lock이 생기면 이 값이 계속 갱신됨.
    Minv_current = Minv_contact

    # batch별 ROM lock col 누적
    rom_locked_cols_per_batch = [[] for _ in range(B)]

    # ROM near 정보는 chain 전체 기준으로 한 번 구함
    rom_near_info = get_rom_near_axes_for_candidate_joints(
        pose_19_euler_deg=pose_19_euler_deg,
        rom_cfg=rom_cfg,
        candidate_joints=chain_joints,
        tol_deg=tol_deg,
    )

    rom_debug_steps = []

    # 실제 충돌 target Jacobian
    # J_collision: (B, 3, D)
    J_collision = J_p[:, j_eval]

    # target force direction -> generalized force direction
    # tau_collision = J_c^T u
    tau_collision = torch.matmul(
        J_collision.transpose(1, 2),
        u_target.unsqueeze(-1)
    ).squeeze(-1)  # (B, D)

    for j_lock in chain_joints:
        j_lock = int(j_lock)

        # 이 joint가 ROM 근처가 아니면 qddot 계산 없이 skip 가능
        if j_lock not in rom_near_info:
            rom_debug_steps.append({
                "joint": j_lock,
                "has_rom_near": False,
                "new_locked_cols_per_batch": [[] for _ in range(B)],
                "qddot_dir": None,
            })
            continue

        # 현재 constrained inverse mass에서 free acceleration 계산
        # qddot_dir = Minv_current @ J_c^T u
        qddot_dir = torch.matmul(
            Minv_current,
            tau_collision.unsqueeze(-1)
        ).squeeze(-1)  # (B, D)

        new_locked_cols_per_batch = [[] for _ in range(B)]

        info = rom_near_info[j_lock]

        # --------------------------------------------------
        # 1) single joint ROM rule
        # --------------------------------------------------
        if "single" in info:
            for q_axis, near_info in info["single"].items():
                col = dof_col(
                    j_lock,
                    q_axis,
                    base_dofs=base_dofs,
                    dof_per_joint=dof_per_joint
                )

                # 이미 structural lock이면 ROM lock으로 다시 추가하지 않음
                if col in structural_locked_set:
                    continue

                qddot_col = qddot_dir[:, col]

                lock_upper = near_info["near_upper"] & (qddot_col > 0.0)
                lock_lower = near_info["near_lower"] & (qddot_col < 0.0)
                lock = lock_upper | lock_lower

                for b in range(B):
                    if bool(lock[b].detach().cpu().item()):
                        if col not in rom_locked_cols_per_batch[b]:
                            new_locked_cols_per_batch[b].append(int(col))

        # --------------------------------------------------
        # 2) shoulder sum ROM rule
        # --------------------------------------------------
        if "sum" in info:
            for smpl_axis, near_info in info["sum"].items():
                q_axis_lock = near_info["q_axis_lock"]

                col = dof_col(
                    j_lock,
                    q_axis_lock,
                    base_dofs=base_dofs,
                    dof_per_joint=dof_per_joint
                )

                if col in structural_locked_set:
                    continue

                qddot_col = qddot_dir[:, col]

                lock_upper = near_info["near_upper"] & (qddot_col > 0.0)
                lock_lower = near_info["near_lower"] & (qddot_col < 0.0)
                lock = lock_upper | lock_lower

                for b in range(B):
                    if bool(lock[b].detach().cpu().item()):
                        if col not in rom_locked_cols_per_batch[b]:
                            new_locked_cols_per_batch[b].append(int(col))

        # 중복 제거
        for b in range(B):
            new_locked_cols_per_batch[b] = sorted(set(new_locked_cols_per_batch[b]))

        # debug 저장
        rom_debug_steps.append({
            "joint": j_lock,
            "has_rom_near": True,
            "rom_near_info": info,
            "qddot_dir": qddot_dir,
            "new_locked_cols_per_batch": new_locked_cols_per_batch,
            "Minv_before": Minv_current,
        })

        # 이 joint에서 새롭게 lock된 DOF가 있으면 즉시 projector 적용
        has_new_lock = any(len(cols) > 0 for cols in new_locked_cols_per_batch)

        if has_new_lock:
            for b in range(B):
                for col in new_locked_cols_per_batch[b]:
                    if col not in rom_locked_cols_per_batch[b]:
                        rom_locked_cols_per_batch[b].append(int(col))

                rom_locked_cols_per_batch[b] = sorted(set(rom_locked_cols_per_batch[b]))

            Minv_current = apply_joint_lock_projector_from_cols(
                Minv=Minv_current,
                locked_cols_per_batch=new_locked_cols_per_batch,
                lam=lam,
            )

            rom_debug_steps[-1]["Minv_after"] = Minv_current
        else:
            rom_debug_steps[-1]["Minv_after"] = Minv_current

    # 최종 중복 제거
    for b in range(B):
        rom_locked_cols_per_batch[b] = sorted(set(rom_locked_cols_per_batch[b]))

    return Minv_current, rom_locked_cols_per_batch, rom_near_info, rom_debug_steps


def get_ordered_chain_from_chain_groups(j_eval, chain_groups=CHAIN_GROUPS_19):
    """
    j_eval이 속한 CHAIN_GROUPS_19를 부모->자식 순서 그대로 반환.

    여러 chain에 동시에 속할 수 있는 pelvis(0), spine 계열은
    j_eval이 들어간 첫 번째 chain을 반환한다.

    예:
        j_eval=1 -> [0, 1, 4, 6]
        j_eval=4 -> [0, 1, 4, 6]
        j_eval=6 -> [0, 1, 4, 6]
    """
    j_eval = int(j_eval)

    for chain in chain_groups:
        if j_eval in chain:
            return list(chain)

    return [j_eval]

def apply_contact_projector_from_indices(
    Minv,
    J_p,
    contact_J_index,
    lam=1e-5,
):
    """
    Foot/contact constraint projector.

    Parameters
    ----------
    Minv : torch.Tensor
        (B, D, D)
    J_p : torch.Tensor
        (B, J, 3, D)
    contact_J_index : list[int]
        contact로 고정할 body/link index.
        예: [6, 7] = L_ankle/foot, R_ankle/foot
    lam : float
        pseudo inverse regularization.

    Returns
    -------
    Minv_contact : torch.Tensor
        (B, D, D)
    Jc : torch.Tensor or None
        (B, 3 * len(contact_J_index), D)
    """
    B, D, _ = Minv.shape

    if contact_J_index is None or len(contact_J_index) == 0:
        return Minv, None

    n_contacts = len(contact_J_index)
    n_constraints = n_contacts * 3

    Jc = J_p[:, contact_J_index, :, :].reshape(B, n_constraints, D)
    Jc_T = Jc.transpose(1, 2)

    Jc_Minv = torch.matmul(Jc, Minv)
    A = torch.matmul(Jc_Minv, Jc_T)

    eye_A = torch.eye(A.shape[1], device=Minv.device, dtype=Minv.dtype).unsqueeze(0)
    Ainv = torch.linalg.pinv(A + lam * eye_A)

    Minv_JcT = torch.matmul(Minv, Jc_T)
    Minv_contact = Minv - torch.matmul(
        torch.matmul(Minv_JcT, Ainv),
        Jc_Minv
    )

    return Minv_contact, Jc

def dof_col(j_idx, axis_name, base_dofs=3, dof_per_joint=3):

    return base_dofs + j_idx * dof_per_joint + AXIS_TO_IDX[axis_name]

def build_structural_locked_cols(
    rom_cfg,
    always_lock_joints=(10, 11),
    base_dofs=3,
    dof_per_joint=3,
):
    locked_cols = set()

    # always lock joints
    for j_lock in always_lock_joints:
        for ax in ALL_AXES:
            col = dof_col(j_lock, ax, base_dofs, dof_per_joint)
            locked_cols.add(col)

    # structural lock axes
    for j_lock, missing_axes in rom_cfg["structural_lock_axes"].items():
        for ax in missing_axes:
            col = dof_col(j_lock, ax, base_dofs, dof_per_joint)
            locked_cols.add(col)

    return sorted(list(locked_cols))


def apply_joint_lock_projector_from_cols(
    Minv,                   # (B, D, D)
    locked_cols_per_batch,  # list[list[int]]
    lam=1e-5,
):
    """
    selector one-hot matrix 없이, locked column index만으로 projector 적용.
    batch별로 locked cols가 다를 수 있는 일반 버전.
    """
    B, D, _ = Minv.shape
    device = Minv.device
    dtype = Minv.dtype

    Minv_lock = Minv.clone()

    for b in range(B):
        cols = locked_cols_per_batch[b]

        if len(cols) == 0:
            continue

        cols_t = torch.tensor(cols, device=device, dtype=torch.long)

        Minv_b = Minv[b]                       # (D, D)
        Minv_cols = Minv_b[:, cols_t]         # (D, R)
        Minv_rows = Minv_b[cols_t, :]         # (R, D)
        A_lock = Minv_b[cols_t][:, cols_t]    # (R, R)

        eye_R = torch.eye(len(cols), device=device, dtype=dtype)
        A_lock_inv = torch.linalg.pinv(A_lock + lam * eye_R)

        Minv_lock[b] = Minv_b - Minv_cols @ A_lock_inv @ Minv_rows

    return Minv_lock

def apply_joint_lock_projector_from_cols(
    Minv,                   # (B, D, D)
    locked_cols_per_batch,  # list[list[int]]
    lam=1e-5,
):
    """
    selector one-hot matrix 없이, locked column index만으로 projector 적용.
    batch별로 locked cols가 다를 수 있는 일반 버전.
    """
    B, D, _ = Minv.shape
    device = Minv.device
    dtype = Minv.dtype

    Minv_lock = Minv.clone()

    for b in range(B):
        cols = locked_cols_per_batch[b]

        if len(cols) == 0:
            continue

        cols_t = torch.tensor(cols, device=device, dtype=torch.long)

        Minv_b = Minv[b]                       # (D, D)
        Minv_cols = Minv_b[:, cols_t]         # (D, R)
        Minv_rows = Minv_b[cols_t, :]         # (R, D)
        A_lock = Minv_b[cols_t][:, cols_t]    # (R, R)

        eye_R = torch.eye(len(cols), device=device, dtype=dtype)
        A_lock_inv = torch.linalg.pinv(A_lock + lam * eye_R)

        Minv_lock[b] = Minv_b - Minv_cols @ A_lock_inv @ Minv_rows

    return Minv_lock


def get_candidate_joints_from_chain_groups(j_eval, chain_groups=CHAIN_GROUPS_19):
    """
    j_eval가 속한 체인 그룹 전체를 candidate joints로 반환.
    여러 그룹에 동시에 속하면 union 반환.
    어느 그룹에도 없으면 최소한 자기 parent chain만 fallback으로 사용.
    """
    candidate = set()

    for group in chain_groups:
        if j_eval in group:
            candidate.update(group)

    if len(candidate) == 0:
        # fallback: torso / head 같은 경우
        candidate.add(j_eval)

    return sorted(candidate)

def get_rom_near_axes_for_candidate_joints(
    pose_19_euler_deg,   # (B, 19, 3)  # q-type axis
    rom_cfg,
    candidate_joints,
    tol_deg=10.0,
):
    B = pose_19_euler_deg.shape[0]
    rom_near_info = {}

    candidate_set = set(candidate_joints)

    # --------------------------------------------------
    # 1) single_joint_rules
    # 이미 q-type axis로 저장되어 있음
    # --------------------------------------------------
    for j_lock, rule in rom_cfg["single_joint_rules"].items():
        if j_lock not in candidate_set:
            continue

        for q_axis, lim in rule["axes"].items():
            ax_idx = AXIS_TO_IDX[q_axis]
            q_now = pose_19_euler_deg[:, j_lock, ax_idx]

            near_upper = q_now >= (lim["max_deg"] - tol_deg)
            near_lower = q_now <= (lim["min_deg"] + tol_deg)

            if torch.any(near_upper | near_lower):
                rom_near_info.setdefault(j_lock, {})
                rom_near_info[j_lock].setdefault("single", {})
                rom_near_info[j_lock]["single"][q_axis] = {
                    "near_upper": near_upper,
                    "near_lower": near_lower,
                    "min_deg": lim["min_deg"],
                    "max_deg": lim["max_deg"],
                }

    # --------------------------------------------------
    # 2) shoulder_sum_rules
    # smpl axis 기준 rule을 joint별 q-axis로 변환해서 사용
    # --------------------------------------------------
    for j_lock, rule in rom_cfg["shoulder_sum_rules"].items():
        if j_lock not in candidate_set:
            continue

        j0, j1 = rule["sum_joints"]   # collar + shoulder

        for smpl_axis, lim in rule["axes_smpl"].items():
            q_axis_j0 = SMPL_TO_Q_AXIS_MAP_19[j0][smpl_axis]
            q_axis_j1 = SMPL_TO_Q_AXIS_MAP_19[j1][smpl_axis]
            q_axis_lock = SMPL_TO_Q_AXIS_MAP_19[j_lock][smpl_axis]

            ax_idx_j0 = AXIS_TO_IDX[q_axis_j0]
            ax_idx_j1 = AXIS_TO_IDX[q_axis_j1]

            q_sum = pose_19_euler_deg[:, j0, ax_idx_j0] + pose_19_euler_deg[:, j1, ax_idx_j1]

            near_upper = q_sum >= (lim["max_deg"] - tol_deg)
            near_lower = q_sum <= (lim["min_deg"] + tol_deg)

            if torch.any(near_upper | near_lower):
                rom_near_info.setdefault(j_lock, {})
                rom_near_info[j_lock].setdefault("sum", {})
                rom_near_info[j_lock]["sum"][smpl_axis] = {
                    "near_upper": near_upper,
                    "near_lower": near_lower,
                    "min_deg": lim["min_deg"],
                    "max_deg": lim["max_deg"],
                    "sum_joints": [j0, j1],
                    "q_axis_j0": q_axis_j0,
                    "q_axis_j1": q_axis_j1,
                    "q_axis_lock": q_axis_lock,
                }

    return rom_near_info


def build_directional_lock_cols_for_target_subset(
    pose_19_euler_deg,   # (B, 19, 3)
    Jp_target,           # (B, 3, D)
    u_target,            # (B, 3)
    rom_cfg,
    candidate_joints,
    tol_deg=10.0,
    base_dofs=3,
    dof_per_joint=3,
):
    B, _, D = Jp_target.shape
    device = Jp_target.device

    tau_dir = torch.matmul(
        Jp_target.transpose(1, 2),   # (B, D, 3)
        u_target.unsqueeze(-1)       # (B, 3, 1)
    ).squeeze(-1)                    # (B, D)

    rom_near_info = get_rom_near_axes_for_candidate_joints(
        pose_19_euler_deg=pose_19_euler_deg,
        rom_cfg=rom_cfg,
        candidate_joints=candidate_joints,
        tol_deg=tol_deg,
    )

    directional_lock_mask = torch.zeros(B, D, dtype=torch.bool, device=device)

    # 1) single rules
    for j_lock, info in rom_near_info.items():
        if "single" not in info:
            continue

        for q_axis, near_info in info["single"].items():
            col = dof_col(j_lock, q_axis, base_dofs, dof_per_joint)
            tau = tau_dir[:, col]

            hit_upper = near_info["near_upper"] & (tau > 0.0)
            hit_lower = near_info["near_lower"] & (tau < 0.0)

            directional_lock_mask[:, col] |= (hit_upper | hit_lower)

    # 2) shoulder sum rules
    for j_lock, info in rom_near_info.items():
        if "sum" not in info:
            continue

        for smpl_axis, near_info in info["sum"].items():
            q_axis_lock = near_info["q_axis_lock"]
            col = dof_col(j_lock, q_axis_lock, base_dofs, dof_per_joint)
            tau = tau_dir[:, col]

            hit_upper = near_info["near_upper"] & (tau > 0.0)
            hit_lower = near_info["near_lower"] & (tau < 0.0)

            directional_lock_mask[:, col] |= (hit_upper | hit_lower)

    dir_locked_cols_per_batch = []
    for b in range(B):
        locked_cols = torch.where(directional_lock_mask[b])[0].tolist()
        dir_locked_cols_per_batch.append(locked_cols)

    return dir_locked_cols_per_batch, tau_dir, rom_near_info


class SMPL(object):
    def __init__(self, model_path=SMPL_MODEL_PATH, device=None):
        super(SMPL, self).__init__()
        """
        Build SMPL model
        """
        self.device = device if device is not None else torch.device('cpu')

        # -- Load SMPL params --        
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
        # Mean template vertices: [6890, 3]
        self.v_template = torch.from_numpy(params['v_template']).type(torch.float32)
        self.faces = params['f']
        # joint regressor
        self.regressor = torch.from_numpy(params['V_regressor']).type(torch.float32).transpose(1, 0)
        self.J_regressor = torch.from_numpy(params['J_regressor']).type(torch.float32).transpose(1, 0)
        # Parent for 24 and 37
        self.parents = params['kintree_table'].astype(np.int32)
        self.J_parents = params['kintree_table_J'].astype(np.int32)
        # Shape blend shape basis: [6890, 3, 10]
        # transposed to [10, 6890, 3]
        self.shapedirs = torch.from_numpy(params['shapedirs'].transpose(2, 0, 1)).type(torch.float32)
        # Pose blend shape basis: [6890, 3, 207]
        # transposed to [207, 6890, 3]
        self.posedirs = torch.from_numpy(params['posedirs'].transpose(2, 0, 1)).type(torch.float32)
        # LBS weights [6890, 24]
        self.weights = torch.from_numpy(params['weights']).type(torch.float32)
        self.joints_num = 24
        self.verts_num = 6890


        for name in ['v_template', 'J_regressor', 'regressor', 'weights', 'posedirs', 'shapedirs',
                     ]:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(self.device))

    def forward(self, betas, rotmat):

        N = betas.size()[0]
        # 1. Add shape blend shapes
        # (N, 10) x (10, 6890, 3) = [N, 6890, 3]
        v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [0])) + self.v_template.unsqueeze(0)
        joint_smpl_shaped = torch.matmul(v_shaped.transpose(1, 2), self.regressor).transpose(1, 2)

        # 2. Add pose blend shapes
        # 2.1 Infer shape-dependent joint locations.
        # transpose [N, 6890, 3] to [N, 3, 6890] and perform multiplication
        # transpose results [N, 3, J] to [N, J, 3]
        J = torch.matmul(v_shaped.transpose(1, 2), self.regressor).transpose(1, 2)
        # 2.2 add pose blend shapes 
        # rotation matrix [N,24,3,3]
        Rs = rotmat
        # ignore global rotation [N,23,3,3]
        pose = Rs[:, 1:, :, :]
        # rotation of T-pose
        pose_I = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        # weight matrix [N, 207]
        lrotmin = (pose - pose_I).view(-1, 207)
        # blended model [N,6890,3]
        v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1], [0]))

        # 3. Do LBS
        # obtain the transformed transformation matrix
        _, A = batch_global_rigid_transformation(Rs, J, self.parents)
        # repeat the weight matrix to [N,6890,24]
        W = self.weights.repeat(N, 1, 1)
        # calculate the blended transformation matrix 
        # [N,6890,24] * [N,24,16] = [N,6890,16] > [N,6890,4,4]
        T = torch.matmul(W, A.view(N, 24, 16)).view(N, 6890, 4, 4)
        # homegeous form of blended model [N,6890,4]
        v_posed_homo = torch.cat([v_posed,
                                  torch.ones([N, self.verts_num, 1]).type(torch.float32).to(self.device)], dim=2)
        # calculate the transformed 3D vertices position
        # [N,6890,4,4] * [N,6890,4,1] = [N,6890,4,1]
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
        verts = v_homo[:, :, :3, 0]  # [N,6890,3]

        # estimate 3D joint locations
        joint_regressed = torch.matmul(verts.transpose(1, 2), self.J_regressor).transpose(1, 2)
        # estimate 3D joint locations
        joint_regressed_smpl = torch.matmul(verts.transpose(1, 2), self.regressor).transpose(1, 2)

        output = ModelOutput(v_shaped=v_shaped,
                             joint_smpl_shaped=joint_smpl_shaped,
                             vertices=verts,
                             joints=joint_regressed,
                             joints_smpl=joint_regressed_smpl)
        return output
    

class SMPLH(object):
    def __init__(self, gender='neutral', num_betas=10, device=None):
        super(SMPLH, self).__init__()
        """
        Build SMPLH model
        """
        # -- Load SMPL params -- 
        self.device = device if device is not None else torch.device('cpu')


        if gender == 'male':
            params = np.load(SMPLH_M_PATH)
        elif gender == 'female':
            params = np.load(SMPLH_F_PATH)
        else:
            params = np.load(SMPLH_N_PATH)
        # Mean template vertices: [6890, 3]
        self.v_template = torch.from_numpy(params['v_template']).type(torch.float32)
        self.faces = params['f']
        # print(f" faces_shape : {self.faces.shape}")
        # joint regressor of the official model
        with open(SMPL_MODEL_PATH, 'rb') as f:
            params_smpl = pickle.load(f)
        # 24 and 37
        self.regressor = torch.from_numpy(params_smpl['V_regressor']).type(torch.float32).transpose(1, 0)
        self.J_regressor = torch.from_numpy(params_smpl['J_regressor']).type(torch.float32).transpose(1, 0)
        # Parent for 24 and 37
        self.parents = params_smpl['kintree_table'].astype(np.int32)
        self.J_parents = params_smpl['kintree_table_J'].astype(np.int32)

        # transposed to [10, 6890, 3]
        self.shapedirs = torch.from_numpy(params['shapedirs'].transpose(2, 0, 1)).type(torch.float32)[:num_betas]
        # Pose blend shape basis: [6890, 3, 207]
        # transposed to [207, 6890, 3]

        #self.posedirs = torch.from_numpy(params['posedirs'].transpose(2, 0, 1)).type(torch.float32)[:9 * 21]
        self.posedirs = torch.from_numpy(params['posedirs'].transpose(2, 0, 1)).type(torch.float32)[:9 * 23]
        # LBS weights [6890, 24]
        self.weights = torch.from_numpy(params['weights_prior']).type(torch.float32)
        self.joints_num = 24
        self.verts_num = 6890

        for name in ['v_template', 'J_regressor', 'regressor', 'weights',
                     'posedirs', 'shapedirs',
                     ]:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(self.device))

    def forward(self, betas, rotmat):
        

        N = betas.size()[0]
        v_shape_mean = self.v_template.clone().repeat(N, 1, 1)
        # 1. Add shape blend shapes
        # (N, 10) x (10, 6890, 3) = [N, 6890, 3]
        v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [0])) + self.v_template.unsqueeze(0)
        joint_smpl_shaped = torch.matmul(v_shaped.transpose(1, 2), self.regressor).transpose(1, 2)

        # 2. Add pose blend shapes
        # 2.1 Infer shape-dependent joint locations.
        # transpose [N, 6890, 3] to [N, 3, 6890] and perform multiplication
        # transpose results [N, 3, J] to [N, J, 3]
        J = torch.matmul(v_shaped.transpose(1, 2), self.regressor).transpose(1, 2)
        # 2.2 add pose blend shapes 
        # rotation matrix [N,24,3,3]
        Rs = rotmat
        # ignore global rotation [N,23,3,3]
        #pose = Rs[:, 1:-2, :, :]
        pose = Rs[:, 1:, :, :]  # root 제외, 나머지 모든 관절 포함pose = Rs[:, 1:, :, :]  # root 제외, 나머지 모든 관절 포함
        # rotation of T-pose
        #pose_I = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        pose_I = torch.eye(3, device=self.device).view(1, 1, 3, 3)
        # weight matrix [N, 207]
        #lrotmin = (pose - pose_I).view(-1, 9 * 21)
        lrotmin = (pose - pose_I).view(N, -1)
        # blended model [N,6890,3]
        v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1], [0]))

        # 3. Do LBS
        # obtain the transformed transformation matrix
        _, A = batch_global_rigid_transformation(Rs, J, self.parents)
        # repeat the weight matrix to [N,6890,24]
        W = self.weights.repeat(N, 1, 1)
        # calculate the blended transformation matrix 
        # [N,6890,24] * [N,24,16] = [N,6890,16] > [N,6890,4,4]
        T = torch.matmul(W, A.view(N, 24, 16)).view(N, 6890, 4, 4)
        # homegeous form of blended model [N,6890,4]
        v_posed_homo = torch.cat([v_posed,
                                  torch.ones([N, self.verts_num, 1]).type(torch.float32).to(self.device)], dim=2)
        # calculate the transformed 3D vertices position
        # [N,6890,4,4] * [N,6890,4,1] = [N,6890,4,1]
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
        verts = v_homo[:, :, :3, 0]  # [N,6890,3]

        # estimate 3D joint locations
        joint_regressed = torch.matmul(verts.transpose(1, 2), self.J_regressor).transpose(1, 2)
        # estimate 3D joint locations
        joint_regressed_smpl = torch.matmul(verts.transpose(1, 2), self.regressor).transpose(1, 2)

        output = ModelOutput(v_shaped=v_shaped,
                             joint_smpl_shaped=joint_smpl_shaped,
                             vertices=verts,
                             joints=joint_regressed,
                             joints_smpl=joint_regressed_smpl)
        return output


# %%
class Physics:
    def __init__(self, device=None):
        
        self.device = device if device is not None else torch.device('cpu')

        PHYSICS_PATH = '/home/robotics/hrc_ws/src/hrc_body_tracking/server/physics.pkl'
        # physics
        with open(PHYSICS_PATH, 'rb') as f:
            physics_data = pickle.load(f)

        self.bodypart_faceid = physics_data['bodypart_faceid']
        self.bodypart_vertexid = physics_data['bodypart_vertexid']

        self.mass = torch.from_numpy(physics_data['mass']).type(torch.float32).unsqueeze(0).to(self.device)
        self.mass_scaling = torch.from_numpy(physics_data['mass_scaling']).type(torch.float32).unsqueeze(0).to(
            self.device)
        self.volumn = torch.from_numpy(physics_data['volumn']).type(torch.float32).unsqueeze(0).to(self.device)
        self.inertia = torch.from_numpy(physics_data['inertia']).type(torch.float32).unsqueeze(0).to(self.device)

        # Static SMPL-19 Jacobian data. These values never change per frame.
        idx_24_to_19 = [
            0, 1, 2, 3, 4, 5, 7, 8, 6,
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        ]
        self._smpl19_idx_24_to_19 = torch.tensor(
            idx_24_to_19,
            dtype=torch.long,
            device=self.device,
        )
        self._smpl19_parent = [int(x) for x in self.get_smpl19_kinematic_tree()]
        self._smpl19_orders = self.get_smpl19_rot_order()
        self._smpl19_topo_order = [
            int(x) for x in self.topo_order_from_parent_array(self._smpl19_parent)
        ]
        self._smpl19_axes = {
            "x": torch.tensor([1.0, 0.0, 0.0], device=self.device).view(1, 3, 1),
            "y": torch.tensor([0.0, 1.0, 0.0], device=self.device).view(1, 3, 1),
            "z": torch.tensor([0.0, 0.0, 1.0], device=self.device).view(1, 3, 1),
        }
        self._smpl19_axis_col = {"x": 0, "y": 1, "z": 2}

        parent_chain_part_19 = [
            [0],
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 1, 4],
            [0, 2, 5],
            [0, 1, 4, 6],
            [0, 2, 5, 7],
            [0, 3, 8],
            [0, 3, 8, 9],
            [0, 3, 8, 10],
            [0, 3, 8, 11],
            [0, 3, 8, 9, 12],
            [0, 3, 8, 10, 13],
            [0, 3, 8, 11, 14],
            [0, 3, 8, 10, 13, 15],
            [0, 3, 8, 11, 14, 16],
            [0, 3, 8, 10, 13, 15, 17],
            [0, 3, 8, 11, 14, 16, 18],
        ]
        self._smpl19_jp_indices = []
        for chain_indices in parent_chain_part_19:
            parent_chain_joint = np.repeat(chain_indices, 3)
            parent_chain_j = parent_chain_joint * 3 + 3
            parent_chain_j[1::3] += 1
            parent_chain_j[2::3] += 2
            self._smpl19_jp_indices.append((
                torch.tensor(parent_chain_joint, dtype=torch.long, device=self.device),
                torch.tensor(parent_chain_j, dtype=torch.long, device=self.device),
            ))


   
    def compute_massinertia(self, v_shaped, joints_smpl):
        N = v_shaped.size(0)
        mass = self.mass.clone().repeat(N, 1)
        volume = self.volumn.clone().repeat(N, 1)
        inertia = self.inertia.clone().repeat(N, 1, 1, 1)
        centroid_all = torch.zeros(N, 24, 3, device=v_shaped.device, dtype=v_shaped.dtype)
        
        vertices_all = torch.cat([v_shaped, joints_smpl], dim=1)

        for n_b in range(24):
            # bodypart별 vertex 인덱스
            vertex_list = self.bodypart_vertexid[n_b]
            centroid = torch.mean(vertices_all[:, vertex_list], dim=1, keepdims=True)
            centroid_all[:, n_b, :] = centroid.squeeze(1)

            # volume 계산
            triangle_list = np.concatenate(self.bodypart_faceid[n_b])
            triangles = (vertices_all[:, triangle_list] - centroid).reshape([-1, 3, 3])
            volume_triangle = (torch.abs(torch.det(triangles)) / 6).reshape(N, len(triangle_list) // 3)
            volume[:, n_b] = torch.sum(volume_triangle, dim=1)

            # mass 업데이트
            mass[:, n_b] = (volume[:, n_b] / self.volumn[:, n_b].clone() - 1) * \
                            self.mass[:, n_b].clone() * self.mass_scaling[:, n_b].clone() + \
                            self.mass[:, n_b].clone()

            # inertia 계산
            triangles_batch = triangles.reshape([N, len(triangle_list) // 3, 3, 3])
            triangles_x = triangles_batch[:, :, :, 0]
            triangles_y = triangles_batch[:, :, :, 1]
            triangles_z = triangles_batch[:, :, :, 2]

            inertia[:, n_b, 2, 2] = torch.sum((triangles_x[:, :, 0] ** 2 + triangles_y[:, :, 0] ** 2 +
                                            triangles_x[:, :, 1] ** 2 + triangles_y[:, :, 1] ** 2 +
                                            triangles_x[:, :, 2] ** 2 + triangles_y[:, :, 2] ** 2 +
                                            (triangles_x[:, :, 0] + triangles_x[:, :, 1] + triangles_x[:, :, 2]) ** 2 +
                                            (triangles_y[:, :, 0] + triangles_y[:, :, 1] + triangles_y[:, :, 2]) ** 2) *
                                            volume_triangle, dim=1)

            inertia[:, n_b, 1, 1] = torch.sum((triangles_x[:, :, 0] ** 2 + triangles_z[:, :, 0] ** 2 +
                                            triangles_x[:, :, 1] ** 2 + triangles_z[:, :, 1] ** 2 +
                                            triangles_x[:, :, 2] ** 2 + triangles_z[:, :, 2] ** 2 +
                                            (triangles_x[:, :, 0] + triangles_x[:, :, 1] + triangles_x[:, :, 2]) ** 2 +
                                            (triangles_z[:, :, 0] + triangles_z[:, :, 1] + triangles_z[:, :, 2]) ** 2) *
                                            volume_triangle, dim=1)

            inertia[:, n_b, 0, 0] = torch.sum((triangles_y[:, :, 0] ** 2 + triangles_z[:, :, 0] ** 2 +
                                            triangles_y[:, :, 1] ** 2 + triangles_z[:, :, 1] ** 2 +
                                            triangles_y[:, :, 2] ** 2 + triangles_z[:, :, 2] ** 2 +
                                            (triangles_y[:, :, 0] + triangles_y[:, :, 1] + triangles_y[:, :, 2]) ** 2 +
                                            (triangles_z[:, :, 0] + triangles_z[:, :, 1] + triangles_z[:, :, 2]) ** 2) *
                                            volume_triangle, dim=1)

            inertia[:, n_b, 0, 1] = -torch.sum((triangles_x[:, :, 0] * triangles_y[:, :, 0] +
                                                triangles_x[:, :, 1] * triangles_y[:, :, 1] +
                                                triangles_x[:, :, 2] * triangles_y[:, :, 2] +
                                                (triangles_x[:, :, 0] + triangles_x[:, :, 1] + triangles_x[:, :, 2]) *
                                                (triangles_y[:, :, 0] + triangles_y[:, :, 1] + triangles_y[:, :, 2])) *
                                            volume_triangle, dim=1)
            inertia[:, n_b, 1, 0] = inertia[:, n_b, 0, 1]

            inertia[:, n_b, 0, 2] = -torch.sum((triangles_x[:, :, 0] * triangles_z[:, :, 0] +
                                                triangles_x[:, :, 1] * triangles_z[:, :, 1] +
                                                triangles_x[:, :, 2] * triangles_z[:, :, 2] +
                                                (triangles_x[:, :, 0] + triangles_x[:, :, 1] + triangles_x[:, :, 2]) *
                                                (triangles_z[:, :, 0] + triangles_z[:, :, 1] + triangles_z[:, :, 2])) *
                                            volume_triangle, dim=1)
            inertia[:, n_b, 2, 0] = inertia[:, n_b, 0, 2]

            inertia[:, n_b, 1, 2] = -torch.sum((triangles_y[:, :, 0] * triangles_z[:, :, 0] +
                                                triangles_y[:, :, 1] * triangles_z[:, :, 1] +
                                                triangles_y[:, :, 2] * triangles_z[:, :, 2] +
                                                (triangles_y[:, :, 0] + triangles_y[:, :, 1] + triangles_y[:, :, 2]) *
                                                (triangles_z[:, :, 0] + triangles_z[:, :, 1] + triangles_z[:, :, 2])) *
                                            volume_triangle, dim=1)
            inertia[:, n_b, 2, 1] = inertia[:, n_b, 1, 2]

        inertia = inertia / 20 * mass[:, :, None, None] / volume[:, :, None, None]
        return mass, volume, inertia, centroid_all
    

    def compute_centorid_24(self, v_shaped, joints_smpl):
        N = v_shaped.size(0)

        centroid_all = torch.zeros(N, 24, 3, device=v_shaped.device, dtype=v_shaped.dtype)
        vertices_all = torch.cat([v_shaped, joints_smpl], dim=1)

        for n_b in range(24):
            # bodypart별 vertex 인덱스
            vertex_list = self.bodypart_vertexid[n_b]
            centroid = torch.mean(vertices_all[:, vertex_list], dim=1, keepdims=True)
            centroid_all[:, n_b, :] = centroid.squeeze(1)

        return centroid_all
    

    def centroid_24_to_19(self, centroid24):
       
        keep_map = {
            0: [0],    # pelvis
            1: [1],    # L_hip
            2: [2],    # R_hip
            3: [3],
            4: [4],    # L_knee
            5: [5],    # R_knee
            9: [12],   # neck
            10: [13],  # L_collar
            11: [14],  # R_collar
            12: [15],  # Head
            13: [16],  # L_shoulder
            14: [17],  # R_shoulder
            15: [18],  # L_Elbow
            16: [19],  # R_Elbow
        }
        merge_map = {
            8:  [6,9],      # spine1 + spine2
            6:  [7, 10],     # L_ankle + L_foot
            7:  [8, 11],     # R_ankle + R_foot
            17: [20, 22],    # L_wrist + L_hand
            18: [21, 23],    # R_wrist + R_hand
        }

        device = centroid24.device
        dtype = centroid24.dtype
        N = centroid24.shape[0]
        centroid19 = torch.zeros((N, 19, 3),    dtype=centroid24.dtype, device=device)

        # ---- 1) 그대로 복사되는 항목 ----
        for new_idx, olds in keep_map.items():
            old = olds[0]
            centroid19[:, new_idx] = centroid24[:, old]
           
        # ---- 2) 병합되는 항목 (vertex 개수 가중 평균 + 평행축 정리 합산) ----
        for new_idx, olds in merge_map.items():
            # centroid 단순 평균
            c_parts = centroid24[:, olds, :]   # (N, k, 3)
            c_total = c_parts.mean(dim=1)      # (N, 3)
            centroid19[:, new_idx] = c_total
         

        return centroid19
    
    def merge_parts_24_to_19_vertex_avg(self, mass24, volume24, inertia24, centroid24):
        """
        24개 파라미터(centroid는 vertex 평균 기준)를 19개로 병합.
        - mass24    : (N, 24)
        - volume24  : (N, 24)
        - inertia24 : (N, 24, 3, 3)    # 각 파트 '자신의 centroid' 기준
        - centroid24: (N, 24, 3)       # 각 파트 vertex 평균 중심

        반환:
        - mass19    : (N, 19)
        - volume19  : (N, 19)
        - inertia19 : (N, 19, 3, 3)    # 병합 centroid 기준
        - centroid19: (N, 19, 3)       # vertex 개수 가중 평균
        """
        keep_map = {
            0: [0],    # pelvis
            1: [1],    # L_hip
            2: [2],    # R_hip
            3: [3],
            4: [4],    # L_knee
            5: [5],    # R_knee
            9: [12],   # neck
            10: [13],  # L_collar
            11: [14],  # R_collar
            12: [15],  # Head
            13: [16],  # L_shoulder
            14: [17],  # R_shoulder
            15: [18],  # L_Elbow
            16: [19],  # R_Elbow
        }
        merge_map = {
            8:  [6,9],      # spine1 + spine2
            6:  [7, 10],     # L_ankle + L_foot
            7:  [8, 11],     # R_ankle + R_foot
            17: [20, 22],    # L_wrist + L_hand
            18: [21, 23],    # R_wrist + R_hand
        }

        device = mass24.device
        dtype = mass24.dtype
        N = mass24.shape[0]
        I3 = torch.eye(3, device=device, dtype=dtype)[None, None, :, :]  # (1,1,3,3)

    

        # ---- 파트별 vertex 개수(가중치) 준비: vertex 평균 방식 유지 ----
        # self.bodypart_vertexid[i] 는 i번째 파트의 vertex 인덱스 리스트라고 가정
        vert_counts = torch.tensor(
            [len(self.bodypart_vertexid[i]) for i in range(24)],
            device=device, dtype=dtype
        ).view(1, 24, 1)  # (1,24,1), 브로드캐스트 용

        # ---- 출력 텐서 ----
        mass19     = torch.zeros((N, 19),       dtype=dtype, device=device)
        volume19   = torch.zeros((N, 19),       dtype=volume24.dtype, device=device)
        inertia19  = torch.zeros((N, 19, 3, 3), dtype=inertia24.dtype, device=device)
        centroid19 = torch.zeros((N, 19, 3),    dtype=centroid24.dtype, device=device)

        # ---- 유틸: 평행축 정리 ----
        def parallel_axis(I, m, d):
            """
            I: (N,3,3)  - 파트 고유 centroid 기준 관성
            m: (N,)     - 파트 질량
            d: (N,3)    - 파트 centroid -> 병합 centroid 벡터
            """
            d = d.view(-1, 3, 1)                       # (N,3,1)
            ddT = d @ d.transpose(1, 2)                # (N,3,3)
            d2I = (d.transpose(1, 2) @ d).view(-1,1,1) * I3[0,0]  # (N,3,3)
            return I + m.view(-1,1,1) * (d2I - ddT)
        
        # joint mass 분배
        mass24[:,3] += mass24[:, 6]/2
        mass24[:, 9] -= mass24[:,6]/2

        # ---- 1) 그대로 복사되는 항목 ----
        for new_idx, olds in keep_map.items():
            old = olds[0]
            mass19[:, new_idx]     = mass24[:, old]
            volume19[:, new_idx]   = volume24[:, old]
            centroid19[:, new_idx] = centroid24[:, old]
            inertia19[:, new_idx]  = inertia24[:, old]

        # ---- 2) 병합되는 항목 (vertex 개수 가중 평균 + 평행축 정리 합산) ----
        for new_idx, olds in merge_map.items():
            # 질량/체적 합
            m_parts = mass24[:, olds]        # (N,k)
            v_parts = volume24[:, olds]      # (N,k)

            # centroid 단순 평균
            c_parts = centroid24[:, olds, :]   # (N, k, 3)
            c_total = c_parts.mean(dim=1)      # (N, 3)

            # 병합 질량 / 체적
            m_total = m_parts.sum(dim=1)       # (N,)
            v_total = v_parts.sum(dim=1)       # (N,)

            # 병합 관성: 새 centroid 기준으로 평행축 정리 후 합산
            I_total = torch.zeros((N,3,3), dtype=inertia24.dtype, device=device)
            for k, old in enumerate(olds):
                Ik = inertia24[:, old, :, :]              # (N,3,3)
                mk = mass24[:, old]                       # (N,)
                dk = centroid24[:, old, :] - c_total      # (N,3)
                I_total = I_total + parallel_axis(Ik, mk, dk)

            # 저장 (+ 대칭화)
            mass19[:, new_idx]     = m_total
            volume19[:, new_idx]   = v_total
            centroid19[:, new_idx] = c_total
            inertia19[:, new_idx]  = 0.5 * (I_total + I_total.transpose(1,2))

        return mass19, volume19, inertia19, centroid19
    
    def get_smpl19_kinematic_tree(self):
        """
        Return parent index for the 19-joint merged SMPL model.
        This is the corrected tree where spine2+3 is merged into node 8.
        """
        parent = -np.ones(19, dtype=np.int64) # 전체를 -1로 초기화 후 필요한 joint만 덮어 씌우기, 부모는 -1로 설정해줘야 함

        parent[1] = 0   # L_hip
        parent[2] = 0   # R_hip
        parent[3] = 0   # spine1

        parent[4] = 1   # L_knee
        parent[5] = 2   # R_knee

        parent[6] = 4   # L_ankle(+foot)
        parent[7] = 5   # R_ankle(+foot)

        parent[8] = 3   # merged spine2+3 ← 핵심

        parent[9]  = 8  # neck
        parent[10] = 8  # L_collar
        parent[11] = 8  # R_collar
        parent[12] = 9  # head

        parent[13] = 10 # L_shoulder
        parent[14] = 11 # R_shoulder

        parent[15] = 13 # L_elbow
        parent[16] = 14 # R_elbow

        parent[17] = 15 # L_wrist(+hand)
        parent[18] = 16 # R_wrist(+hand)

        return parent
    
    def get_joint_rot_order(self):
        """
        Return a list of rotation orders for all 24 SMPL joints.
        """
        orders = ['yzx'] + ['zyx'] * 15 + ['zxy'] * 2 + ['yzx'] * 2 + ['zyx'] * 4
        return orders
    
    def get_smpl19_rot_order(self):
        """
        19-joint rotation order (inherited from SMPL 24-joint version).
        """
        orders_24 = self.get_joint_rot_order()  # original 24 개의 회전 순서
        idx_24_to_19 = [
        0, 1, 2, 3, 4, 5,   # pelvis, hips, spine1, knees
        7, 8,               # ankles (merged with feet)
        6,                  # spine2 (merged with spine3)
        12, 13, 14, 15,     # neck, L_collar, R_collar, head
        16, 17,             # shoulders
        18, 19,             # elbows   ← ★ 여기가 빠져 있었음!
        20, 21              # wrists (merged with hands)
        ]
        return [orders_24[i] for i in idx_24_to_19] # 해당하는 24개 조인트의 회전 순서를 그대로 가져온다."

        
    def topo_order_from_parent_array(self, parent):
        """
        parent_19 = [
        -1,   # pelvis (root)
        0,   # L_hip
        0,   # R_hip
        0,   # spine1
        1,   # L_knee
        2,   # R_knee
        4,   # L_ankle
        5,   # R_ankle
        3,   # spine2 (merged)
        8,   # neck
        8,   # L_collar
        8,   # R_collar
        9,   # head
        10,   # L_shoulder
        11,   # R_shoulder
        13,   # L_elbow
        14,   # R_elbow
        15,   # L_wrist
        16    # R_wrist
        ]
        """
        # 자식 리스트
        J = len(parent)
        children = [[] for _ in range(J)]
        for i, p in enumerate(parent):
            if p >= 0:
                children[p].append(i)
        """
        children[0] = [1,2,3]
        children[3] = [8]
        children[8] = [9,10,11]
        children[10] = [13]
        """

        order = []
        # root node 찾기
        q = [i for i,x in enumerate(parent) if x == -1]  # roots
        """
        root부터 시작
        root의 모든 자식을 방문
        그다음 자식들의 자식 방문
        """
        while q:
            u = q.pop(0)
            order.append(u)
            for v in children[u]:
                q.append(v)

        #따라서 순서가 이렇게 되어야 한다:

        #0 → 3 → 8 → 9 → 12  
        return order

    
    def compute_Jr_19_smpl(self,
        rotMat_individual_flat: torch.Tensor,
    ):
        """
        Compute rotational Jacobian J_r for custom 19-body SMPL model.

        Definitions
        -----------
        body type:
            [trans,
            pelvis_x, pelvis_y, pelvis_z,
            left_thigh_x, left_thigh_y, left_thigh_z,
            ...]
            즉, 19-body order + joint 내부 [x,y,z] 저장 순서.

        qtype:
            q_body[:, constants.smpl2q_19]

        rotMat_individual_flat:
            (B, 72, 3, 3), output from batch_roteulerSMPL(pose)

            Reshape:
                (B, 24, 3, 3, 3)

            For each joint:
                S0 = rotMat_individual[:, joint, 0]
                S1 = rotMat_individual[:, joint, 1]
                S2 = rotMat_individual[:, joint, 2]

            If order_str == "yzx":
                S0 = Ry
                S1 = Rz
                S2 = Rx

            The actual local rotation is always:
                R_local = S2 @ S1 @ S0

        return_type:
            "body":
                returns J_r_body.
                Use with qdot_body.

            "qtype":
                returns J_r_qtype = J_r_body[..., constants.smpl2q_19].
                Use with qdot_qtype.

        Returns
        -------
        J_r_out:
            (B, 19, 3, 60)

        world_rot:
            (B, 19, 3, 3)
        """


        B = rotMat_individual_flat.shape[0]
        device = rotMat_individual_flat.device
        dtype = rotMat_individual_flat.dtype

        J = 19
        DOF = 3 + 3 * J

        # (B,72,3,3) -> (B,24,3,3,3)
        rotMat_individual = rotMat_individual_flat.view(B, 24, 3, 3, 3)

        # 24 -> 19 representative
        rotMat_19 = rotMat_individual[:, self._smpl19_idx_24_to_19]  # (B,19,3,3,3)

        parent_19 = self._smpl19_parent
        orders_19 = self._smpl19_orders
        assert len(orders_19) == 19

        topo_order = self._smpl19_topo_order

        J_r_body = torch.zeros((B, J, 3, DOF), device=device, dtype=dtype)
        world_rot = torch.zeros((B, J, 3, 3), device=device, dtype=dtype)

        e = self._smpl19_axes
        axis_col = self._smpl19_axis_col

        I3_batch = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3).expand(B, 3, 3)

        for jnt in topo_order:
            parent = parent_19[jnt]

            if parent < 0:
                R_parent_world = I3_batch
            else:
                R_parent_world = world_rot[:, parent]
                J_r_body[:, jnt] = J_r_body[:, parent]

            S0 = rotMat_19[:, jnt, 0]
            S1 = rotMat_19[:, jnt, 1]
            S2 = rotMat_19[:, jnt, 2]

            order_str = orders_19[jnt]
            assert len(order_str) == 3, f"Invalid order for joint {jnt}: {order_str}"

            a0 = e[order_str[0]]
            a1 = e[order_str[1]]
            a2 = e[order_str[2]]

            # --------------------------------------------------
            # world rotation
            # rotMat_individual stores [rightmost, middle, leftmost]
            # therefore:
            # R_local = S2 @ S1 @ S0
            # --------------------------------------------------
            R_local = S2 @ S1 @ S0
            world_rot[:, jnt] = R_parent_world @ R_local

            # --------------------------------------------------
            # rotational Jacobian
            #
            # R_world = R_parent @ S2 @ S1 @ S0
            #
            # If S0 corresponds to axis order_str[0],
            # then its spatial axis is:
            #   R_parent @ S2 @ S1 @ axis(order_str[0])
            #
            # But the q column is not slot index.
            # The q column is x/y/z storage index:
            #   x -> base+0
            #   y -> base+1
            #   z -> base+2
            # --------------------------------------------------
            axis0_world = (R_parent_world @ S2 @ S1 @ a0).squeeze(-1)
            axis1_world = (R_parent_world @ S2 @ a1).squeeze(-1)
            axis2_world = (R_parent_world @ a2).squeeze(-1)

            base = 3 + 3 * jnt

            col0 = base + axis_col[order_str[0]]
            col1 = base + axis_col[order_str[1]]
            col2 = base + axis_col[order_str[2]]

            J_r_body[:, jnt, :, col0] = axis0_world
            J_r_body[:, jnt, :, col1] = axis1_world
            J_r_body[:, jnt, :, col2] = axis2_world

            J_r_out = J_r_body
            

        return J_r_out, world_rot

    def compute_J_p_19(self, com_all_19, joints_19, J_r_19):
        """
        19개 조인트 버전에 맞춘 Linear Jacobian 계산 함수
        
        Args:
            com_all_19: (N, 19, 3) - 19개 링크의 CoM 좌표
            joints_19: (N, 19, 3) - 19개 조인트의 월드 좌표 (순서는 정의한 0~18 따름)
            J_r_19: (N, 19, 3, 60) - 회전 자코비안 (Floating base 3 + Joints 57)
            
        Returns:
            J_p_19_clone: (N, 19, 3, 60) - 선형 자코비안
        """

        N = com_all_19.shape[0]
        device = com_all_19.device
        
        # 1. 초기화 (N, 19개 링크, 3축, 60 DOF)
        Jp_com = torch.zeros(
            (N, 19, 3, 60),
            dtype=com_all_19.dtype,
            device=device,
        )
        
        # 2. Floating Base 처리 (Identity)
        Jp_com[:, :, 0, 0] = 1.
        Jp_com[:, :, 1, 1] = 1.
        Jp_com[:, :, 2, 2] = 1.
        
        # 3. 19개 링크에 대해 Loop
        for n_b in range(19):
            parent_chain_joint, parent_chain_j = self._smpl19_jp_indices[n_b]

            # (2) Lever Arm 계산 (r = CoM - Joint)
            # com_curr: (N, 1, 3)
            com_curr = com_all_19[:, n_b:n_b+1, :] 
            
            # joints_subset: (N, num_ancestors*3, 3)
            # parent_chain_joint를 이용해 해당 조인트의 위치를 반복해서 가져옴
            # (x회전축, y회전축, z회전축 모두 같은 조인트 위치를 공유하므로)
            joints_subset = joints_19[:, parent_chain_joint, :]
            
            # r_vec: (N, num_ancestors*3, 3)
            r_com_n_b = com_curr - joints_subset

            # (3) Cross Product 및 할당
            # J_r_subset: (N, 3, num_ancestors*3) -> 해당 링크의 회전축들
            # parent_chain_j를 사용하여 60차원 중 필요한 컬럼만 추출
            J_r_subset = J_r_19[:, n_b, :, parent_chain_j]
            
            # r_com_n_b.transpose(1, 2) -> (N, 3, num_ancestors*3)
            # Cross product: omega x r
            linear_term = torch.cross(J_r_subset, r_com_n_b.transpose(1, 2), dim=1)
            
            # 결과 저장
            Jp_com[:, n_b, :, parent_chain_j] = linear_term

        return Jp_com
    
    def compute_all_joint_meff_cholesky_constrained_debug_lam(
        self,
        Mq,                # (B, D, D)
        J_p,               # (B, J, 3, D)
        J_r,               # (B, J, 3, D)
        u,                 # (B, J, 3)
        contact_J_index,   # ex) [6,7]  (발 contact)
        constarind_J_index=None,   # ex) [10,11] (잠글 joint index)
        base_dofs=3,
        dof_per_joint=3
    ):
        """
        제약 조건(foot contact + joint lock)을 반영한 유효질량 및 작업공간 관성 계산.

        Parameters:
            - Mq: (B, D, D) - 일반화된 질량 행렬
            - J_p: (B, J, 3, D) - 선형 자코비안 (각 body CoM 기준)
            - J_r: (B, J, 3, D) - 회전 자코비안
            - u: (B, J, 3) - 단위 방향 벡터
            - contact_J_index: [list] - contact가 걸린 조인트 인덱스 (예: 발 [6,7])
            - constarind_J_index: [list] or None - joint-level lock (예: 어깨 [10,11])
            - base_dofs: root translation DOF 개수 (SMPL: 3)
            - dof_per_joint: joint당 DOF (SMPL: 3)

        Returns:
            - Meff: (B, J)                - 방향 u에서의 유효 질량
            - Lambda_inertia_constrained: (B, J, 6, 6) - 6D 작업공간 관성 (kg)
            - Lambda_inertia_lin_constrained: (B, J, 3, 3) - 선형부 3x3 관성 (kg)
        """
        # --- 0. shape 체크 ---
        Bm, Dm, Dm2 = Mq.shape
        assert Dm == Dm2, "Mq must be (B, D, D)"
        Bj, J, _, Dj = J_p.shape
        assert Bm == Bj, f"Batch mismatch: Mq B={Bm}, J_p B={Bj}"
        assert Dj == Dm, f"DoF mismatch: Mq D={Dm}, J_p last dim={Dj}"

        B, D = Bm, Dm
        epsilon = 1e-6
        lam = 1e-5

        # --- 1. Minv 계산 ---
        eye_D = torch.eye(D, device=Mq.device, dtype=Mq.dtype).unsqueeze(0)
        Mq_reg = Mq + epsilon * eye_D

        try:
            L = torch.linalg.cholesky(Mq_reg)   # (B, D, D)
            I_D = eye_D.expand(B, D, D)         # (B, D, D)
            Minv = torch.cholesky_solve(I_D, L) # (B, D, D)
        except torch.linalg.LinAlgError:
            print("Warning: Mq Cholesky failed. Falling back to Pinv.")
            Minv = torch.linalg.pinv(Mq_reg)
            L = None

        # --- 2. joint-level lock 제약 (constarind_J_index) ---
        #     예: constarind_J_index = [10,11] -> 어깨 joint 잠금(q̇=0)
        if constarind_J_index is not None and len(constarind_J_index) > 0:
            lock_dofs = []
            for j_idx in constarind_J_index:
                start = base_dofs + j_idx * dof_per_joint
                end   = start + dof_per_joint
                lock_dofs.extend(range(start, end))

            n_lock = len(lock_dofs)
            S_lock = torch.zeros(B, n_lock, D, device=Mq.device, dtype=Mq.dtype)
            for row, di in enumerate(lock_dofs):
                S_lock[:, row, di] = 1.0

            S_lock_T = S_lock.transpose(1, 2)        # (B, D, n_lock)
            Minv_S_T = torch.matmul(Minv, S_lock_T)  # (B, D, n_lock)
            S_Minv   = Minv_S_T.transpose(1, 2)      # (B, n_lock, D)

            A_lock = torch.matmul(S_Minv, S_lock_T)  # (B, n_lock, n_lock)
            eye_A_lock = torch.eye(n_lock, device=Mq.device, dtype=Mq.dtype).unsqueeze(0)
            A_lock_reg = A_lock + lam * eye_A_lock
            A_lock_inv = torch.linalg.pinv(A_lock_reg)

            tmp = torch.matmul(Minv_S_T, A_lock_inv) # (B, D, n_lock)
            Minv_lock = Minv - torch.matmul(tmp, S_Minv)  # (B, D, D)
        else:
            Minv_lock = Minv

        # --- 3. contact 제약 (발 등) ---
        # J_full : (B, J, 6, D)
        J_full = torch.cat([J_p, J_r], dim=2)
        J_full_T = J_full.transpose(2, 3)        # (B, J, D, 6)

        n_contacts = len(contact_J_index)
        n_constraints = n_contacts * 6

        # contact 자코비안: (B, n_constraints, D)
        Jc = J_full[:, contact_J_index, :, :].view(B, n_constraints, D)
        Jc_T = Jc.transpose(1, 2)  # (B, D, n_constraints)

        # A = Jc Minv_lock Jc^T
        Jc_Minv = torch.matmul(Jc, Minv_lock)        # (B, n_constraints, D)
        A = torch.matmul(Jc_Minv, Jc_T)              # (B, n_constraints, n_constraints)
        eye_A = torch.eye(A.shape[1], device=Mq.device, dtype=Mq.dtype).unsqueeze(0)
        A_reg = A + lam * eye_A
        Ainv = torch.linalg.pinv(A_reg)

        # Minv_eff = Minv_lock - Minv_lock Jc^T Ainv Jc Minv_lock
        Minv_JcT = torch.matmul(Minv_lock, Jc_T)     # (B, D, n_constraints)
        tmp2 = torch.matmul(Minv_JcT, Ainv)          # (B, D, n_constraints)
        Minv_eff = Minv_lock - torch.matmul(tmp2, Jc_Minv)  # (B, D, D)

        # --- 4. Lambda_mobility = J Minv_eff J^T ---
        BJ = B * J

        # (B, J, 6, D) -> (B*J, 6, D)
        J_full_flat = J_full.reshape(BJ, 6, D)

        # Minv_eff: (B, D, D) -> (B, J, D, D) -> (B*J, D, D)
        Minv_eff_exp = Minv_eff.unsqueeze(1).expand(B, J, D, D).reshape(BJ, D, D)

        # (B*J, 6, D) @ (B*J, D, D) -> (B*J, 6, D)
        JMinv_eff_flat = torch.matmul(J_full_flat, Minv_eff_exp)

        # (B*J, 6, D) -> (B, J, 6, D)
        JMinv_eff = JMinv_eff_flat.view(B, J, 6, D)

        # Lambda_mobility: (B, J, 6, 6)  (단위: 1/kg)
        Lambda_mobility = torch.matmul(JMinv_eff, J_full_T)

        # --- 5. Meff 계산 (기동성 기반) ---
        Lambda_lin_mobility = Lambda_mobility[:, :, :3, :3]  # (B, J, 3, 3)

        u_vec = u.unsqueeze(-1)        # (B, J, 3, 1)
        u_T = u_vec.transpose(2, 3)    # (B, J, 1, 3)

        denom = torch.matmul(u_T, torch.matmul(Lambda_lin_mobility, u_vec))
        Meff = 1.0 / (denom.squeeze(-1).squeeze(-1) + epsilon)  # (B, J)

        # --- 6. Mobility(1/kg) -> Inertia(kg) 변환 (Cholesky 기반) ---
        eye6 = torch.eye(6, dtype=Mq.dtype, device=Mq.device).view(1, 1, 6, 6)
        eye3 = torch.eye(3, dtype=Mq.dtype, device=Mq.device).view(1, 1, 3, 3)

        # 수치 안정화용 regularization
        Lambda_mobility_reg    = Lambda_mobility    + lam * eye6
        Lambda_lin_mob_reg     = Lambda_lin_mobility + lam * eye3

        # 6x6 전체 관성
        try:
            L_lam = torch.linalg.cholesky(Lambda_mobility_reg)        # (B, J, 6, 6)
            I6 = eye6.expand(B, J, 6, 6)
            Lambda_inertia_constrained = torch.cholesky_solve(I6, L_lam)  # (B, J, 6, 6)
        except torch.linalg.LinAlgError:
            Lambda_inertia_constrained = torch.linalg.pinv(Lambda_mobility_reg)

        # 3x3 선형 부분 관성
        try:
            L_lam_lin = torch.linalg.cholesky(Lambda_lin_mob_reg)     # (B, J, 3, 3)
            I3 = eye3.expand(B, J, 3, 3)
            Lambda_inertia_lin_constrained = torch.cholesky_solve(I3, L_lam_lin)  # (B, J, 3, 3)
        except torch.linalg.LinAlgError:
            Lambda_inertia_lin_constrained = torch.linalg.pinv(Lambda_lin_mob_reg)

        # 대칭화 (수치오차 보정)
        Lambda_inertia_constrained = 0.5 * (
            Lambda_inertia_constrained + Lambda_inertia_constrained.transpose(-2, -1)
        )
        Lambda_inertia_lin_constrained = 0.5 * (
            Lambda_inertia_lin_constrained + Lambda_inertia_lin_constrained.transpose(-2, -1)
        )

        return Meff, Lambda_inertia_lin_constrained
    
    def compute_all_joint_meff_cholesky_constrained_debug_lam_rom_per_target_debug(
        self,
        Mq,
        J_p,
        J_r,
        u,
        pose_19_euler_deg,
        rom_cfg,
        contact_J_index,
        j_eval,
        always_lock_joints=(10, 11),
        base_dofs=3,
        dof_per_joint=3,
        tol_deg=10.0,
        return_debug=False,
        print_timing=True,
    ):
        import time

        B, D, _ = Mq.shape
        _, J, _, _ = J_p.shape

        epsilon = 1e-6
        lam = 1e-5

        t0 = time.time()

        # 1) Minv
        eye_D = torch.eye(D, device=Mq.device, dtype=Mq.dtype).unsqueeze(0)
        Mq_reg = Mq + epsilon * eye_D

        try:
            L = torch.linalg.cholesky(Mq_reg)
            I_D = eye_D.expand(B, D, D)
            Minv = torch.cholesky_solve(I_D, L)
        except torch.linalg.LinAlgError:
            Minv = torch.linalg.pinv(Mq_reg)

        t1 = time.time()

        # 2) structural lock -> Minv_base
        structural_locked_cols = build_structural_locked_cols(
            rom_cfg=rom_cfg,
            always_lock_joints=always_lock_joints,
            base_dofs=base_dofs,
            dof_per_joint=dof_per_joint,
        )

        structural_locked_cols_per_batch = [structural_locked_cols for _ in range(B)]

        Minv_base = apply_joint_lock_projector_from_cols(
            Minv=Minv,
            locked_cols_per_batch=structural_locked_cols_per_batch,
            lam=lam,
        )

        t2 = time.time()

        # 3) directional lock cols
        candidate_joints = get_candidate_joints_from_chain_groups(j_eval, CHAIN_GROUPS_19)

        dir_locked_cols_per_batch, tau_dir, rom_near_info = \
            build_directional_lock_cols_for_target_subset(
                pose_19_euler_deg=pose_19_euler_deg,
                Jp_target=J_p[:, j_eval],
                u_target=u[:, j_eval],
                rom_cfg=rom_cfg,
                candidate_joints=candidate_joints,
                tol_deg=tol_deg,
                base_dofs=base_dofs,
                dof_per_joint=dof_per_joint,
            )

        t3 = time.time()

        # 4) directional-only projector on Minv_base
        Minv_lock = apply_joint_lock_projector_from_cols(
            Minv=Minv_base,
            locked_cols_per_batch=dir_locked_cols_per_batch,
            lam=lam,
        )

        t4 = time.time()

        # 5) foot contact projector
        n_contacts = len(contact_J_index)
        n_constraints = n_contacts * 3

        Jc = J_p[:, contact_J_index, :, :].reshape(B, n_constraints, D)
        Jc_T = Jc.transpose(1, 2)

        Jc_Minv = torch.matmul(Jc, Minv_lock)
        A = torch.matmul(Jc_Minv, Jc_T)

        eye_A = torch.eye(A.shape[1], device=Mq.device, dtype=Mq.dtype).unsqueeze(0)
        Ainv = torch.linalg.pinv(A + lam * eye_A)

        Minv_JcT = torch.matmul(Minv_lock, Jc_T)
        tmp2 = torch.matmul(Minv_JcT, Ainv)
        Minv_eff = Minv_lock - torch.matmul(tmp2, Jc_Minv)

        t5 = time.time()

        # 6) Lambda / Meff
        J_full_j = torch.cat([J_p[:, j_eval], J_r[:, j_eval]], dim=1)
        J_full_j_T = J_full_j.transpose(1, 2)

        JMinv_eff = torch.matmul(J_full_j, Minv_eff)
        Lambda_mobility = torch.matmul(JMinv_eff, J_full_j_T)
        Lambda_lin = Lambda_mobility[:, :3, :3]

        u_vec = u[:, j_eval].unsqueeze(-1)
        u_T = u_vec.transpose(1, 2)

        denom = torch.matmul(u_T, torch.matmul(Lambda_lin, u_vec))
        Meff = 1.0 / (denom.squeeze(-1).squeeze(-1) + epsilon)

        eye3 = torch.eye(3, dtype=Mq.dtype, device=Mq.device).unsqueeze(0)
        Lambda_lin_reg = Lambda_lin + lam * eye3

        try:
            L_lam_lin = torch.linalg.cholesky(Lambda_lin_reg)
            I3 = eye3.expand(B, 3, 3)
            Lambda_inertia_lin = torch.cholesky_solve(I3, L_lam_lin)
        except torch.linalg.LinAlgError:
            Lambda_inertia_lin = torch.linalg.pinv(Lambda_lin_reg)

        Lambda_inertia_lin = 0.5 * (
            Lambda_inertia_lin + Lambda_inertia_lin.transpose(-2, -1)
        )

        t6 = time.time()

        if print_timing:
            print("\n[TIMING DEBUG - single j_eval optimized v2]")
            print(f"  Minv calc                  : {t1 - t0:.6f}s")
            print(f"  structural projector       : {t2 - t1:.6f}s")
            print(f"  directional cols build     : {t3 - t2:.6f}s")
            print(f"  directional projector      : {t4 - t3:.6f}s")
            print(f"  contact projector          : {t5 - t4:.6f}s")
            print(f"  lambda / meff              : {t6 - t5:.6f}s")
            print(f"  total                      : {t6 - t0:.6f}s")

        if return_debug:
            debug_dict = {
                "j_eval": j_eval,
                "candidate_joints": candidate_joints,
                "structural_locked_cols": structural_locked_cols,
                "dir_locked_cols_per_batch": dir_locked_cols_per_batch,
                "tau_dir": tau_dir,
                "rom_near_info": rom_near_info,
                "Minv_base": Minv_base,
                "Minv_lock": Minv_lock,
                "Minv_eff": Minv_eff,
            }


            return Meff, Lambda_inertia_lin, debug_dict

        return Meff, Lambda_inertia_lin
    
    def prepare_meff_constrained_common(
        self,
        Mq,
        J_p,
        rom_cfg,
        contact_J_index,
        always_lock_joints=(10, 11),
        base_dofs=3,
        dof_per_joint=3,
    ):
        """Compute query-independent inverse-mass projectors for one snapshot."""
        B, D, D2 = Mq.shape
        assert D == D2, f"Mq must be (B,D,D), got {Mq.shape}"

        Bj, _, _, Dj = J_p.shape
        assert Bj == B and Dj == D, (
            f"J_p shape mismatch: Mq={Mq.shape}, J_p={J_p.shape}"
        )

        epsilon = 1e-6
        lam = 1e-5
        eye_D = torch.eye(D, device=Mq.device, dtype=Mq.dtype).unsqueeze(0)
        Mq_reg = Mq + epsilon * eye_D

        try:
            L = torch.linalg.cholesky(Mq_reg)
            Minv = torch.cholesky_solve(eye_D.expand(B, D, D), L)
        except torch.linalg.LinAlgError:
            Minv = torch.linalg.pinv(Mq_reg)

        structural_locked_cols = build_structural_locked_cols(
            rom_cfg=rom_cfg,
            always_lock_joints=always_lock_joints,
            base_dofs=base_dofs,
            dof_per_joint=dof_per_joint,
        )
        Minv_struct = apply_joint_lock_projector_from_cols(
            Minv=Minv,
            locked_cols_per_batch=[
                structural_locked_cols for _ in range(B)
            ],
            lam=lam,
        )
        Minv_contact, _ = apply_contact_projector_from_indices(
            Minv=Minv_struct,
            J_p=J_p,
            contact_J_index=contact_J_index,
            lam=lam,
        )
        return Minv_contact, tuple(structural_locked_cols)

    def compute_meff_constrained_from_common(
        self,
        Minv_contact,
        structural_locked_cols,
        J_p,
        J_r,
        u,
        pose_19_euler_deg,
        rom_cfg,
        j_eval,
        base_dofs=3,
        dof_per_joint=3,
        tol_deg=10.0,
        return_debug=False,
    ):
        """Compute target-dependent ROM constraints and M_eff from a snapshot."""
        B, D, D2 = Minv_contact.shape
        assert D == D2, (
            f"Minv_contact must be (B,D,D), got {Minv_contact.shape}"
        )

        Bj, J, _, Dj = J_p.shape
        assert Bj == B and Dj == D, (
            f"J_p shape mismatch: Minv={Minv_contact.shape}, J_p={J_p.shape}"
        )

        Br, JrJ, _, Dr = J_r.shape
        assert Br == B and JrJ == J and Dr == D, (
            f"J_r shape mismatch: J_r={J_r.shape}"
        )

        Bu, Ju, _ = u.shape
        assert Bu == B and Ju == J, (
            f"u shape mismatch: u={u.shape}, expected (B,{J},3)"
        )

        Bq, Jq, Aq = pose_19_euler_deg.shape
        assert Bq == B and Jq == 19 and Aq == 3, (
            f"pose_19_euler_deg must be (B,19,3), "
            f"got {pose_19_euler_deg.shape}"
        )

        j_eval = int(j_eval)
        if j_eval < 0 or j_eval >= J:
            raise ValueError(f"Invalid j_eval={j_eval}, J={J}")

        epsilon = 1e-6
        lam = 1e-5
        chain_joints = get_ordered_chain_from_chain_groups(
            j_eval=j_eval,
            chain_groups=CHAIN_GROUPS_19,
        )
        u_target = u[:, j_eval]
        u_target = u_target / (
            torch.norm(u_target, dim=-1, keepdim=True) + 1e-8
        )

        (
            Minv_eff,
            rom_locked_cols_per_batch,
            rom_near_info,
            rom_debug_steps,
        ) = apply_sequential_rom_projector_qddot_contact_free(
            Minv_contact=Minv_contact,
            J_p=J_p,
            u_target=u_target,
            pose_19_euler_deg=pose_19_euler_deg,
            rom_cfg=rom_cfg,
            chain_joints=chain_joints,
            structural_locked_cols=list(structural_locked_cols),
            base_dofs=base_dofs,
            dof_per_joint=dof_per_joint,
            tol_deg=tol_deg,
            lam=lam,
            j_eval=j_eval,
        )

        J_full_j = torch.cat([J_p[:, j_eval], J_r[:, j_eval]], dim=1)
        Lambda_mobility = (
            J_full_j @ Minv_eff @ J_full_j.transpose(1, 2)
        )
        Lambda_lin_mobility = Lambda_mobility[:, :3, :3]
        Lambda_lin_mobility = 0.5 * (
            Lambda_lin_mobility
            + Lambda_lin_mobility.transpose(-2, -1)
        )

        u_vec = u_target.unsqueeze(-1)
        denom = (
            u_vec.transpose(1, 2)
            @ Lambda_lin_mobility
            @ u_vec
        ).squeeze(-1).squeeze(-1)
        Meff = 1.0 / (denom + epsilon)

        eye3 = torch.eye(
            3,
            dtype=Minv_contact.dtype,
            device=Minv_contact.device,
        ).unsqueeze(0)
        Lambda_lin_reg = Lambda_lin_mobility + lam * eye3
        try:
            L_lam_lin = torch.linalg.cholesky(Lambda_lin_reg)
            Lambda_inertia_lin = torch.cholesky_solve(
                eye3.expand(B, 3, 3),
                L_lam_lin,
            )
        except torch.linalg.LinAlgError:
            Lambda_inertia_lin = torch.linalg.pinv(Lambda_lin_reg)

        Lambda_inertia_lin = 0.5 * (
            Lambda_inertia_lin + Lambda_inertia_lin.transpose(-2, -1)
        )

        if return_debug:
            return Meff, Lambda_inertia_lin, {
                "j_eval": j_eval,
                "chain_joints": chain_joints,
                "structural_locked_cols": list(structural_locked_cols),
                "rom_locked_cols_per_batch": rom_locked_cols_per_batch,
                "Minv_contact": Minv_contact,
                "Minv_eff": Minv_eff,
                "rom_near_info": rom_near_info,
                "rom_debug_steps": rom_debug_steps,
                "Lambda_lin_mobility": Lambda_lin_mobility,
                "Lambda_inertia_lin": Lambda_inertia_lin,
            }

        return Meff, Lambda_inertia_lin

    def fibonacci_sphere_directions(
        self,
        num_directions,
        dtype=torch.float32,
        device=None,
    ):
        """Return deterministic approximately uniform unit directions."""
        num_directions = int(num_directions)
        if num_directions < 4:
            raise ValueError("num_directions must be >= 4")

        if device is None:
            device = self.device

        idx = torch.arange(num_directions, dtype=dtype, device=device)
        golden_angle = torch.tensor(
            np.pi * (3.0 - np.sqrt(5.0)),
            dtype=dtype,
            device=device,
        )
        z = 1.0 - 2.0 * (idx + 0.5) / float(num_directions)
        radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        theta = golden_angle * idx
        dirs = torch.stack(
            [
                radius * torch.cos(theta),
                radius * torch.sin(theta),
                z,
            ],
            dim=-1,
        )

        axes = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=dtype,
            device=device,
        )
        dirs = torch.cat([dirs, axes], dim=0)
        return dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-8)

    def find_worst_meff_direction_constrained_from_common(
        self,
        Minv_contact,
        structural_locked_cols,
        J_p,
        J_r,
        pose_19_euler_deg,
        rom_cfg,
        j_eval,
        num_directions=64,
        tol_deg=5.0,
        meff_offset=0.0,
    ):
        """
        Brute-force search the direction with maximum constrained M_eff.

        ROM constraints are direction-dependent, so a simple eigenvector of a
        single inertia tensor is not generally sufficient. This helper evaluates
        a deterministic set of unit directions in one batch and returns the best
        direction for the requested target.
        """
        B = int(Minv_contact.shape[0])
        if B != 1:
            raise ValueError(
                "find_worst_meff_direction_constrained_from_common currently "
                f"expects one snapshot, got batch size {B}"
            )

        j_eval = int(j_eval)
        dirs = self.fibonacci_sphere_directions(
            num_directions=num_directions,
            dtype=Minv_contact.dtype,
            device=Minv_contact.device,
        )
        n_dirs = int(dirs.shape[0])

        u_full = torch.zeros(
            (n_dirs, 19, 3),
            dtype=Minv_contact.dtype,
            device=Minv_contact.device,
        )
        u_full[:, j_eval, :] = dirs

        meff, _ = self.compute_meff_constrained_from_common(
            Minv_contact=Minv_contact.expand(n_dirs, -1, -1).contiguous(),
            structural_locked_cols=structural_locked_cols,
            J_p=J_p.expand(n_dirs, -1, -1, -1).contiguous(),
            J_r=J_r.expand(n_dirs, -1, -1, -1).contiguous(),
            u=u_full,
            pose_19_euler_deg=pose_19_euler_deg.expand(
                n_dirs, -1, -1
            ).contiguous(),
            rom_cfg=rom_cfg,
            j_eval=j_eval,
            tol_deg=tol_deg,
            return_debug=False,
        )

        meff = meff.reshape(-1) + float(meff_offset)
        best_index = int(torch.argmax(meff).item())
        return meff[best_index], dirs[best_index], meff, dirs

    def compute_meff_constrained(
        self,
        Mq,
        J_p,
        J_r,
        u,
        pose_19_euler_deg,
        rom_cfg,
        contact_J_index,
        j_eval,
        always_lock_joints=(10, 11),
        base_dofs=3,
        dof_per_joint=3,
        tol_deg=10.0,
        return_debug=False,
        print_timing=True,
    ):
        """
        Contact-constrained free-acceleration 기반 sequential ROM constrained Meff 계산.

        이론적 순서:
            1) Mq -> Minv
            2) structural lock -> Minv_struct
            3) foot contact -> Minv_contact
            4) chain 순서대로 qddot_dir = Minv_current @ J_eval^T @ u 로 ROM active set 구성
            5) ROM lock이 생기면 즉시 projector 적용
            6) 최종 Minv_eff로 Lambda mobility / Meff 계산

        Parameters
        ----------
        Mq : torch.Tensor
            (B, D, D)
        J_p : torch.Tensor
            (B, J, 3, D)
        J_r : torch.Tensor
            (B, J, 3, D)
        u : torch.Tensor
            (B, J, 3). 기존 함수와 호환되도록 전체 body 방향 배열 사용.
            실제 계산은 u[:, j_eval] 사용.
        pose_19_euler_deg : torch.Tensor
            (B, 19, 3)
        rom_cfg : dict
            ROM 설정.
        contact_J_index : list[int]
            foot/contact link index. 예: [6, 7]
        j_eval : int
            collision target link index.
        """

        import time

        B, D, D2 = Mq.shape
        assert D == D2, f"Mq must be (B,D,D), got {Mq.shape}"

        Bj, J, _, Dj = J_p.shape
        assert Bj == B and Dj == D, f"J_p shape mismatch: Mq={Mq.shape}, J_p={J_p.shape}"

        Br, JrJ, _, Dr = J_r.shape
        assert Br == B and JrJ == J and Dr == D, f"J_r shape mismatch: J_r={J_r.shape}"

        Bu, Ju, _ = u.shape
        assert Bu == B and Ju == J, f"u shape mismatch: u={u.shape}, expected (B,{J},3)"

        Bq, Jq, Aq = pose_19_euler_deg.shape
        assert Bq == B and Jq == 19 and Aq == 3, (
            f"pose_19_euler_deg must be (B,19,3), got {pose_19_euler_deg.shape}"
        )

        j_eval = int(j_eval)
        if j_eval < 0 or j_eval >= J:
            raise ValueError(f"Invalid j_eval={j_eval}, J={J}")

        epsilon = 1e-6
        lam = 1e-5

        t0 = time.time()

        # --------------------------------------------------
        # 1) Minv 계산
        # --------------------------------------------------
        eye_D = torch.eye(D, device=Mq.device, dtype=Mq.dtype).unsqueeze(0)
        Mq_reg = Mq + epsilon * eye_D

        try:
            L = torch.linalg.cholesky(Mq_reg)
            I_D = eye_D.expand(B, D, D)
            Minv = torch.cholesky_solve(I_D, L)
        except torch.linalg.LinAlgError:
            Minv = torch.linalg.pinv(Mq_reg)

        t1 = time.time()

        # --------------------------------------------------
        # 2) structural lock -> Minv_struct
        # --------------------------------------------------
        structural_locked_cols = build_structural_locked_cols(
            rom_cfg=rom_cfg,
            always_lock_joints=always_lock_joints,
            base_dofs=base_dofs,
            dof_per_joint=dof_per_joint,
        )

        structural_locked_cols_per_batch = [structural_locked_cols for _ in range(B)]

        Minv_struct = apply_joint_lock_projector_from_cols(
            Minv=Minv,
            locked_cols_per_batch=structural_locked_cols_per_batch,
            lam=lam,
        )

        t2 = time.time()

        # --------------------------------------------------
        # 3) foot contact를 먼저 반영 -> Minv_contact
        # --------------------------------------------------
        Minv_contact, J_contact = apply_contact_projector_from_indices(
            Minv=Minv_struct,
            J_p=J_p,
            contact_J_index=contact_J_index,
            lam=lam,
        )

        t3 = time.time()

        # --------------------------------------------------
        # 4) j_eval이 속한 chain을 부모->자식 순서로 가져옴
        # --------------------------------------------------
        chain_joints = get_ordered_chain_from_chain_groups(
            j_eval=j_eval,
            chain_groups=CHAIN_GROUPS_19,
        )

        # --------------------------------------------------
        # 5) contact-aware qddot 기반 sequential ROM lock
        # --------------------------------------------------
        u_target = u[:, j_eval]
        u_target = u_target / (torch.norm(u_target, dim=-1, keepdim=True) + 1e-8)

        Minv_eff, rom_locked_cols_per_batch, rom_near_info, rom_debug_steps = \
            apply_sequential_rom_projector_qddot_contact_free(
                Minv_contact=Minv_contact,
                J_p=J_p,
                u_target=u_target,
                pose_19_euler_deg=pose_19_euler_deg,
                rom_cfg=rom_cfg,
                chain_joints=chain_joints,
                structural_locked_cols=structural_locked_cols,
                base_dofs=base_dofs,
                dof_per_joint=dof_per_joint,
                tol_deg=tol_deg,
                lam=lam,
                j_eval=j_eval,
            )

        t4 = time.time()

        # --------------------------------------------------
        # 6) Lambda mobility / Meff
        # --------------------------------------------------
        # 기존 함수와 동일하게 linear + rotational Jacobian으로 6D mobility를 만든 뒤,
        # linear 3x3 block만 사용.
        J_full_j = torch.cat([J_p[:, j_eval], J_r[:, j_eval]], dim=1)  # (B, 6, D)
        J_full_j_T = J_full_j.transpose(1, 2)                          # (B, D, 6)

        JMinv_eff = torch.matmul(J_full_j, Minv_eff)                   # (B, 6, D)
        Lambda_mobility = torch.matmul(JMinv_eff, J_full_j_T)          # (B, 6, 6)

        # 선형 mobility: Lambda^{-1}_{lin}
        Lambda_lin_mobility = Lambda_mobility[:, :3, :3]
        Lambda_lin_mobility = 0.5 * (
            Lambda_lin_mobility + Lambda_lin_mobility.transpose(-2, -1)
        )

        u_vec = u_target.unsqueeze(-1)          # (B, 3, 1)
        u_T = u_vec.transpose(1, 2)             # (B, 1, 3)

        denom = torch.matmul(
            u_T,
            torch.matmul(Lambda_lin_mobility, u_vec)
        ).squeeze(-1).squeeze(-1)

        Meff = 1.0 / (denom + epsilon)

        # --------------------------------------------------
        # 7) 선형 작업공간 관성 Lambda_lin = inv(mobility)
        # --------------------------------------------------
        eye3 = torch.eye(3, dtype=Mq.dtype, device=Mq.device).unsqueeze(0)
        Lambda_lin_reg = Lambda_lin_mobility + lam * eye3

        try:
            L_lam_lin = torch.linalg.cholesky(Lambda_lin_reg)
            I3 = eye3.expand(B, 3, 3)
            Lambda_inertia_lin = torch.cholesky_solve(I3, L_lam_lin)
        except torch.linalg.LinAlgError:
            Lambda_inertia_lin = torch.linalg.pinv(Lambda_lin_reg)

        Lambda_inertia_lin = 0.5 * (
            Lambda_inertia_lin + Lambda_inertia_lin.transpose(-2, -1)
        )

        t5 = time.time()

        # --------------------------------------------------
        # timing
        # --------------------------------------------------
        if print_timing:
            print("\n[TIMING DEBUG - compute_meff_constrained]")
            print(f"  Minv calc                  : {t1 - t0:.6f}s")
            print(f"  structural projector       : {t2 - t1:.6f}s")
            print(f"  contact projector          : {t3 - t2:.6f}s")
            print(f"  sequential ROM projector   : {t4 - t3:.6f}s")
            print(f"  lambda / meff              : {t5 - t4:.6f}s")
            print(f"  total                      : {t5 - t0:.6f}s")

        if return_debug:
            debug_dict = {
                "j_eval": j_eval,
                "chain_joints": chain_joints,

                # constraint columns
                "structural_locked_cols": structural_locked_cols,
                "rom_locked_cols_per_batch": rom_locked_cols_per_batch,

                # inverse masses
                "Minv": Minv,
                "Minv_struct": Minv_struct,
                "Minv_contact": Minv_contact,
                "Minv_eff": Minv_eff,

                # contact
                "contact_J_index": contact_J_index,
                "J_contact": J_contact,

                # ROM
                "rom_near_info": rom_near_info,
                "rom_debug_steps": rom_debug_steps,

                # final mobility / inertia
                "Lambda_lin_mobility": Lambda_lin_mobility,
                "Lambda_inertia_lin": Lambda_inertia_lin,
                "denom": denom,
                "Meff": Meff,
            }

            return Meff, Lambda_inertia_lin, debug_dict

        return Meff, Lambda_inertia_lin
