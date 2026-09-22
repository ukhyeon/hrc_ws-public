from copy import deepcopy
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import time
import yaml
import torch
import os.path as osp
import os
import glob
from PIL import Image as PILImage, ImageDraw, ImageFont
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import (
    HistoryPolicy,
    ReliabilityPolicy,
    DurabilityPolicy
)
import threading
from typing import Optional

# 1. ROS 2 메시지 타입 임포트
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
# 우리가 만든 커스텀 메시지 (빌드가 성공했다면 import 되어야 함)
from hrc_interfaces.msg import (
    HumanState,
    HumanPose,
    CollisionCandidates,
    DemoState,
    DirectionalMeffResult,
)
from copy import deepcopy
# 2. Azure Kinect SDK 임포트
import pyk4a
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS

# smpl 
from .inferer import Inferer
from .filterer import Filterer
try:
    from smplx_kinect.common.smplx_vis import VisualizerMeshSMPLX, VisualizerMeshSMPL
    VIS = True
except:
    print(f" at aggregate not vis")
    VIS = False
from smplx_kinect.common.body_models import HandsSMPLXWrapper, SMPLWrapper

from .physcis import SMPL
from collections import OrderedDict
# from .models.cliff_hr48.cliff import CLIFF as cliff_hr48
from hrc_body_tracking.models.cliff_hr48.cliff import CLIFF 
from pathlib import Path
from .cliff_img_processor import Img_Processor


# CLIFF 
PKG_ROOT = Path(__file__).resolve().parent
# PKG_ROOT = .../hrc_body_tracking/
MODEL_DIR = PKG_ROOT / "models" / "cliff_hr48"
mean_params_path = MODEL_DIR / "smpl_mean_params.npz"
checkpoint_path  = MODEL_DIR / "hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt"

# 시각화 최적화
import minimal_pytorch_rasterizer as mpr
from .effective_mass_colorizer import EffectiveMassColorizer19
from .collision_top3_colorizer import CollisionTop3Colorizer19
from .perf_logger import PerfCsvLogger
from .utils import camera_to_world_rotmat
from .activation_sphere_overlay import (
    draw_projected_sphere_overlay,
    project_world_points_to_image,
    project_sphere_wireframe,
)

def cam_params_decode(cam_params):
    """
    Azure Kinect 에서 가져온 Raw 캘리브레이션 데이터를 사용하기 쉬운 표준 행렬 형태로 변환
    """

    # 1. 내부 파라미터 추출
    K_rgb_distorted = np.eye(3, dtype=np.float32)
    for [i, j, k] in [[0, 0, 'fx'], [1, 1, 'fy'], [0, 2, 'cx'], [1, 2, 'cy']]:
        K_rgb_distorted[i, j] = cam_params['rgb_intrinsics'][k]
    # 2. 렌즈 왜곡 계수 추출
    rgb_distortion = np.array([
        cam_params['rgb_intrinsics'][k]
        for k in ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
    ], dtype=np.float32)
    # 외부  파라미터(Epth 센서와 rgb  센서 변환)
    rvec = np.array([cam_params['depth_to_rgb']['r'][i] for i in range(3)], dtype=np.float32)
    tvec = np.array([cam_params['depth_to_rgb']['t'][i] for i in range(3)], dtype=np.float32) / 1000 # mm > m 
    rotmtx = cv2.Rodrigues(np.array(rvec))[0]

    # print(f"cam_parmas : {cam_params['color_resolution']}") # cam_parmas : {'h': 720, 'w': 1280}

    return {
        'K_rgb_distorted': K_rgb_distorted,
        'rgb_distortion': rgb_distortion,
        'rvec': rvec,
        'rotmtx': rotmtx,
        'tvec': tvec,
        'color_resolution': cam_params['color_resolution'] # cam_parmas : {'h': 720, 'w': 1280}
    }

def bbox_full_body(
    img,
    body_joints_3d,          # (N,3) Kinect or SMPL joints
    cam_params,
    scale=1.2,
    out_size=(256, 512)      # 체형 추론용 (세로 길게 추천)
):
    h, w = img.shape[:2]

    # 1. 3D joints → 2D projection
    joints_2d, _ = cv2.projectPoints(
        body_joints_3d.astype(np.float32),
        cam_params['rvec'],
        cam_params['tvec'],
        cam_params['K_rgb_distorted'],
        cam_params['rgb_distortion']
    )
    joints_2d = joints_2d.reshape(-1, 2)

    # 유효한 관절만 사용 (NaN, inf 방지)
    valid = np.isfinite(joints_2d).all(axis=1)
    joints_2d = joints_2d[valid]

    if joints_2d.shape[0] == 0:
        return None, None

    # 2. bbox 계산
    lu = joints_2d.min(axis=0) # [x_min, y_min]
    rb = joints_2d.max(axis=0) # [x_max, y_max]

    # 3. bbox 확장 (여유 주기)
    center = (lu + rb) / 2.0
    size = (rb - lu) * scale
    lu = center - size / 2.0
    rb = center + size / 2.0

    # 4. image boundary clip
    lu = np.clip(lu, [0, 0], [w - 1, h - 1]).astype(int)
    rb = np.clip(rb, [0, 0], [w - 1, h - 1]).astype(int)
    
    bbox = [lu[0], lu[1], rb[0], rb[1]]

    # 5. 디버그용
    crop = img[lu[1]:rb[1], lu[0]:rb[0], :3].copy()
    
    # 조인트 디버그용 
    x0, y0, x1, y1 = bbox  # lu.x, lu.y, rb.x, rb.y
    for (u, v) in joints_2d:
        u_crop = int(round(u - x0))
        v_crop = int(round(v - y0))

        if 0 <= u_crop < crop.shape[1] and 0 <= v_crop < crop.shape[0]:
            cv2.circle(
                crop,
                (u_crop, v_crop),
                radius=4,
                color=(0, 255, 0),
                thickness=-1
            )

    if crop.size == 0:
        return None, None

    return crop, bbox


def plot_kinect_joint_2d(
    img,
    body_joints_3d,          # (N,3) Kinect joints (RGB frame)
    cam_params,
    kinect_bones
):
    # 전체 이미지 복사 (OpenCV write 가능)
    img_rgb = img[:, :, :3].copy()
    h, w = img_rgb.shape[:2]


    # 1. 3D joints → 2D projection (distorted RGB image space)
    joints_2d, _ = cv2.projectPoints(
        body_joints_3d.astype(np.float32),
        cam_params['rvec'],
        cam_params['tvec'],
        cam_params['K_rgb_distorted'],
        cam_params['rgb_distortion']
    )
    joints_2d = joints_2d.reshape(-1, 2)

    # 유효한 관절만
    valid = np.isfinite(joints_2d).all(axis=1)
    joints_2d = joints_2d[valid]

    if joints_2d.shape[0] == 0:
        return None

    # 2. 전체 이미지 좌표계에 그대로 draw
    for a, b in kinect_bones:
        if not (valid[a] and valid[b]):
            continue

        xa, ya = joints_2d[a]
        xb, yb = joints_2d[b]

        xa, ya = int(round(xa)), int(round(ya))
        xb, yb = int(round(xb)), int(round(yb))

        if (0 <= xa < w and 0 <= ya < h and
            0 <= xb < w and 0 <= yb < h):
            cv2.line(
                img_rgb,
                (xa, ya),
                (xb, yb),
                color=(255, 255, 0),  # cyan
                thickness=2
            )

    return img_rgb

def get_joint_2d_from_3d(
    body_joints_3d,          # (N,3) Kinect Joint
    cam_params,
):

    # 1. 3D joints → 2D projection
    joints_2d, _ = cv2.projectPoints(
        body_joints_3d.astype(np.float32),
        cam_params['rvec'],
        cam_params['tvec'],
        cam_params['K_rgb_distorted'],
        cam_params['rgb_distortion']
    )
    joints_2d = joints_2d.reshape(-1, 2)

    return joints_2d

def bbox_to_cliff_format(lu, rb, img_shape, cam_params):
    h, w = img_shape[:2]

    # center
    cx = (lu[0] + rb[0]) / 2
    cy = (lu[1] + rb[1]) / 2

    # scale (CLIFF 기준)
    box_size = max(rb[0] - lu[0], rb[1] - lu[1])
    b = box_size / 200.0   # CLIFF convention

    focal = cam_params['K_rgb_distorted'][0, 0]

    bbox_info = np.array([
        (cx - w / 2) / focal * 2.8,
        (cy - h / 2) / focal * 2.8,
        (b * 200 - 0.24 * focal) / (0.06 * focal)
    ], dtype=np.float32)

    return bbox_info

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not any(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


class ShapeBetaBuffer:
    def __init__(self, max_len=10):
        self.max_len = max_len
        self.buffer = []

    def push(self, beta):  # beta: (10,)
        self.buffer.append(beta)
        if len(self.buffer) > self.max_len:
            self.buffer.pop(0)
        return len(self.buffer) == self.max_len
        

    def is_ready(self):
        return len(self.buffer) == self.max_len

    def get_mean(self):
        assert self.is_ready()
        return np.mean(np.stack(self.buffer, axis=0), axis=0)  # (10,)


class HRCSensingNode(Node):
    def __init__(self):
        super().__init__('hrc_sensing_node')
        
        # Publisher 생성 (토픽명: /hrc/human_state)
        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # 큐 사이즈 1: 최신 데이터만 중요하므로, 밀리면 과감히 버림 (Real-time성 확보)
        self.pub = self.create_publisher(HumanPose, '/hrc/human_pose', qos_profile)
        self.sub = self.create_subscription(HumanState, '/hrc/human_state', self.state_callback,qos_profile)
        self.collision_sub = self.create_subscription(CollisionCandidates, '/hrc/collision_candidates', self.collision_candidates_callback,qos_profile)
        self.demo_state_sub = self.create_subscription(
            DemoState,
            '/hrc/demo_state',
            self.demo_state_callback,
            qos_profile
        )
        self.worst_head_meff_sub = None


        self.bridge = CvBridge()

        CONFIG_PATH = "/home/robotics/hrc_ws/src/hrc_body_tracking/server/renat.yaml"
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        self.k4a_processor_config=config['k4a']
        self.body_processor_config = config['body_pose']
        self.aggregate_vis_processor_config = config['aggregate']['vis_pose']
        self.aggregate_filterer_processor_config = config['aggregate']['filterer']
        self.pykinect_data_dp = osp.join(config['data_dirpath'], 'pykinect') # pykinect
        self.aggregate_filterer_processor_config['body_models_dp'] = osp.join(self.pykinect_data_dp, 'body_models')
        self.data_dir = config['data_dirpath']
        self.pose_npz_dir = '/home/robotics/hrc_ws/pose_npz'
        self.enable_filter = config['enable_filter']
    

        # 오래된 프레임 스킵 설정(시스템이 느릴 때 최신 프레임 ㅇ유지를 위해)
        self.skip_old_atol_ms = 0.2 if self.k4a_processor_config.get('skip_old', False) else None
        # Color 이미지의 타임 스탬프를 가져올지 여부
        self.get_color_timestamp = self.k4a_processor_config.get('get_color_timestamp', True)
        # Body Tracking 연산을 별도 스레드에서 돌릴지 여부
        self.parallel_bt = self.k4a_processor_config.get('parallel_bt', True) 
        self.log_level = self.k4a_processor_config.get('log_level', 2) # 로그 레벨 출력
        self.gpu_id = self.k4a_processor_config.get('gpu_id', 0)
       

        # --- 상태 변수 초기화 ---- # 
        self.prev_time = time.time()
        self.k4a = None # 카메라 객체 (아직 연결 안함)
        self.kinect_dump_frames = None # 덤프 파일 데이터 저장소
        self.decoded_cam_params = None # 카메라 캘리브레이션 정보
        self.index = -1                 # 프레임 번호 카운트
        self.dump_index = -1            # 덤프 파일 재생 인덱스
        self.output_count = 0           # 출력 횟수
        self.R_imu = None               # IMU 회전 행렬 (지금 사용 안됨)
        
        self.fps = self.k4a_processor_config['fps'] 

        'body pose'
        self.device = torch.device(self.body_processor_config['device']) # cuda 0
        self.gender = self.body_processor_config['gender'] # male
        self.beta = np.load(osp.join(
            self.data_dir,
            self.body_processor_config['person_shape_path'],
            'betas.npy'
        ))
        self.model_dp = osp.join(self.data_dir, self.body_processor_config['model_path']) # "smplx_kinect/04"
        self.inferer = None
        self.cliff_beta = None

        'aggregate'
        self.vis_config = self.aggregate_vis_processor_config
        self.visualization_enabled = bool(self.vis_config.get('enable', False))
        self.show_activation_sphere = bool(
            self.vis_config.get('show_activation_sphere', False)
        )
        self.find_worst_vis = bool(
            self.vis_config.get('find_worst_vis', False)
        )
        self.find_worst_target_index = int(
            self.vis_config.get('find_worst_target_index', 12)
        )
        self.worst_head_arrow_length_m = float(
            self.vis_config.get('worst_head_arrow_length_m', 0.35)
        )
       
        self.body_part_names = [
            "Pelvis",
            "Thigh",
            "Thigh",
            "Abdomen",
            "Calf",
            "Calf",
            "Foot",
            "Foot",
            "Chest",
            "Neck",
            "Shoulder",
            "Shoulder",
            "Head",
            "Upper Arm",
            "Upper Arm",
            "Lower Arm",
            "Lower Arm",
            "Hand",
            "Hand",
        ]

        self.body_part_to_smpl_joint_index = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 7,
            7: 8,
            8: 6,
            9: 12,
            10: 13,
            11: 14,
            12: 15,
            13: 16,
            14: 17,
            15: 18,
            16: 19,
            17: 20,
            18: 21,
        }
        self.iso_json_path = "/home/robotics/indy_ws/src/indy_control_cpp/ISO15066/iso_15066.json"
        self.iso_fixed_mass_by_body_name = self._load_iso_fixed_mass_map(
            self.iso_json_path
        )
        if self.find_worst_vis:
            self.worst_head_meff_sub = self.create_subscription(
                DirectionalMeffResult,
                '/hrc/worst_head_meff',
                self.worst_head_meff_callback,
                qos_profile,
            )
        self.activation_sphere_center_world = np.array(
            [0.0, 0.0, 0.6],
            dtype=np.float32,
        )
        # radius
        self.activation_sphere_radius = 1.3 
        self.camera_rotation = camera_to_world_rotmat(
            roll=110.8,
            pitch=0.0,
            yaw=0.0,
            device="cpu",
        ).cpu().numpy()
        self.camera_position_world = np.array(
            [-0.80, -2.3, 1.55],
            dtype=np.float32,
        )
        self.activation_sphere_projection = None
        self.smplx_wrapper = None
        self.smpl_wrapper = None
        self.visualizer = None
        self.smpl_visualizer = None
        self.vis_smplx = None
        self.vis_smpl = None
        self.vis_dp = "/home/robotics/hrc_ws/vis_output"
        if self.visualization_enabled:
            os.makedirs(self.vis_dp, exist_ok=True)

            # 새 실험 시작 시 기존 저장 프레임 삭제
            for img_path in glob.glob(osp.join(self.vis_dp, "*.jpg")):
                try:
                    os.remove(img_path)
                except Exception as e:
                    self.get_logger().warn(f"Failed to remove old frame {img_path}: {e}")

        self.filterer = None
        self.decoded_cam_params = None
        self.smplx_device = None
        self.visualize_device = None
        self.prev_time = time.time()
        self.x ={}

        '시각화를 위한 작업'
        self.smpl = None
        self.vertex_colorweights_data = None
        self.mass_colorizer = None
        self.collision_colorizer = None
        if self.visualization_enabled:
            self.smpl = SMPL(device=self.device)
            self.vertex_colorweights_data = self.smpl.weights.detach().cpu().numpy()
            self.mass_colorizer = EffectiveMassColorizer19(
                vertex_weights_24=self.vertex_colorweights_data,
                device=self.device,
            )
            self.collision_colorizer = CollisionTop3Colorizer19(
                vertex_weights_24=self.vertex_colorweights_data,
                device=self.device,
            )
        self.prev_collision_target_top3 = None
        

        self.effective_mass = None
        self.mass_vertex_colors = None

        self.collision_target_top3  = None
        self.collision_vertex_colors = None
        self.collision_vertex_alpha = None
        self.faces=None

        # demo time
        self.latest_demo_state = None
        self.latest_demo_state_lock = threading.Lock()
        self.latest_worst_head_meff = None
        self.latest_worst_head_meff_lock = threading.Lock()
        self.has_person_in_frame = False
        self.has_person_lock = threading.Lock()
        self.perf_logger = PerfCsvLogger(
            "/home/robotics/hrc_ws/analysis/realtime_perf/"
            "paper_human_sensing_perf.csv",
            [
                "t_ros_sec",
                "frame_index",
                "capture_ms",
                "pose_inference_ms",
                "visualization_ms",
                "publish_ms",
                "total_operation_ms",
                "nominal_deadline_ms",
                "deadline_miss",
            ],
            flush_every=30,
        )
        # self.collision_msg_received = False

        # debug
        self.prev_debug_pose = None          # (32, 3)
        self.debug_jump_thresh_mean = 0.1   # 관절 평균 이동량 임계값 [m]
        self.debug_jump_thresh_pelvis = 0.2 # pelvis 이동량 임계값 [m]
                            

        # -----------------------------
        # render thread state 시각화 모드
        # -----------------------------
        self.render_mode = "full_smpl_collision_top3"   # "full_smpl_collision_top3", "only_top3", "effectivemass_vis"
        self.render_thread_enable = self.visualization_enabled

        self._render_lock = threading.Lock()
        self._render_event = threading.Event()
        self._render_stop = False
        self._latest_render_packet = None
        self._render_thread = None
        if self.render_thread_enable:
            self._render_thread = threading.Thread(
                target=self._render_worker_loop,
                name="hrc_render_worker",
                daemon=True,
            )
            self._render_thread.start()

        self.meff_example_19 = torch.tensor(
            [
                21.5,  # 0 pelvis
                15.2,  # 1 L_hip
                16.8,  # 2 R_hip
                17.0,  # 3 spine1
                14.9,  # 4 L_knee
                15.5,  # 5 R_knee
                16.0,  # 6 L_ankle
                16.3,  # 7 R_ankle
                20.1,  # 8 spine2
                7.6,   # 9 neck
                5.8,   # 10 L_collar
                6.2,   # 11 R_collar
                10,   # 12 head
                3.4,   # 13 L_shoulder
                3.9,   # 14 R_shoulder
                1.6,   # 15 L_elbow
                1.9,   # 16 R_elbow
                0.45,  # 17 L_wrist
                0.62,  # 18 R_wrist
            ],
            dtype=torch.float32,
            device=self.device
        )
        self.kinect_bones = [
            # spine / head
            (0, 1),   # pelvis -> spine_naval
            (1, 2),   # spine_naval -> spine_chest
            (2, 3),   # spine_chest -> neck
            (3, 26),  # neck -> head
            (26, 27), # head -> nose
            (27, 28), # nose -> eye_left
            (28, 29), # eye_left -> ear_left
            (27, 30), # nose -> eye_right
            (30, 31), # eye_right -> ear_right

            # left arm
            (2, 4),   # spine_chest -> clavicle_left
            (4, 5),   # clavicle_left -> shoulder_left
            (5, 6),   # shoulder_left -> elbow_left
            (6, 7),   # elbow_left -> wrist_left
            (7, 8),   # wrist_left -> hand_left
            (8, 9),   # hand_left -> handtip_left
            (7, 10),  # wrist_left -> thumb_left

            # right arm
            (2, 11),  # spine_chest -> clavicle_right
            (11, 12), # clavicle_right -> shoulder_right
            (12, 13), # shoulder_right -> elbow_right
            (13, 14), # elbow_right -> wrist_right
            (14, 15), # wrist_right -> hand_right
            (15, 16), # hand_right -> handtip_right
            (14, 17), # wrist_right -> thumb_right

            # left leg
            (0, 18),  # pelvis -> hip_left
            (18, 19), # hip_left -> knee_left
            (19, 20), # knee_left -> ankle_left
            (20, 21), # ankle_left -> foot_left

            # right leg
            (0, 22),  # pelvis -> hip_right
            (22, 23), # hip_right -> knee_right
            (23, 24), # knee_right -> ankle_right
            (24, 25), # ankle_right -> foot_right
        ]
        

        '체형 추론'
        self.shape_buffer = ShapeBetaBuffer(max_len= 10)
        self.model = CLIFF(str(mean_params_path)).to(self.device)
        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device
        )["model"]
        state_dict = strip_prefix_if_present(state_dict, prefix="module.")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.is_ready_shape = False
        self.image_processor = Img_Processor()


        '임시 저장 로그'
        self.log_buffer = {
            "pose" : [],
            "trans" :[],
        }

        # 타이머 설정 (30Hz = 약 0.033초)
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)

    def timer_callback(self):
        self.index += 1         # 프로 세스 번호 증가
        init_done = False       # 초기화 수행 여부 체크 
        timings ={}

        try:
            if self.k4a is None:
                if self.fps == 30:
                    fps_config = pyk4a.FPS.FPS_30
                elif self.fps == 15:
                    fps_config = pyk4a.FPS.FPS_15
                else:
                    raise Exception(f'fps {self.fps} not found')
                
                """
                성능
                color_resolutuin RES_720P(0.02s)  RES_1536P(0.)
                depth_mode 정확도:  NFOV_UNBINNED > NFOV_2X2BINNED > WFOV_2X2BINNED
                """
                # 해상도
                k4a = PyK4A(Config(
                    color_resolution=ColorResolution.RES_720P, # RES_1536P, or RES_720P 
                    depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED, # NFOV_UNBINNED or WFOV_2X2BINNED
                    camera_fps=fps_config,
                    gpu_id=self.gpu_id  # 0
                ))
                k4a.connect(lut=True)
                self.decoded_cam_params = cam_params_decode(k4a.get_cam_params())
                if self.show_activation_sphere:
                    self.activation_sphere_projection = project_sphere_wireframe(
                        center_world=self.activation_sphere_center_world,
                        radius=self.activation_sphere_radius,
                        camera_rotation=self.camera_rotation,
                        camera_position_world=self.camera_position_world,
                        camera_matrix=self.decoded_cam_params['K_rgb_distorted'],
                        distortion=self.decoded_cam_params['rgb_distortion'],
                    )
                self.k4a = k4a

            if self.inferer is None and self.cliff_beta is not None:
                self.inferer = Inferer(
                    model_dp=self.model_dp,
                    checkpoint=self.body_processor_config['checkpoint'],
                    device=self.device,
                    beta=self.cliff_beta, # 학습된 beta 사용 ?
                    gender=self.gender,
                    pykinect_data_dp=self.pykinect_data_dp
                )

            start_time = time.time()
            log_time = time.time()

            d = self.k4a.get_capture2(
                verbose=self.fps if self.log_level >= 2 else 0,     # 로그 레벨
                skip_old_atol_ms=self.skip_old_atol_ms,             # 오래된 프레임 스킵 설정
                get_color_timestamp=self.get_color_timestamp,       # 시간 정보
                get_color=True, undistort_color=False,              # 컬러 이미지 가져옴
                get_depth=False,                                    # Depth 이미지는 안 가져옴 (Body Tracking 내부에서만 씀)
                get_bt=True, undistort_bt=False, # Undistort_.. 카메라 렌즈의 왜곡 보정
                parallel_bt=self.parallel_bt # Body Tracking 연산을 이미지 처리와 병렬 수행
            )
            '시간 측정'
            torch.cuda.synchronize()
            t0 = time.time()
            timings["Capture"] = (t0-start_time) * 1000

            # color / pose 키 자체가 없으면 이번 프레임은 스킵
            if ('color' not in d) or ('pose' not in d):
                return

            img = d['color'][..., :3]  # 3채널 BGR만 씀

            # 사람이 안 잡히면 SMPL 렌더는 하지 않고, 배경만 계속 갱신
            if len(d['pose']) < 1:
                self.collision_target_top3 = None
                self.collision_vertex_colors = None
                self.prev_collision_target_top3 = None
                with self.latest_demo_state_lock:
                    self.latest_demo_state = None    
                with self.has_person_lock:
                    self.has_person_in_frame = False


                if VIS and self.visualization_enabled and self.vis_config['vis_smpl']:
                    save_path = None
                    if self.vis_config['imsave']:
                        save_path = osp.join(self.vis_dp, f'{self.index:06d}.jpg')

                    with self.latest_demo_state_lock:
                        # demo_overlay = deepcopy(self.latest_demo_state)
                        demo_overlay = None

                    bg_img_for_render = img.copy()

                    if self.smpl_visualizer is not None:
                        target_h = self.smpl_visualizer.pinhole2d.h
                        target_w = self.smpl_visualizer.pinhole2d.w

                        if bg_img_for_render.shape[0] != target_h or bg_img_for_render.shape[1] != target_w:
                            bg_img_for_render = cv2.resize(bg_img_for_render, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                    render_packet = {
                        "smpl_vertices": None,
                        "target_center_world": None,
                        "worst_head_meff": None,
                        "bg_img": bg_img_for_render.copy(),
                        "mass_vertex_colors": None,
                        "collision_vertex_colors": None,
                        "render_mode": "bg_only",
                        "save_path": save_path,
                        "do_imshow": bool(self.vis_config['imshow']),
                        "demo_overlay": demo_overlay,
                        "activation_sphere_projection":
                            self.activation_sphere_projection,
                    }
                    self._submit_render_packet(render_packet)

                return

            # self.get_logger().info(f" body_pose : {d['pose']}") # if not d['pose']: 조건문

            if len(d['pose']) > 1:
                self.get_logger().info(f" 2 person error !! ")

            with self.has_person_lock:
                self.has_person_in_frame = True

            # ===== debug: warn only when pose suddenly jumps =====
            # debug_pose = d['pose'][0, :, 2:5] / 1000.0   # (32, 3), meter

            # if self.prev_debug_pose is not None and self.prev_debug_pose.shape == debug_pose.shape:
            #     joint_diff = np.linalg.norm(debug_pose - self.prev_debug_pose, axis=1)  # (32,)
            #     mean_jump = float(np.mean(joint_diff))
            #     max_jump = float(np.max(joint_diff))
            #     pelvis_jump = float(np.linalg.norm(debug_pose[0] - self.prev_debug_pose[0]))

            #     if pelvis_jump > 0.30:
            #         self.get_logger().warn(
            #             f"[pose0 identity jump suspicious] "
            #             f"pelvis_jump={pelvis_jump:.3f} m | "
            #             f"mean_jump={mean_jump:.3f} m | "
            #             f"max_jump={max_jump:.3f} m"ㅇㅇㅇ
            #         )

            # self.prev_debug_pose = debug_pose.copy()
            # ===== end debug =====
            
            # 첫번째 사진 모든 관절의 x,y,z 좌표만 슬라이싱
            body_pose = d['pose'][0, :, 2:5] / 1000
            # 신뢰도 점수 추출
            body_conf = d['pose'][0, :, -1]

    

            '체형 crop 디버그'
            with torch.no_grad():
                if self.is_ready_shape != True:
                    body_crop_vis_debug, bbox = bbox_full_body(
                        img,
                        body_pose,  # (32,3) Kinect joints
                        self.decoded_cam_params,
                        scale=1.2,
                    )
                    if body_crop_vis_debug is None:
                        return None
                    if bbox is None:
                        return None
                    
                    # --- CLIFF 입력 전처리 ---
                    img_Y = self.image_processor.forward(img_rgb=img, bbox=bbox)
                    if img_Y is None:
                        self.get_logger().warn("Img processor returned None")
                        return None

                    norm_img = torch.from_numpy(img_Y["norm_img"]).unsqueeze(0).to(self.device).float()
                    center = torch.as_tensor(img_Y["center"], device=self.device).float().unsqueeze(0)
                    scale  = torch.as_tensor(img_Y["scale"],  device=self.device).float().unsqueeze(0)
                    img_h  = torch.tensor(img_Y["img_h"], device=self.device).float().unsqueeze(0)
                    img_w  = torch.tensor(img_Y["img_w"], device=self.device).float().unsqueeze(0)
                    focal_length = torch.tensor(img_Y["focal_length"], device=self.device).float().unsqueeze(0)

                    cx, cy, b = center[:, 0], center[:, 1], scale * 200
                    bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
                    bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
                    bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]
                    pred_pose, pred_beta, _ = self.model(norm_img, bbox_info)

                    beta = pred_beta[0].cpu().numpy()  # (10,)
                    self.is_ready_shape = self.shape_buffer.push(beta) # 체형이 다 쌓이면
                    self.get_logger().info(f" is_ready_shape : {self.is_ready_shape}")

                    if body_crop_vis_debug is not None:
                        save_dir = "/home/robotics/hrc_ws/body_crop"
                        os.makedirs(save_dir, exist_ok= True)
                        save_path = os.path.join(save_dir, f"body_crop_{self.index}.jpg")
                        cv2.imwrite(save_path, body_crop_vis_debug)
                        cv2.imshow("body", body_crop_vis_debug)
                        cv2.waitKey(1)

                    if self.is_ready_shape:
                        beta_mean = self.shape_buffer.get_mean()
                        self.cliff_beta = beta_mean # (10,)
                        self.get_logger().info(f" cliff beta mean : {beta_mean}")
                        
            '체형 crop 디버그'

            'kinect 2d 디버그'
            # img_joint_vis = plot_kinect_joint_2d(
            #             img,
            #             body_pose,  # (32,3) Kinect joints
            #             self.decoded_cam_params,
            #             self.kinect_bones
            #         )
            # cv2.imshow("kinect joints (RGB)", img_joint_vis)
            'kinect 2d 디버그'

            # 양 손 사이의 유클리드 거리 계산
            hand_dist = np.sqrt(np.sum((body_pose[8] - body_pose[15]) ** 2))
            # 두 손의 거리가 0.2m 미만이면 True
            is_close_hands = hand_dist < 0.2
         
            w = img.shape[1]
            h = img.shape[0]
        
            # Depth 카메라 좌표계 -> rgb 카메라 좌표계로 옮김
            body_pose_rgb = body_pose @ self.decoded_cam_params['rotmtx'].T + \
                        self.decoded_cam_params['tvec']
            
            # smpl
            kinect_joints = body_pose_rgb
            """
            return {
            'input_kinect_joints': input_kinect_joints,
            'body_pose': body_pose,
            'global_rot': global_rot,
            'global_trans': global_trans
            }
            """
            if self.inferer is None:
                self.get_logger().warn("Inferer not ready yet, skipping frame")
                return None

         
            inf_result = self.inferer.inf(kinect_joints)
            inf_result['is_close_hands'] = is_close_hands
            inf_result['image'] = img
            inf_result['time'] = start_time
            # trans = inf_result['global_trans'] 
            # self.get_logger().info(f" trans shape : {trans.shape}")
      
            '시간 측정'
            torch.cuda.synchronize()
            t1 = time.time()
            timings["Inf"] = (t1-t0) * 1000
            
            '시각화'
            if self.filterer is None:
                self.smplx_device = torch.device(self.aggregate_filterer_processor_config['device']) # cpu
                self.filterer = Filterer(**self.aggregate_filterer_processor_config)
                if self.visualization_enabled:
                    self.visualize_device = torch.device(self.vis_config['device']) #  cuda 0

                    self.smplx_wrapper = HandsSMPLXWrapper(
                        body_models_dp=self.aggregate_filterer_processor_config['body_models_dp'],
                        use_pca=False, device=self.smplx_device, flat_hand_mean=True # cpu
                    )
                    
                    self.smpl_wrapper = SMPLWrapper(
                        body_models_dp=self.aggregate_filterer_processor_config['body_models_dp'],
                        device=self.smplx_device # cpu
                    )
                
                    if VIS==True:
                        if self.vis_config['vis_smplx']:
                            self.visualizer = VisualizerMeshSMPLX(
                                K=self.decoded_cam_params['K_rgb_distorted'],
                                scale=self.vis_config.get('scale', 1),
                                size=self.decoded_cam_params['color_resolution'],
                                device=self.visualize_device,
                                smplx_wrapper=self.smplx_wrapper,
                                sides=False,
                            )
                        elif self.vis_config['vis_smpl']:
                            self.smpl_visualizer = VisualizerMeshSMPL(
                                K=self.decoded_cam_params['K_rgb_distorted'],
                                scale=self.vis_config.get('scale', 1),
                                size=self.decoded_cam_params['color_resolution'],
                                device=self.visualize_device,
                                smpl_wrapper=self.smpl_wrapper,
                                sides=False,  
                                normals=True
                            )
                            self.faces = self.smpl_wrapper.models['male'].faces_tensor.to(
                                dtype=torch.int32,
                                device=self.device
                            )
                        else:
                            self.get_logger().warn(f"⚠️ Visualizer Error ")

            '필터'
            if self.enable_filter:
                x= inf_result
                x = {
                    'body_pose': {
                        'input_kinect_joints': x['input_kinect_joints'],
                        'body_pose': x['body_pose'],
                        'global_rot': x['global_rot'],
                        'global_trans': x['global_trans'],
                        'is_close_hands': x['is_close_hands'],
                        # 'image': x['image'],
                        'time': x['time'],
                    }
                }
                x = self.filterer.filter(x)  # 1~2 ms
                x['smpl_body_pose'] = np.zeros((69,), dtype=np.float32)
                x['smpl_body_pose'][:63] = x['body_pose']
                # smpl용 trans
                trans_smpl = x['global_trans'].astype(np.float32).copy()
                trans_smpl[1] -= 0.1
                x['global_trans'] = trans_smpl
            else :
                '필터 x'
                x= inf_result
                x['right_hand_pose'] = np.zeros((45,), dtype=np.float32)
                x['left_hand_pose'] = np.zeros((45,), dtype=np.float32)
                x['smpl_body_pose'] = np.zeros((69,), dtype=np.float32)
                x['smpl_body_pose'][:63] = x['body_pose']
                # smpl용 trans
                trans_smpl = x['global_trans'].astype(np.float32).copy()
                trans_smpl[1] -= 0.1
                x['global_trans'] = trans_smpl

            '임시 파일 저장'
            body_full_pose = np.concatenate([x['global_rot'], x['body_pose']], axis= 0 ).astype(np.float32) # (66,)
            self.log_buffer["pose"].append(body_full_pose.astype(np.float32))
            self.log_buffer["trans"].append(x["global_trans"].astype(np.float32))

            smpl_output = None
            if self.visualization_enabled and self.cliff_beta is not None:
                torch.cuda.synchronize()
                vis_start = time.time()
               
                if self.vis_config['vis_smplx']:
                    smplx_output = self.smplx_wrapper.get_output(
                        gender=self.gender,
                        betas=self.cliff_beta,
                        body_pose=x['body_pose'], # (63,)
                        rvec=x['global_rot'], # (3,)
                        tvec=x['global_trans'],  # (3,)
                        left_hand_pose=x['left_hand_pose'], # (45,)
                        right_hand_pose=x['right_hand_pose'], # (45,)
                        expression=x.get('face_expression', None),
                        jaw_pose=x.get('jaw_pose', None)
                    )
                    vertices = torch.tensor(smplx_output.vertices[0].contiguous()).to(
                        device=self.visualize_device)
                    if VIS==True:
                        vis = self.visualizer.vertices2vis(vertices,bg_img=img) 
                        if self.vis_config['imsave']:
                            cv2.imwrite(
                                osp.join(self.vis_dp, f'{self.index:06d}.jpg'), vis)
                        if self.vis_config['imshow']:
                            cv2.imshow('a', vis)
                            cv2.waitKey(1)
                # smplx

                elif self.vis_config['vis_smpl']:
                    smpl_output = self.smpl_wrapper.get_output(
                        gender=self.gender,
                        betas=self.cliff_beta[:10],
                        body_pose=x['smpl_body_pose'],
                        rvec=x['global_rot'],
                        tvec=x['global_trans'],
                    )

                    smpl_vertices = torch.tensor(
                        smpl_output.vertices[0].contiguous()
                    ).to(device=self.visualize_device)

                    if VIS:
                        # -----------------------------
                        # render input preparation only
                        # -----------------------------
                        img_vis = img.copy()

                        # 유효질량 전체 시각화는 해당 모드에서만 계산
                        if self.render_mode == "effectivemass_vis":
                            if self.effective_mass is not None:
                                self.mass_vertex_colors = self.mass_colorizer.get_vertex_colors(
                                    self.effective_mass
                                )
                            else:
                                self.mass_vertex_colors = None
                        else:
                            self.mass_vertex_colors = None

                        # collision Top3 색은 top3가 바뀔 때만 갱신
                        if self.collision_target_top3 is not None:
                            if self.prev_collision_target_top3 != self.collision_target_top3:
                                self.collision_vertex_colors = self.collision_colorizer.get_vertex_overlay(
                                    self.collision_target_top3
                                )[0]
                                self.prev_collision_target_top3 = list(self.collision_target_top3)
                        else:
                            self.collision_vertex_colors = None
                            self.prev_collision_target_top3 = None

                        save_path = None
                        if self.vis_config['imsave']:
                            save_path = osp.join(self.vis_dp, f'{self.index:06d}.jpg')
                        
                        with self.latest_demo_state_lock:
                            demo_overlay = deepcopy(self.latest_demo_state)
                        with self.latest_worst_head_meff_lock:
                            worst_head_meff = deepcopy(self.latest_worst_head_meff)

                        target_center_world = None
                        if smpl_output is not None and self.find_worst_vis:
                            smpl_joints_np = (
                                smpl_output.joints[0]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )
                            target_index = self.find_worst_target_index
                            if worst_head_meff is not None:
                                target_index = int(
                                    worst_head_meff.get(
                                        "target_index",
                                        target_index,
                                    )
                                )
                            smpl_joint_index = self.body_part_to_smpl_joint_index.get(
                                target_index,
                                target_index,
                            )
                            if 0 <= smpl_joint_index < smpl_joints_np.shape[0]:
                                target_camera = smpl_joints_np[smpl_joint_index]
                                target_center_world = (
                                    self.camera_rotation.T @ target_camera
                                    + self.camera_position_world
                                ).astype(np.float32)

                        render_packet = {
                            "smpl_vertices": smpl_vertices.detach(),
                            "target_center_world": target_center_world,
                            "worst_head_meff": worst_head_meff,
                            "bg_img": img_vis.copy(),
                            "mass_vertex_colors": self.mass_vertex_colors,
                            "collision_vertex_colors": self.collision_vertex_colors,
                            "render_mode": self.render_mode,
                            "save_path": save_path,
                            "do_imshow": bool(self.vis_config['imshow']),
                            "demo_overlay": demo_overlay,
                            "activation_sphere_projection":
                                self.activation_sphere_projection,
                        }
                        self._submit_render_packet(render_packet)


                else :
                    self.get_logger().warn(f"⚠️ SMPL or SMPLX Vis Error ")

                torch.cuda.synchronize()
                t2 = time.time()
                timings["VIS"] = (t2-vis_start) * 1000
            else:
                # vis_pose.enable=false이면 시각화 처리와 CUDA 동기화를 모두 건너뛴다.
                t2 = time.time()
                timings["VIS"] = 0.0

            '메시지 생성'
            msg = HumanPose()
            # 카메라 좌표계 기준
            msg.body_full_pose = body_full_pose.tolist()
            msg.global_trans = x['global_trans'].astype(np.float32).tolist()
            if smpl_output is not None:
                smpl_betas = smpl_output.betas[0].detach().cpu()
                msg.smpl_betas = smpl_betas.reshape(-1).tolist()

                # 카메라 기준 결과 데이터
                verts = smpl_output.vertices[0].detach().cpu() # 6890, 3 -> 20670
                msg.smpl_vertices = verts.reshape(-1).tolist()
                smpl_joints = smpl_output.joints[0].detach().cpu() # 135
                msg.smpl_joint = smpl_joints.reshape(-1).tolist()
            else:
                msg.smpl_betas = np.asarray(
                    self.cliff_beta[:10],
                    dtype=np.float32,
                ).reshape(-1).tolist()
                msg.smpl_vertices = []
                msg.smpl_joint = []
    


            # Header 채우기 (시간 동기화 필수)
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "azure_kinect_base"

            '이미지 토픽 고려'
            "4. 이미지 담기 (무거운 작업, 원본 이미지를 Control Node로 보내 시각화/저장용으로 씀"

            # msg.image = self.bridge.cv2_to_imgmsg(img, encoding="bgr8") 
            # img_msg = CompressedImage()
            # img_msg.header = msg.header
            # img_msg.format = "jpeg"
            # _, enc = cv2.imencode(
            #     '.jpg',
            #     img,
            #     [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            # )
            # img_msg.data = enc.tobytes()

            # msg.image = img_msg
            '이미지 고려'
    
            # 5. 최종 패키징 및 발송
            self.pub.publish(msg)

            torch.cuda.synchronize()
      
            '시간 측정'
            torch.cuda.synchronize()
            t3 = time.time()
            timings["Topic_pub"] = (t3-t2) * 1000

            # 성능 모니터링 (너무 느리면 경고)
            process_time = time.time() - start_time
            timings["Total"] = (process_time) * 1000

            self.perf_logger.write(
                t_ros_sec=f"{self.get_clock().now().nanoseconds / 1e9:.9f}",
                frame_index=self.index,
                capture_ms=f"{timings['Capture']:.6f}",
                pose_inference_ms=f"{timings['Inf']:.6f}",
                visualization_ms=f"{timings['VIS']:.6f}",
                publish_ms=f"{timings['Topic_pub']:.6f}",
                total_operation_ms=f"{timings['Total']:.6f}",
                nominal_deadline_ms=f"{1000.0 / self.fps:.6f}",
                deadline_miss=int(timings["Total"] > (1000.0 / self.fps)),
            )
           
            # if process_time > 0.033:
            #     self.get_logger().warn(f"⚠️ Loop slow: {process_time*1000:.2f}ms")

            #--- [Log] 시간 정보 출력 ---
            # log_str = " | ".join([f"{k}: {v:.1f}ms" for k, v in timings.items()])
            # self.get_logger().info(log_str)



        except Exception as e:
            self.get_logger().error(f"Processing Error: {e}")

    def state_callback(self, msg:HumanState):

        effective_mass = np.array(msg.effmass, dtype=np.float32)
        effective_mass_torch = torch.from_numpy(effective_mass).type(torch.float32).to(self.device)
        self.effective_mass = effective_mass_torch

    def demo_state_callback(self, msg: DemoState):
        with self.latest_demo_state_lock:
            self.latest_demo_state = {
                "t_demo_sec": float(msg.t_demo_sec),
                "selected_link_name": msg.selected_link_name,
                "selected_body_name": msg.selected_body_name,
                "applied_speed_scale": float(msg.applied_speed_scale),
                "estimated_collision_force": float(msg.estimated_collision_force),
                "threshold_force": float(msg.threshold_force),
                "link_speed_magnitudes": list(msg.link_speed_magnitudes),
            }

    def worst_head_meff_callback(self, msg: DirectionalMeffResult):
        if not msg.valid:
            return

        direction = np.array(
            [
                msg.direction_u.x,
                msg.direction_u.y,
                msg.direction_u.z,
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            return

        display_target_index = self.find_worst_target_index
        if int(msg.target_index) != display_target_index:
            self.get_logger().warn(
                "worst-vis target index mismatch: "
                f"msg={int(msg.target_index)} config={display_target_index}. "
                "Using config index for ISO display."
            )

        with self.latest_worst_head_meff_lock:
            self.latest_worst_head_meff = {
                "direction_world": direction / norm,
                "meff": float(msg.directional_meff),
                "target_index": int(msg.target_index),
                "display_target_index": int(display_target_index),
                "target_name": self._get_body_part_name(int(display_target_index)),
                "iso_fixed_mass": float(
                    self.get_iso_fixed_mass(int(display_target_index))
                ),
            }

    def _submit_render_packet(self, packet: dict) -> None:
        if not self.render_thread_enable:
            return

        with self._render_lock:
            self._latest_render_packet = packet
            self._render_event.set()

    def draw_demo_overlay(self, vis, demo_overlay):
        if demo_overlay is None:
            return vis

        lines = [
            f"t = {demo_overlay['t_demo_sec']:.2f} s",
            # f"selected link = {demo_overlay['selected_link_name']}",
            f"body = {demo_overlay['selected_body_name']}",
            f"speed scale = {demo_overlay['applied_speed_scale']:.2f}",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        margin_x = 20
        margin_y = 30
        line_gap = 26

        # 먼저 텍스트 크기 계산
        sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
        max_w = max(w for w, h in sizes)

        # 우측 상단 고정
        x_fixed = int(vis.shape[1] - max_w - margin_x)
        y0 = int(margin_y)

        for i, line in enumerate(lines):
            y = int(y0 + i * line_gap)

            # 검은 외곽선
            cv2.putText(
                vis, line, (x_fixed, y),
                font, font_scale, (0, 0, 0), 3, cv2.LINE_AA
            )
            # 흰 글자
            cv2.putText(
                vis, line, (x_fixed, y),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
            )

        return vis

    def _get_body_part_name(self, target_index: int) -> str:
        if target_index == 15:
            return "Left Upper Arm"
        if 0 <= target_index < len(self.body_part_names):
            return self.body_part_names[target_index]
        return f"index_{target_index}"

    def _load_iso_fixed_mass_map(self, json_path: str) -> dict:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            result = {}
            for body_name, entry in data.items():
                try:
                    result[body_name] = float(entry["m_h_iso"])
                except Exception:
                    continue
            return result
        except Exception as e:
            self.get_logger().warn(f"Failed to load ISO fixed-mass table: {e}")
            return {}

    def get_iso_fixed_mass(self, target_index: int) -> float:
        if target_index == 15:
            return 3.0
        body_name = self._get_body_part_name(target_index)
        if body_name in self.iso_fixed_mass_by_body_name:
            return float(self.iso_fixed_mass_by_body_name[body_name])
        self.get_logger().warn(
            f"ISO fixed mass not found for body part '{body_name}', falling back to 4.4 kg"
        )
        return 4.4

    def draw_worst_target_text_overlay(self, vis, worst_target_meff):
        if worst_target_meff is None:
            return vis

        target_meff = float(worst_target_meff.get("meff", 0.0))
        target_index = int(
            worst_target_meff.get(
                "display_target_index",
                worst_target_meff.get("target_index", self.find_worst_target_index),
            )
        )
        target_name = worst_target_meff.get(
            "target_name",
            self._get_body_part_name(target_index),
        )
        iso_fixed_mass = float(
            worst_target_meff.get(
                "iso_fixed_mass",
                self.get_iso_fixed_mass(target_index),
            )
        )
        lines = [
            f"{target_name}: maximum mass direction",
            f"Effective mass = {target_meff:.1f} kg",
            f"ISO fixed mass = {iso_fixed_mass:.1f} kg",
        ]
        try:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            font_size = max(24, int(round(vis.shape[0] * 0.030)))
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        img_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img, "RGBA")

        margin_x = 22
        margin_y = 20
        pad_x = 14
        pad_y = 10
        line_gap = 7

        text_boxes = []
        max_w = 0
        total_h = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=1)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            text_boxes.append((w, h))
            max_w = max(max_w, w)
            total_h += h
        total_h += line_gap * (len(lines) - 1)

        box_x0 = margin_x
        box_y0 = margin_y
        box_x1 = box_x0 + max_w + 2 * pad_x
        box_y1 = box_y0 + total_h + 2 * pad_y

        cursor_y = box_y0 + pad_y
        for idx, line in enumerate(lines):
            w, h = text_boxes[idx]
            draw.text(
                (box_x0 + pad_x, cursor_y),
                line,
                font=font,
                fill=(20, 20, 20, 255),
                stroke_width=0,
            )
            cursor_y += h + line_gap

        out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        vis[:] = out
        return vis

    def draw_worst_target_meff_overlay(
        self,
        vis,
        target_center_world,
        worst_target_meff,
    ):
        if (
            not self.find_worst_vis
            or target_center_world is None
            or worst_target_meff is None
            or self.decoded_cam_params is None
        ):
            return vis

        direction = np.asarray(
            worst_target_meff.get("direction_world"),
            dtype=np.float32,
        ).reshape(3)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            return vis
        direction = direction / norm

        target_world = np.asarray(target_center_world, dtype=np.float32).reshape(3)
        tail_world = target_world + self.worst_head_arrow_length_m * direction

        pixels, valid = project_world_points_to_image(
            np.stack([target_world, tail_world], axis=0),
            camera_rotation=self.camera_rotation,
            camera_position_world=self.camera_position_world,
            camera_matrix=self.decoded_cam_params['K_rgb_distorted'],
            distortion=self.decoded_cam_params['rgb_distortion'],
        )
        if not (valid[0] and valid[1]) or not np.isfinite(pixels).all():
            return vis

        source_res = self.decoded_cam_params['color_resolution']
        scale = np.array(
            [
                vis.shape[1] / float(source_res["w"]),
                vis.shape[0] / float(source_res["h"]),
            ],
            dtype=np.float32,
        )
        target_px = tuple(np.rint(pixels[0] * scale).astype(np.int32))
        tail_px = tuple(np.rint(pixels[1] * scale).astype(np.int32))

        color = (0, 0, 255)
        outline = (20, 20, 20)
        cv2.arrowedLine(vis, tail_px, target_px, outline, 8, cv2.LINE_AA, tipLength=0.20)
        cv2.arrowedLine(vis, tail_px, target_px, color, 4, cv2.LINE_AA, tipLength=0.20)
        cv2.circle(vis, target_px, 8, outline, -1, cv2.LINE_AA)
        cv2.circle(vis, target_px, 5, color, -1, cv2.LINE_AA)
        return vis

    # legacy names retained for compatibility with any external calls
    def draw_worst_head_text_overlay(self, vis, worst_head_meff):
        return self.draw_worst_target_text_overlay(vis, worst_head_meff)

    def draw_worst_head_meff_overlay(
        self,
        vis,
        head_center_world,
        worst_head_meff,
    ):
        return self.draw_worst_target_meff_overlay(
            vis,
            head_center_world,
            worst_head_meff,
        )

    def draw_activation_sphere(self, vis, projected_curves):
        return draw_projected_sphere_overlay(
            image=vis,
            projection=projected_curves,
            source_resolution=self.decoded_cam_params['color_resolution'],
            radius=self.activation_sphere_radius,
        )

    def _render_worker_loop(self):
        while not self._render_stop:
            self._render_event.wait(timeout=0.1)

            if self._render_stop:
                break

            with self._render_lock:
                packet = self._latest_render_packet
                self._latest_render_packet = None
                self._render_event.clear()

            if packet is None:
                continue

            try:
                smpl_vertices = packet["smpl_vertices"]
                target_center_world = packet.get("target_center_world", None)
                worst_head_meff = packet.get("worst_head_meff", None)
                bg_img = packet["bg_img"]
                mass_vertex_colors = packet["mass_vertex_colors"]
                collision_vertex_colors = packet["collision_vertex_colors"]
                render_mode = packet["render_mode"]
                save_path = packet["save_path"]
                do_imshow = packet["do_imshow"]
                demo_overlay = packet.get("demo_overlay", None)
                activation_sphere_projection = packet.get(
                    "activation_sphere_projection",
                    None,
                )

                t0 = time.time()
                with self.has_person_lock:
                    has_person = self.has_person_in_frame

                if (not has_person) or smpl_vertices is None or render_mode == "bg_only":
                    vis = bg_img.copy()
                    if self.smpl_visualizer is not None:
                        target_h = self.smpl_visualizer.pinhole2d.h
                        target_w = self.smpl_visualizer.pinhole2d.w

                        if vis.shape[0] != target_h or vis.shape[1] != target_w:
                            vis = cv2.resize(vis, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                else:
                    # SMPL 시각화
                    # img_vis = img[::4, ::4].copy() # 픽셀 건너뛰기, 실시간 시각화에서는 계산 속도를 위해
                    vis = self.smpl_visualizer.vertices2vis_test(
                        smpl_vertices,
                        vertex_colors=mass_vertex_colors,
                        collision_vertex_colors=collision_vertex_colors,
                        bg_img=bg_img,
                        render_mode=render_mode,
                    )
                    # vis = self.smpl_visualizer.vertices2vis(smpl_vertices,bg_img=None)
                    # 텍스트 
                    if self.find_worst_vis:
                        vis = self.draw_worst_target_text_overlay(
                            vis,
                            worst_head_meff,
                        )
                    else:
                        vis = self.draw_demo_overlay(vis, demo_overlay)

                vis = self.draw_worst_target_meff_overlay(
                    vis,
                    target_center_world,
                    worst_head_meff,
                )

                vis = self.draw_activation_sphere(
                    vis,
                    activation_sphere_projection,
                )

                torch.cuda.synchronize()
                t1 = time.time()

                if save_path is not None:
                    cv2.imwrite(save_path, vis) # 이미지 저장은 고화질

                if do_imshow:
                    # vis = vis[::4, ::4] # 실시간 시각화느 저화질; 계산 비용 절약 
                    cv2.imshow("a", vis)
                    cv2.waitKey(1)

                t2 = time.time()
                # self.get_logger().info(
                #     f"[render thread] render={(t1-t0)*1000:.1f}ms | display/save={(t2-t1)*1000:.1f}ms"
                # )

            except Exception as e:
                self.get_logger().error(f"[render thread] Rendering Error: {e}")

    def collision_candidates_callback(self, msg: CollisionCandidates):
        
        if len(msg.human_indices) == 0:
            self.collision_target_top3 = None
            self.collision_vertex_colors = None
            self.prev_collision_target_top3 = None
            return

        self.collision_target_top3 = [int(idx) for idx in msg.human_indices[:3]]
        
    def _stop_render_thread(self):
        if not self.render_thread_enable:
            return

        self._render_stop = True
        self._render_event.set()
        if self._render_thread is not None and self._render_thread.is_alive():
            self._render_thread.join(timeout=1.0)

    def _cleanup_k4a(self):
        if getattr(self, 'k4a', None) is not None:
            try:
                print("🛑 [K4A] Disconnecting camera...")
                self.k4a.disconnect()
                print("✅ [K4A] Camera closed safely.")
            except Exception as e:
                print(f"⚠️ [K4A] Error during cleanup: {e}")
            finally:
                self.k4a = None

        if hasattr(self, "perf_logger"):
            self.perf_logger.close()

    def _cleanup_infere(self):
        if hasattr(self, 'inferer') and self.inferer is not None:
            print("🛑 [BodyPose] Releasing GPU resources...")
            self.inferer = None
            del self.inferer
            torch.cuda.empty_cache() # VRAM 비우기
            print("✅ [BodyPose] GPU Cleaned.")
    
    def _save_pose_npz(self):
        self.get_logger().info("Saving logged data to npz")

        try:
            save_dir = osp.join(self.pose_npz_dir)
            os.makedirs(save_dir, exist_ok=True)
            save_path = osp.join(save_dir, f"human_state_{int(time.time())}.npz")
            poses_np = np.array(self.log_buffer['pose'])
            trans_np = np.array(self.log_buffer['trans'])
            self.get_logger().info(f"poses shape: {poses_np.shape}, dtype: {poses_np.dtype}")
            self.get_logger().info(f"trans shape: {trans_np.shape}, dtype: {trans_np.dtype}")


            np.savez_compressed(
                save_path,
                poses = np.array(self.log_buffer['pose']),
                trans = np.array(self.log_buffer['trans'])
            )
            
            self.get_logger().info(f" saved: {save_path}")
        except Exception as e:
            self.get_logger().error(f" Failed to save log {e}")
        

    def destroy_node(self):
        self._stop_render_thread()
        self._save_pose_npz()
        self._cleanup_k4a()
        self._cleanup_infere()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()
    

def main(args=None):
    rclpy.init(args=args)
    node = HRCSensingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
