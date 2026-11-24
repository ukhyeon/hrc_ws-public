import rclpy
from rclpy.node import Node
from hrc_interfaces.msg import HumanPose, HumanState, HumanDynamicsState
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import (
    HistoryPolicy,
    ReliabilityPolicy,
    DurabilityPolicy
)
import os
import numpy as np
import torch
import time
from .utils import batch_rodrigues, rotmat2euleryzx,rotmat2eulerzxy,rotmat2eulerzyx, batch_roteulerSMPL, wrap_to_pi_torch, load_rom_config_19, camera_to_world_rotmat
from. import constants
from .physcis import Physics, SMPL,SMPLH

class DynamicsNode(Node):
    def __init__(self):
        super().__init__('human_dynamics_node')
        
        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.pub = self.create_publisher(HumanState, '/hrc/human_state',1)
        self.dynamics_pub = self.create_publisher(HumanDynamicsState, '/hrc/human_dynamics_state', qos_profile)
        self.sub = self.create_subscription(HumanPose, '/hrc/human_pose', self.pose_callback, qos_profile) 
        self.humanpose_msg =None
        self.pose_torch_camera = None
        self.betas = None
        # self.full_pose_torch_world = None
        self.pose_world = None
        self.q= None
        self.previous_q =None
        self.previous_com_world = None
        self.trans_torch_world = None
        self.link_velocity = torch.zeros((1, 19, 3), dtype=torch.float32, device=self.device)

        self.body_part_indices = list(range(19))
        self.body_part_names = [
            "pelvis",
            "left_thigh",
            "right_thigh",
            "abdomen",
            "left_calf",
            "right_calf",
            "left_foot",
            "right_foot",
            "chest",
            "neck",
            "left_shoulder",
            "right_shoulder",
            "head",
            "left_upper_arm",
            "right_upper_arm",
            "left_lower_arm",
            "right_lower_arm",
            "left_hand",
            "right_hand",
        ]

        "aa2euler 캐싱"
        self.full_pose_torch_world = torch.zeros((1, 72), dtype=torch.float32, device=self.device)
        self.cam_R = camera_to_world_rotmat(roll=90, pitch=0, yaw=0, device=self.device).to(dtype=torch.float32).unsqueeze(0)
        self.t_cam_to_world = torch.tensor([-0.4, -3.4, 1.55], dtype=torch.float32, device=self.device)
        self.euler_keep_idx = torch.tensor([i for i in range(22) if i not in (9, 10)], device=self.device)
        self.idx_smpl_ordered = [0, 1, 2, 3, 4, 5, 7, 8, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

    

        self.mass_19 = None
        self.inertia_body = None
        self.first_cal_inertia_mass =True

        self.has_new_pose =False

        self.u_dummy = torch.zeros((1, 19, 3), device= self.device)

        self.physics = Physics(device= self.device)
        self.smpl = SMPL(device=self.device)
        self.smplh = SMPLH(gender="male", device=self.device)
        
    
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)
        self.keep_idx = torch.tensor(
            [i for i in range(24) if i not in (9,10)],
            device=self.device
        )

        self.merge_joint_map = {
            0: [0],        # pelvis
            1: [1],        # L_hip
            2: [2],        # R_hip
            3: [3],        # spine1
            4: [4],        # L_knee
            5: [5],        # R_knee
            6: [7, 10],    # L_ankle + L_foot
            7: [8, 11],    # R_ankle + R_foot
            8: [6,9],      # spine 2 + spine3
            9: [12],       # neck
            10: [13],      # L_collar
            11: [14],      # R_collar
            12: [15],      # head
            13: [16],      # L_shoulder
            14: [17],      # R_shoulder
            15: [18],      # L_elbow
            16: [19],      # R_elbow
            17: [20, 22],  # L_wrist + L_hand
            18: [21, 23],  # R_wrist + R_hand
        }
        self.rom_cfg = load_rom_config_19(json_path=os.path.join(os.path.expanduser(os.environ.get("HRC_WS_DIR", "~/hrc_ws")), "src/body_models/joint_rom/joint_ROM.json"))

    def q24_to_q19(self,q):
        q_19_list = [q[:,:3]]
        for new_idx, olds in self.merge_joint_map.items():
            j_rep = olds[0]  # 대표로 쓸 joint index 8: [6,9] 이면 6이 대표 조인트

            base = 3 + 3 * j_rep          # 이 조인트의 첫 축 인덱스
            cols = slice(base, base + 3)  # (base, base+1, base+2)

            # 대표 조인트의 회전 3축만 그대로 사용 (평균 X)
            q_19_list.append(q[:, cols])           # (T, 3)
        q_19 = torch.cat(q_19_list, dim=1) # smpl type 
        q_19 = q_19[:, constants.smpl2q_19] # q type
        pose_19_euler_deg = q_19[:,3:]
        pose_19_euler_deg = pose_19_euler_deg.reshape(-1,19,3)
        
        return pose_19_euler_deg 
        


    def timer_callback(self):

        if not self.has_new_pose:
            return 
        
        if self.betas is None or self.pose_torch_camera is None:
            return
    
        if self.inertia_body == None :
            self.mass_inertia() # 기본 자세에서 mass와 inertia 계산
        
        if self.pose_torch_camera != None: 
            t_start = self._t()

            t0 = self._t()
            self.pose_world, self.trans_torch_world = self.aa2euler() # camera 좌표계 axis anglg 를 world 좌표계 euler 로 변환
            t1 = self._t()
            # self.full_pose_torch_world = torch.zeros([1, 72]).type(torch.float32).to(self.device)
            self.full_pose_torch_world.zero_()
            self.full_pose_torch_world[:, constants.G2Hpose_idx] = self.pose_world # wolrd 좌표계로 변환된 값

            # self.get_logger().info(f" trans : {self.trans_torch.shape} \n pose : {self.full_pose_torch_world.shape}")
            rotmat_world, rotmat_individual_world = batch_roteulerSMPL(self.full_pose_torch_world)
            smpl_output = self.smplh.forward(betas=self.betas, rotmat=rotmat_world)
            smpl_vertices = smpl_output.vertices
            smpl_joints = smpl_output.joints_smpl
            # smpl_vertices_world = smpl_vertices + self.trans_torch_world.unsqueeze(1) # 1, 6890, 3
            smpl_joints_world = smpl_joints + self.trans_torch_world.unsqueeze(1) # 1, 24, 3
            
            '좌표계 정합 확인'
            # print(f" trnas_world : {self.trans_torch_world[0]}") 


            t2 = self._t()
            self.q = torch.cat((self.trans_torch_world, self.full_pose_torch_world), dim=1) # 1, 75
            pose_19_euler_deg = self.q24_to_q19(self.q)

            # # q_dot 필요 한가 ?
            # if self.previous_q != None:
            #     self.q_dot = self.calculate_vel()
            
            centroid_24 = self.physics.compute_centorid_24(smpl_vertices, smpl_joints)
            centroid_19 = self.physics.centroid_24_to_19(centroid_24)
            t3 = self._t()

            com_world = centroid_19 + self.trans_torch_world[:, None, :] # 이 값은 따로 world 좌표계로 변환해줘야 함
            'world 좌표계 변환'
            # print(f" com_world shape : {com_world.shape}") # 1, 19, 3

            analytic_Jr_19, world_rot = self.physics.compute_Jr_19_smpl(rotmat_individual_world) # frame, 24, 3, 75 # qtype 
            world_inerita = torch.einsum('nbij,nbjk,nbkm->nbim', world_rot, self.inertia_body, world_rot.transpose(2, 3))
            idx_smpl_ordered = [0, 1, 2, 3, 4, 5, 7, 8, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            joints_19_world = smpl_joints_world[:, idx_smpl_ordered, :] 
            analytic_Jp_19 = self.physics.compute_J_p_19(com_world, joints_19_world, analytic_Jr_19) # qtype
            t4 = self._t()

            Mq_angular_19 = torch.einsum('nbij,nbjk,nbkm->nbim', analytic_Jr_19.transpose(2, 3), world_inerita, analytic_Jr_19) # 그냥 inertia 를 사용ss
            Mq_linear_19 = torch.einsum('nbij,nbjk->nbik', analytic_Jp_19.transpose(2, 3), analytic_Jp_19) * self.mass_19.clone()[:, :, None, None]
            Mq_19 = Mq_linear_19.sum(1) + Mq_angular_19.sum(1)
            t5 = self._t()
            # self.get_logger().info(f" Mq shape : {Mq_19.shape}") # 1, 60, 60
            self.u_dummy[..., 0] = -1.0
            # self.u_dummy[..., 1] = 1.0
            norm = torch.norm(self.u_dummy, dim=-1, keepdim=True) + 1e-8  # 0 나눗셈 방지
            u_unit = self.u_dummy / norm
            
            "전체 유효질량 구하기 (NoN -rom constrained)"
            Meff_all, Lambda_constrained = self.physics.compute_all_joint_meff_cholesky_constrained_debug_lam(
                Mq_19,
                analytic_Jp_19,
                analytic_Jr_19,
                u_unit,
                contact_J_index= [6,7],
                constarind_J_index= [10,11, 4, 5])
            # self.get_logger().info(f" Meff _all : {Meff_all[0]}")
            t6 = self._t()
            # self.get_logger().info(f" mass : {torch.sum(self.mass_19[0,:])}")

            "특정 link 유효질량 구하기 (rom constrained)"
            start_time_s = time.time()
            Meff_target,_ = self.physics.compute_all_joint_meff_cholesky_constrained_debug_lam_rom_per_target_debug(
                Mq_19,
                analytic_Jp_19,
                analytic_Jr_19,
                u_unit,
                pose_19_euler_deg,
                self.rom_cfg,
                j_eval=1,
                contact_J_index=[6,7],
                always_lock_joints=(10,11),
                tol_deg=5.0,
                return_debug=False,
                print_timing=False
            )
            # print(f" 유효질량(rom 최적화 디버그)  : {Meff_target}") 
            # print(f" cal time : {time.time() - start_time_s}") # 0.003


            if self.previous_com_world is not None: # 속도 계산
                self.link_velocity = ( com_world - self.previous_com_world ) / constants.dt
            else:
                self.link_velocity.zero_()
    

            "msg 1"
            msg = HumanState()
            msg.effmass = Meff_all.detach().cpu().reshape(-1).tolist()
            self.pub.publish(msg)

            "msg 2"
            dyn_msg = HumanDynamicsState()
            dyn_msg.header.stamp = self.get_clock().now().to_msg()
            dyn_msg.header.frame_id = "world"

            dyn_msg.body_part_indices = self.body_part_indices
            dyn_msg.body_part_names = self.body_part_names
            dyn_msg.body_part_positions = com_world.detach().cpu().reshape(-1).tolist()
            dyn_msg.body_part_velocities = self.link_velocity.detach().cpu().reshape(-1).tolist()

            self.dynamics_pub.publish(dyn_msg)

      

            self.get_logger().info(
            f"aa2euler {(t1-t0)*1000:.1f} | roteuler {(t2-t1)*1000:.1f} | "
            f"mass {(t3-t2)*1000:.1f} | J {(t4-t3)*1000:.1f} | meff {(t5-t4)*1000:.1f} | constrained {(t6 - t5)* 1000:.1f} | \n"
            f"total {(t5-t_start)*1000:.1f} ms \n" 
            )

            self.previous_q = self.q
            self.previous_com_world = com_world
            self.debug_meff(Meff_all)
            # self.get_logger().info(f" head eff : {Meff_all[0,12]:.1f}")

    def debug_meff(self,meff_19):
        for j in [6, 7]:
            v = meff_19[0,j]
            if v < 0:
                self.get_logger().info(f"Joint {j} Meff = {v:.1f}")

    def calculate_vel(self):
        q_trans_dot = (self.q[:,:3] - self.previous_q[:,:3]) / constants.dt
        q_pose_dot = wrap_to_pi_torch(self.q[:,3:] - self.previous_q[:, 3:]) / constants.dt
        q_dot = torch.cat((q_trans_dot,q_pose_dot),dim=1) # 1 , 60
           
        return q_dot


    def _t(self): 
        torch.cuda.synchronize()
        return time.perf_counter()
    
    def mass_inertia(self):
        pose_zeros = torch.zeros((1,72),dtype=torch.float32,device=self.device)
        rotmat_zeros, rotmat_individual = batch_roteulerSMPL(pose_zeros)
        output_smpl_body_frame = self.smplh.forward(betas=self.betas, rotmat= rotmat_zeros)
        vertices_body = output_smpl_body_frame.vertices
        joints_smpl_body = output_smpl_body_frame.joints_smpl
        mass_24_body, volume_24_body, inertia_24_body, centroid_24_body = self.physics.compute_massinertia(vertices_body, joints_smpl_body)
        mass_19, _, inertia_19, centorid_19 = self.physics.merge_parts_24_to_19_vertex_avg(mass_24_body, volume_24_body,
                                                                                            inertia_24_body, centroid_24_body)
        self.inertia_body = inertia_19
        self.mass_19 = mass_19


    def pose_callback(self, msg: HumanPose):
        "smplh 에 맞게 수정 "

        self.humanpose_msg = msg
        pose_np_camera = np.array(self.humanpose_msg.body_full_pose, dtype=np.float32)
        trans_camera = np.array(self.humanpose_msg.global_trans, dtype=np.float32)
        betas = np.array(self.humanpose_msg.smpl_betas, dtype=np.float32)
        pose_torch_camera = torch.from_numpy(pose_np_camera).type(torch.float32).to(self.device)
        trans_torch_cameara = torch.from_numpy(trans_camera).type(torch.float32).to(self.device)
        betas_torch = torch.from_numpy(betas).type(torch.float32).to(self.device)
        # 배치 단위 확장
        self.pose_torch_camera = pose_torch_camera.unsqueeze(0) 
        self.trans_torch_camera = trans_torch_cameara.unsqueeze(0)
        self.betas = betas_torch.unsqueeze(0)


        
        self.has_new_pose = True

        """
        최종 shpae은 배치 차원 추가
        pose : 1, 66
        trans : 1, 3 
        """

    def aa2euler(self):
        "비실시간에서 AMASS_from_k4a.py 역할을 해야 함"
        "sensing_node.py 에서 카메라 좌표계로 저장된 pose 값을 world 기준으로 변환 "
        # 4ms
        # torch.cuda.synchronize()

        start_time = time.perf_counter()
        rotmat = batch_rodrigues(self.pose_torch_camera.view(-1,1,3)).view(-1,22,3,3)
        
        '좌표계 변환 '
        # 카메라 좌표계 -> world 좌표계
        # cam_R = camera_to_world_rotmat(roll=90,pitch= 0,yaw= 0,device=self.device).unsqueeze(0)
        # t_cam_to_world = torch.tensor([-0.4, -3.4, 1.55], dtype=torch.float32, device=self.device) # world 기준 카메라의 위치
        cam_R = self.cam_R
        t_cam_to_world = self.t_cam_to_world

        # root rotation만 world 기준으로 변환
        rotmat_world = rotmat.clone()
        rotmat_world[:, 0] = cam_R.transpose(1, 2) @ rotmat_world[:, 0]
        trans_torch_world = torch.bmm(cam_R.transpose(1,2), self.trans_torch_camera.unsqueeze(-1)).squeeze(-1) + t_cam_to_world
        
      
        'axis angle - > euler'
        # torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start_time) * 1000 # 0.5 ms
        # print(f"aa -> rotmat time  per frame : {elapsed:.1f}ms")

        euler_root = rotmat2euleryzx(rotmat_world[:,:1,:,:].view(-1,3,3)).view(-1,1,3) # [N,1,3,3] -> [N*1,3,3] -> [N*3,1,3]
        euler_s = rotmat2eulerzyx(rotmat_world[:,1:16,:,:].view(-1,3,3)).view(-1,15,3) # 1~16 번 관절
        euler_shoulder = rotmat2eulerzxy(rotmat_world[:,16:18,:,:].view(-1,3,3)).view(-1,2,3) # 16~18번 관절
        euler_elbow = rotmat2euleryzx(rotmat_world[:,18:20,:,:].view(-1,3,3)).view(-1,2,3) # 18~20 번 관절
        euler_e = rotmat2eulerzyx(rotmat_world[:,20:22,:,:].view(-1,3,3)).view(-1,2,3) # 20~22번 관절

        euler = torch.cat((euler_root,euler_s,euler_shoulder,euler_elbow,euler_e), dim=1) # 관절차원으로 결합, (N, 24, 3)
        # euler = euler[:, torch.tensor([i for i in range(euler.shape[1]) if i not in (9, 10)], device=euler.device), :].reshape(euler.shape[0], -1)
        euler = euler[:, self.euler_keep_idx, :].reshape(euler.shape[0], -1)

        euler = torch.remainder(euler, 2* torch.pi)
        euler[euler >torch.pi] -= 2* torch.pi
        # print(f" euler shape : {euler.shape}") # 1, 60

        # torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start_time) * 1000
        # print(f"rotmat -> euler time  per frame : {elapsed:.1f}ms")
        return euler, trans_torch_world
    

    def destroy_node(self):
        return super().destroy_node()




def main(args =None):
    rclpy.init(args=args)
    node = DynamicsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ =='__main__':
    main