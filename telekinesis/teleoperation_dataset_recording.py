#!/usr/bin/env python3
from datetime import datetime
import pybullet as p
import numpy as np
import os
import time

from oculus_reader.scripts import *
from oculus_reader.scripts.reader import OculusReader

import kinova_msgs.msg

from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
import sys
from pathlib import Path
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
from scipy.spatial.transform import Rotation as R

# import keyboard
from pynput.keyboard import Key, Listener
import rospy
import threading
import math
import actionlib

from collections import defaultdict

# from teleop.visualizer import RawScene
import h5py
import cv2
import termios
import tty

from kinova_msgs.srv import HomeArm
"""
This takes the glove data, and runs inverse kinematics and then publishes onto LEAP Hand.

Note how the fingertip positions are matching, but the joint angles between the two hands are not.  :) 

Inspired by Dexcap https://dex-cap.github.io/ by Wang et. al. and Robotic Telekinesis by Shaw et. al.
"""
# visualizer = RawScene()

space_pressed = False
record_pressed = False

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)

camera_base_rot = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

leap2human_rot = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])

correction_rot = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

reflection_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

Rx_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])


# Convert 3*3 matrix to quaternion
def mat2quat(mat):
    q = np.zeros([4])
    q[3] = np.sqrt(1 + mat[0][0] + mat[1][1] + mat[2][2]) / 2
    q[0] = (mat[2][1] - mat[1][2]) / (4 * q[3])
    q[1] = (mat[0][2] - mat[2][0]) / (4 * q[3])
    q[2] = (mat[1][0] - mat[0][1]) / (4 * q[3])
    return q


# VR ==> MJ mapping when teleOp user is standing infront of the robot
def vrfront2mj(pose):
    pos = np.zeros([3])
    pos[0] = -1.0 * pose[2][3]
    pos[1] = -1.0 * pose[0][3]
    pos[2] = +1.0 * pose[1][3]

    mat = np.zeros([3, 3])
    mat[0][:] = -1.0 * pose[2][:3]
    mat[1][:] = +1.0 * pose[0][:3]
    mat[2][:] = -1.0 * pose[1][:3]

    return pos, mat2quat(mat)

Rotation_left = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, -1, 0]
])

Rotation_right = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, -1, 0]
])


# VR ==> MJ mapping when teleOp user is behind the robot
def vrbehind2mj(pose):
    pos = camera_base_rot @ pose[:3, 3]
    mat = Rotation_left @ pose[:3, :3] @ Rotation_right
    q = mat2quat(mat)

    return pos, q


def negQuat(quat):
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])


def mulQuat(qa, qb):
    res = np.zeros(4)
    res[0] = qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3]
    res[1] = qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2]
    res[2] = qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1]
    res[3] = qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0]
    return res


def diffQuat(quat1, quat2):
    neg = negQuat(quat1)
    diff = mulQuat(quat2, neg)
    return diff


class KinovaNode:
    def __init__(self):
        rospy.init_node("teleoperation_node", anonymous=True)

        # 运动链初始化
        urdf_path = "../kinova-ros/kinova_description/urdf/robot.urdf"
        
        self.execute_result = False

        # 关节配置
        self.controlled_joints = [
            "j2n6s300_joint_1",
            "j2n6s300_joint_2",
            "j2n6s300_joint_3",
            "j2n6s300_joint_4",
            "j2n6s300_joint_5",
            "j2n6s300_joint_6",
        ]

        # ROS通信
        self.joint_state_sub = rospy.Subscriber(
            "/j2n6s300_driver/out/joint_state",
            JointState,
            self.joint_state_callback,
            queue_size=1,
        )
        self.joint_state_pub = rospy.Publisher(
            "/teleoperation_joint_state", JointState, queue_size=1
        )
        self.joint_state = JointState()
        self.joint_state.name = self.controlled_joints

        action_address = '/j2n6s300_driver/joints_action/joint_angles'
        self.client = actionlib.SimpleActionClient(action_address, kinova_msgs.msg.ArmJointAnglesAction)
        self.client.wait_for_server()
        self.current_qpos = defaultdict(lambda: [])
        self.current_qpos_from_callback = None

        self.rospy_thread = threading.Thread(target=rospy.spin)
        self.rospy_thread.start()

        # wait 5 sec
        rospy.sleep(3)
        print("init success")
        print(self.current_qpos_from_callback)
        self.current_qpos['j2n6s300_joint_1'] = self.current_qpos_from_callback['j2n6s300_joint_1']
        self.current_qpos['j2n6s300_joint_2'] = self.current_qpos_from_callback['j2n6s300_joint_2']
        self.current_qpos['j2n6s300_joint_3'] = self.current_qpos_from_callback['j2n6s300_joint_3']
        self.current_qpos['j2n6s300_joint_4'] = self.current_qpos_from_callback['j2n6s300_joint_4']
        self.current_qpos['j2n6s300_joint_5'] = self.current_qpos_from_callback['j2n6s300_joint_5']
        self.current_qpos['j2n6s300_joint_6'] = self.current_qpos_from_callback['j2n6s300_joint_6']

    def joint_state_callback(self, msg):
        """关节状态回调函数"""
        q_dict = {name: pos for name, pos in zip(msg.name, msg.position)}
        self.current_qpos_from_callback = q_dict
        return

    def publish_joint_command(self, q_target):
        """发布关节指令"""
        pi = math.pi
        self.joint_state.position = [
            q_target[0] * 360 / (2 * pi),
            q_target[1] * 360 / (2 * pi),
            q_target[2] * 360 / (2 * pi),
            q_target[3] * 360 / (2 * pi),
            q_target[4] * 360 / (2 * pi),
            q_target[5] * 360 / (2 * pi),
        ]

        rospy.loginfo(f"发布关节指令: {q_target}")
        self.joint_state_pub.publish(self.joint_state)

    def publish_action_command(self, q_target):
        """发布关节指令"""
        if q_target is None:
            return 
        pi = math.pi
        rospy.loginfo(f"receive command")
        goal = kinova_msgs.msg.ArmJointAnglesGoal()
        goal.angles.joint1 = q_target[0] * 360 / (2*pi)
        goal.angles.joint2 = q_target[1] * 360 / (2*pi)
        goal.angles.joint3 = q_target[2] * 360 / (2*pi)
        goal.angles.joint4 = q_target[3] * 360 / (2*pi)
        goal.angles.joint5 = q_target[4] * 360 / (2*pi)
        goal.angles.joint6 = q_target[5] * 360 / (2*pi)
        
        # self.client.cancel_all_goals()
        time0 = time.time()
        self.client.send_goal(goal)
        if self.client.wait_for_result(rospy.Duration(8)):  # 从15秒减少到8秒
            self.execute_result = True
            result = self.client.get_result()
            # print("    Action完成，结果: ", result)
            self.current_qpos['j2n6s300_joint_1'] = result.angles.joint1 * (2*pi) / 360
            self.current_qpos['j2n6s300_joint_2'] = result.angles.joint2 * (2*pi) / 360
            self.current_qpos['j2n6s300_joint_3'] = result.angles.joint3 * (2*pi) / 360
            self.current_qpos['j2n6s300_joint_4'] = result.angles.joint4 * (2*pi) / 360
            self.current_qpos['j2n6s300_joint_5'] = result.angles.joint5 * (2*pi) / 360
            self.current_qpos['j2n6s300_joint_6'] = result.angles.joint6 * (2*pi) / 360
            time1 = time.time()
            # print("    执行时间: ", time1 - time0)
        else:
            print("    动作超时，取消目标")
            self.client.cancel_all_goals()
    
        return 

class LeapNode:
    def __init__(self):
        # Some parameters
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(
            np.zeros(16))

        # You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors = [0, 1, 2, 3, 4, 5,
                                6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        try:
            self.dxl_client = DynamixelClient(motors, "/dev/ttyUSB0", 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(
                    motors, "/dev/ttyUSB1", 4000000)
                self.dxl_client.connect()
            except Exception:
                try:
                    self.dxl_client = DynamixelClient(
                        motors, "/dev/ttyUSB3", 4000000)
                    self.dxl_client.connect()
                except Exception:
                    self.dxl_client = DynamixelClient(motors, "COM13", 4000000)
                    self.dxl_client.connect()
        # Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.kP, 84, 2
        )  # Pgain stiffness
        self.dxl_client.sync_write(
            [0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2
        )  # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.kI, 82, 2
        )  # Igain
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.kD, 80, 2
        )  # Dgain damping
        self.dxl_client.sync_write(
            [0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2
        )  # Dgain damping for side to side should be a bit less
        # Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(
            len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # allegro compatibility
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # Sim compatibility, first read the sim value in range [-1,1] and then convert to leap
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # read position
    def read_pos(self):
        return self.dxl_client.read_pos()  # 16dof

    # read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()

    # read current
    def read_cur(self):
        return self.dxl_client.read_cur()


class SystemPybulletIK:
    def __init__(self):
        # start pybullet
        p.connect(p.GUI)
        # load right leap hand
        path_src = os.path.abspath(__file__)
        path_src = os.path.dirname(path_src)
        self.glove_to_leap_mapping_scale = 1.6
        self.leapEndEffectorIndex = [4, 9, 14, 19]
        self.kinovaEndEffectorIndex = 9
        self.hand_q = None
        self.kinova_node = KinovaNode()
        kinova_path_src = "../kinova-ros/kinova_description/urdf/robot.urdf"
        self.kinovaId = p.loadURDF(
            kinova_path_src,
            [0.5, 0.0, 0.0],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )

        leap_path_src = os.path.join(
            path_src, "leap_hand_mesh_right/robot_pybullet.urdf"
        )
        self.leapId = p.loadURDF(
            leap_path_src,
            [0.0, 0.038, 0.098],
            p.getQuaternionFromEuler([0, -1.57, 0]),
            useFixedBase=True,
        )

        self.numJoints = p.getNumJoints(self.kinovaId)
        self.leapnumJoints = p.getNumJoints(self.leapId)
        for i in range(2, 8):
            p.resetJointState(self.kinovaId, i, np.pi)

        # for i in range(0, self.numJoints):
        #     print(p.getJointInfo(self.kinovaId, i))

        # for i in range(0, self.leapnumJoints):
        #     print(p.getJointInfo(self.leapId, i))

        p.setGravity(0, 0, 0)
        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)

        print("=" * 20)
        self.operator2mano = OPERATOR2MANO_RIGHT
        self.leap_node = LeapNode()
        self.oculus_reader = OculusReader()
        self.joint_names = [
            "Index1",
            "Index2",
            "Index3",
            "IndexTip",
            "Middle1",
            "Middle2",
            "Middle3",
            "MiddleTip",
            "Ring1",
            "Ring2",
            "Ring3",
            "RingTip",
            "Thumb1",
            "Thumb2",
            "Thumb3",
            "ThumbTip",
        ]

        print("OculusReader success.")
        print("+" * 20)

        self.episode_idx = 27
        # self.dataset_name = "leap_action.hdf5"
        current_time = datetime.now().strftime("%m%d_%H_%M")
        self.dataset_path = f"/media/yaxun/B197/teleop_data/leap_action_{current_time}"
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        # Camera thread variables
        self.camera_running = True
        self.current_color_image = None
        self.current_depth_image = None
        self.camera_lock = threading.Lock()
        
        # Initialize camera first
        self.init_camera()
        
        # Start the camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
        print("Camera thread started")

    def init_camera(self):
        """Initialize the Azure Kinect camera"""
        import pyk4a
        from pyk4a import Config, PyK4A
        
        # launch rgbd camera
        self.k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_1536P,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                camera_fps=pyk4a.FPS.FPS_30,
                synchronized_images_only=True,
            )
        )
        self.k4a.start()

        shorter_side = 720
        calibration = self.k4a.calibration
        self.K = calibration.get_camera_matrix(1)  # stand for color type
        
        # Get initial capture to determine downscale factor
        capture = self.k4a.get_capture()
        H, W = capture.color.shape[:2]
        self.downscale = shorter_side / min(H, W)
        self.K[:2] *= self.downscale
        
        print("Camera initialized")

    def camera_loop(self):
        """Camera thread loop - continuously captures images"""
        zfar = 2.0
        
        while self.camera_running:
            try:
                capture = self.k4a.get_capture()
                
                if capture.color is not None and capture.transformed_depth is not None:
                    H, W = capture.color.shape[:2]
                    H = int(H * self.downscale)
                    W = int(W * self.downscale)
        
                    color = capture.color[..., :3].astype(np.uint8)
                    color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
                    
                    depth = capture.transformed_depth.astype(np.float32) / 1e3
                    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
                    depth[(depth < 0.01) | (depth >= zfar)] = 0

                    # Thread-safe update of current images
                    with self.camera_lock:
                        self.current_color_image = color.copy()
                        self.current_depth_image = depth.copy()
                # cv2.imshow("Color Image", self.current_color_image)
                # cv2.imshow("Depth Image", self.current_depth_image)
                # cv2.waitKey(1)

            except Exception as e:
                print(f"Camera capture error: {e}")
                time.sleep(0.1)
                
            # time.sleep(0.033)  # ~30 FPS capture rate
            
    def get_current_images(self):
        """Get the current images from the camera thread"""
        with self.camera_lock:
            if self.current_color_image is not None and self.current_depth_image is not None:
                return self.current_color_image.copy(), self.current_depth_image.copy()
            else:
                return None, None

    def compute_IK(self, hand_pos, rot, target_pos, target_quat, arm_moving):
        p.stepSimulation()

        index_mcp_pos = hand_pos[0]
        index_pip_pos = hand_pos[1]
        index_dip_pos = hand_pos[2]
        index_tip_pos = hand_pos[3]
        middle_mcp_pos = hand_pos[4]
        middle_pip_pos = hand_pos[5]
        middle_dip_pos = hand_pos[6]
        middle_tip_pos = hand_pos[7]
        ring_mcp_pos = hand_pos[8]
        ring_pip_pos = hand_pos[9]
        ring_dip_pos = hand_pos[10]
        ring_tip_pos = hand_pos[11]
        thumb_mcp_pos = hand_pos[12]
        thumb_pip_pos = hand_pos[13]
        thumb_dip_pos = hand_pos[14]
        thumb_tip_pos = hand_pos[15]

        index_mcp_rot = rot[0]
        index_pip_rot = rot[1]
        index_dip_rot = rot[2]
        index_tip_rot = rot[3]
        middle_mcp_rot = rot[4]
        middle_pip_rot = rot[5]
        middle_dip_rot = rot[6]
        middle_tip_rot = rot[7]
        ring_mcp_rot = rot[8]
        ring_pip_rot = rot[9]
        ring_dip_rot = rot[10]
        ring_tip_rot = rot[11]
        thumb_mcp_rot = rot[12]
        thumb_pip_rot = rot[13]
        thumb_dip_rot = rot[14]
        thumb_tip_rot = rot[15]

        leapEndEffectorPos = [
            index_tip_pos,
            middle_tip_pos,
            ring_tip_pos,
            thumb_tip_pos,
        ]

        leapEndEffectorRot = [
            index_tip_rot,
            middle_tip_rot,
            ring_tip_rot,
            thumb_tip_rot,
        ]

        # Compute the joint angles for the leap hand and send qpos
        leap_jointPoses = p.calculateInverseKinematics2(
            self.leapId,
            self.leapEndEffectorIndex,
            leapEndEffectorPos,
            leapEndEffectorRot,
            solver=p.IK_DLS,
            maxNumIterations=500,
            residualThreshold=0.0001,
        )

        leap_whole_jointPoses = (
            leap_jointPoses[0:4]
            + (0.0,)
            + leap_jointPoses[4:8]
            + (0.0,)
            + leap_jointPoses[8:12]
            + (0.0,)
            + leap_jointPoses[12:16]
            + (0.0,)
        )
        leap_whole_jointPoses = list(leap_whole_jointPoses)
        
        for i in range(self.leapnumJoints):
            p.setJointMotorControl2(
                bodyIndex=self.leapId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=leap_whole_jointPoses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.3,
                velocityGain=1,
            )

        real_robot_hand_q = np.array([float(0.0) for _ in range(16)])
        real_robot_hand_q[0:4] = leap_jointPoses[0:4]
        real_robot_hand_q[4:8] = leap_jointPoses[4:8]
        real_robot_hand_q[8:12] = leap_jointPoses[8:12]
        real_robot_hand_q[12:16] = leap_jointPoses[12:16]
        real_robot_hand_q[0:2] = real_robot_hand_q[0:2][::-1]
        real_robot_hand_q[4:6] = real_robot_hand_q[4:6][::-1]
        real_robot_hand_q[8:10] = real_robot_hand_q[8:10][::-1]
        self.leap_node.set_allegro(real_robot_hand_q)
        self.hand_q = real_robot_hand_q
        if arm_moving:
            if target_pos[2] < 0.15:
                print("Target position is too low, skipping moving.")
                return
        # Compute the joint angles for the kinova arm and send qpos
            kinova_jointPoses = p.calculateInverseKinematics(
                self.kinovaId,
                self.kinovaEndEffectorIndex,
                target_pos,
                target_quat,
                solver=p.IK_DLS,
                maxNumIterations=500,
                residualThreshold=0.0001,
            )

            arm_qpos_now = []
            for i in range(2, 8):
                arm_qpos_now.append(p.getJointState(self.kinovaId, i)[0])
            arm_qpos_now = tuple(arm_qpos_now)

            if any(math.isnan(pose) for pose in kinova_jointPoses):
                print("IK solution is None")
                kinova_jointPoses = arm_qpos_now
            else:
                kinova_jointPoses = kinova_jointPoses[:6]
                # qpos_arm_err = np.linalg.norm(jointPoses - qpos_now)
                sum = 0.0
                for i in range(6):
                    sum += (kinova_jointPoses[i] - arm_qpos_now[i]) * (
                        kinova_jointPoses[i] - arm_qpos_now[i]
                    )
                qpos_arm_err = math.sqrt(sum)

                # if qpos_arm_err > 0.5:
                #     print("Jump detechted. Joint error {}. This is likely caused when hardware detects something unsafe. Resetting goal to where the arm curently is to avoid sudden jumps.".format(qpos_arm_err))
                #     kinova_jointPoses = arm_qpos_now

            combined_jointPoses = (
                (0.0,)
                + (0.0,)
                + kinova_jointPoses[0:6]
                + (0.0,)
                + (0.0,)
                + leap_jointPoses[0:4]
                + (0.0,)
                + leap_jointPoses[4:8]
                + (0.0,)
                + leap_jointPoses[8:12]
                + (0.0,)
                + leap_jointPoses[12:16]
                + (0.0,)
            )
            combined_jointPoses = list(combined_jointPoses)

            for i in range(0, 6):
                p.setJointMotorControl2(
                    bodyIndex=self.kinovaId,
                    jointIndex=i + 2,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=kinova_jointPoses[i],
                    targetVelocity=0,
                    force=500,
                    positionGain=0.3,
                    velocityGain=1,
                )

            for i in range(10, 30):
                p.setJointMotorControl2(
                    bodyIndex=self.kinovaId,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=combined_jointPoses[i],
                    targetVelocity=0,
                    force=500,
                    positionGain=0.3,
                    velocityGain=1,
                )

            self.kinova_node.publish_action_command(kinova_jointPoses)
        
        return 

    def cleanup(self):
        """Cleanup resources when shutting down"""
        print("Cleaning up camera thread...")
        self.camera_running = False
        if hasattr(self, 'camera_thread'):
            self.camera_thread.join(timeout=1.0)
        if hasattr(self, 'k4a'):
            self.k4a.stop()
        print("Camera cleanup completed")

    def operation(self):
        VRP0 = None
        VRR0 = None
        MJP0 = None
        MJR0 = None
        episode_ended = False
        recording = True

        episode_data = defaultdict(lambda: [])

        window_name = "k4a"
        start_trigger = False
        annotation = False
        first_tracking_frame = False
        index = 0

        first_recording = True
        # first loop: for each trajectory
        while True:
            arm_moving = False
            step_idx = 0
            recording = False
            self.dataset_name = f"{self.dataset_path}/leap_action_{self.episode_idx}.hdf5"
            if Path(self.dataset_name).exists():
                with h5py.File(self.dataset_name, "r") as f:
                    self.episode_idx = len(f["data"].keys())

            # second loop: for each step in one trajectory
            while True:
                print("episode_idx:", self.episode_idx, "step_idx:", step_idx)
                key = get_key_non_blocking()
                joint_pos = self.oculus_reader.get_joint_transformations()[1]

                # poll input device --------------------------------------
                transformations, buttons = (
                    self.oculus_reader.get_transformations_and_buttons()
                )
                if key == 'r':
                    recording = True
                    print("-"*10, "Start recording")
                    key = None
                if key == 'p':
                    recording = False
                    print("-"*10, "Stop recording")
                    key = None
                if key == 'm':
                    arm_moving = True
                    print("-"*10, "Start moving")
                    key = None
                if key == 'n':
                    arm_moving = False
                    print("-"*10, "Stop moving")
                    self.kinova_node.client.cancel_all_goals()
                    key = None
                if key == 'q':
                    break
                
                self.kinova_node.execute_result = False

                # receive info from VR
                if transformations and "r" in transformations:
                    # print("mapping")
                    right_controller_pose = transformations["r"]
                    # VRpos, VRquat = vrfront2mj(right_controller_pose) # front x, left y, up z
                    VRpos, VRquat = vrbehind2mj(
                        right_controller_pose
                    )  # front -x, left -y, up z

                    if arm_moving: # add the offset to the position
                        print("+"*10, "Arm moving")
                        dVRP = VRpos - VRP0
                        # dVRR = diffQuat(VRR0, VRquat)
                        curr_pos = MJP0 + dVRP
                        # curr_quat = mulQuat(MJR0, dVRR)
                        curr_quat = VRquat

                    # Adjust origin if not engaged
                    else:
                        # set the current qpos from real arm
                        kinova_joint_state = self.kinova_node.current_qpos  # arm qpose
                        p.resetJointState(
                            self.kinovaId, 2, kinova_joint_state["j2n6s300_joint_1"]
                        )
                        p.resetJointState(
                            self.kinovaId, 3, kinova_joint_state["j2n6s300_joint_2"]
                        )
                        p.resetJointState(
                            self.kinovaId, 4, kinova_joint_state["j2n6s300_joint_3"]
                        )
                        p.resetJointState(
                            self.kinovaId, 5, kinova_joint_state["j2n6s300_joint_4"]
                        )
                        p.resetJointState(
                            self.kinovaId, 6, kinova_joint_state["j2n6s300_joint_5"]
                        )
                        p.resetJointState(
                            self.kinovaId, 7, kinova_joint_state["j2n6s300_joint_6"]
                        )

                        # current end effector pose
                        link_state = p.getLinkState(self.kinovaId, 9)
                        curr_pos = link_state[4]
                        curr_quat = link_state[5]

                        MJP0 = curr_pos  # real kinova pos origin
                        MJR0 = curr_quat  # real kinova quat origin

                        VRP0 = VRpos
                        VRR0 = VRquat

                    # udpate desired pos
                    target_pos = curr_pos
                    target_quat = VRquat

                    # leaphand
                    pos = []
                    final_pos = []
                    rot = []
                    if joint_pos == {}:
                        continue
                    else:
                        mediapipe_wrist_rot = joint_pos["WristRoot"][:3, :3]
                        wrist_position = joint_pos["WristRoot"][:3, 3]

                        for joint_name in self.joint_names:
                            joint_transformation = joint_pos[joint_name]

                            pos.append(
                                joint_transformation[:3, 3] - wrist_position)
                            if joint_name == "MiddleTip":
                                pos[-1][0] += 0.009375
                            if joint_name == "RingTip":
                                pos[-1][0] -= 0.004375
                            if joint_name == "IndexTip":
                                pos[-1][0] -= 0.00125
                            if joint_name == "ThumbTip":
                                pos[-1][0] += 0.00375
                            pos[-1] = pos[-1] @ mediapipe_wrist_rot @ self.operator2mano

                            # Turn the rotation matrix into quaternion
                            rotation = (
                                joint_transformation[:3, :3]
                                @ mediapipe_wrist_rot
                                @ self.operator2mano
                            )
                            quaternion = R.from_matrix(rotation).as_quat()
                            rot.append(quaternion)

                            final_pos.append(
                                [
                                    pos[-1][0] *
                                    self.glove_to_leap_mapping_scale * 1.15,
                                    pos[-1][1] *
                                    self.glove_to_leap_mapping_scale,
                                    pos[-1][2] *
                                    self.glove_to_leap_mapping_scale,
                                ]
                            )
                            final_pos[-1][2] -= 0.05

                    self.compute_IK(final_pos, rot, target_pos, target_quat, arm_moving)
                    # recording dataset
                    
                    if (
                        transformations
                        and "r" in transformations
                        and not episode_ended
                        and recording
                        and self.kinova_node.execute_result
                    ):
                        print("="*10, "recording", "="*10)
                        hand_state = self.leap_node.read_pos()  # gripper_state
                        
                        kinova_joint_state = self.kinova_node.current_qpos  # arm qpose
                        p.resetJointState(
                            self.kinovaId, 2, kinova_joint_state["j2n6s300_joint_1"]
                        )
                        p.resetJointState(
                            self.kinovaId, 3, kinova_joint_state["j2n6s300_joint_2"]
                        )
                        p.resetJointState(
                            self.kinovaId, 4, kinova_joint_state["j2n6s300_joint_3"]
                        )
                        p.resetJointState(
                            self.kinovaId, 5, kinova_joint_state["j2n6s300_joint_4"]
                        )
                        p.resetJointState(
                            self.kinovaId, 6, kinova_joint_state["j2n6s300_joint_5"]
                        )
                        p.resetJointState(
                            self.kinovaId, 7, kinova_joint_state["j2n6s300_joint_6"]
                        )

                        # current end effector pose
                        link_state = p.getLinkState(self.kinovaId, 9)
                        curr_pos = link_state[4]
                        curr_quat = link_state[5]
                        
                        # Apply offset to curr_pos[0] for recording
                        curr_pos_adjusted = list(curr_pos)
                        curr_pos_adjusted[0] -= 0.5
                        curr_pos = tuple(curr_pos_adjusted)
                        
                        target_pos[0] -= 0.5
                        
                        # Get current images from camera thread
                        color_image, depth_image = self.get_current_images()
                        print("depth_image shape:", depth_image.shape if depth_image is not None else "No depth image")
                        print("color_image shape:", color_image.shape if color_image is not None else "No color image")
                        episode_data["actions"] += [
                            np.concatenate(
                                [target_pos, target_quat, self.hand_q])
                        ]
                        episode_data["dones"] += [False]
                        episode_data["rewards"] += [False]
                        # Current data
                        episode_data["obs/time"] += [time.time()]
                        episode_data["obs/arm_qpos"] += [
                            list(kinova_joint_state.values())]
                        episode_data["obs/eef_pos"] += [curr_pos]
                        episode_data["obs/eef_rot"] += [curr_quat]
                        episode_data["obs/gripper_qpos"] += [hand_state]
                        
                        # Add camera images if available
                        if color_image is not None and depth_image is not None:
                            episode_data["obs/agentview_image"] += [color_image]
                            episode_data["obs/depth_image"] += [depth_image]
                        else:
                            print("Warning: No camera images available")

                        step_idx += 1

                if key == 's':
                    print("You pressed 's': Succes and Next Episode ...")
                    key = None
                    if not (episode_data.get("actions") == None):
                        episode_data['rewards'][-1] = 1.0
                        episode_data['dones'][-1] = True
                        self.episode_idx += 1
                        with h5py.File(self.dataset_name, 'a') as f:
                            for k in episode_data:
                                f[f'data/demo/{k}'] = np.stack(episode_data[k])
                        print(f"Episode {self.episode_idx} saved")

                    episode_data = defaultdict(lambda: [])
                    break

                # end if
            if key == 'q':
                break
            # end while 2

        # end while 1
        self.cleanup()
        print(f"All {self.episode_idx} Episodes saved")


def get_key_non_blocking():
    """Read a single keypress without Enter and without blocking."""
    import select
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # Set terminal to cbreak mode
        if select.select([sys.stdin], [], [], 0.1)[0]:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return None


def on_press(key):
    global space_pressed
    if key == Key.space:
        space_pressed = True


# 键盘释放回调
def on_release(key):
    global space_pressed
    if key == Key.space:
        space_pressed = False


def main(args=None):

    leappybulletik = SystemPybulletIK()

    leappybulletik.operation()

def call_home_arm():
        """调用home arm服务"""
        print("正在调用home arm服务...")
        try:
            # 等待home arm服务
            service_name = f'/j2n6s300_driver/in/home_arm'
            rospy.wait_for_service(service_name, timeout=10.0)
            
            # 调用服务
            home_arm_service = rospy.ServiceProxy(service_name, HomeArm)
            response = home_arm_service()
            
            print(f"Home arm服务响应: {response.homearm_result}")
            
            if "KINOVA ARM HAS BEEN RETURNED HOME" in response.homearm_result:
                print("✅ Home arm服务调用成功")
                # 等待机械臂完成home动作
                time.sleep(5.0)
                return True
            else:
                print(f"❌ Home arm服务调用失败: {response.homearm_result}")
                return False
                
        except rospy.ServiceException as e:
            print(f"❌ Home arm服务调用异常: {e}")
            return False
        except rospy.ROSException as e:
            print(f"❌ 等待home arm服务超时: {e}")
            return False

if __name__ == "__main__":
    # if not call_home_arm():
    #     print("❌ Home arm失败，程序退出")
    # else:
    main()
