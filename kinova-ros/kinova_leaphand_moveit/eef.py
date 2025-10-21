#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose

def move_to_pose():
    # 初始化
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('moveit_pose_control', anonymous=True)
    
    # 创建MoveGroupCommander对象
    robot = moveit_commander.RobotCommander()
    group = moveit_commander.MoveGroupCommander("kinova_arm")  # 修改为你的规划组名称
    
    # 设置目标位姿
    target_pose = Pose()
    target_pose.position.x = 0.5   # 目标位置X（米）
    target_pose.position.y = 0.2
    target_pose.position.z = 0.4
    target_pose.orientation.w = 1.0  # 四元数表示旋转
    
    # 设置参考坐标系
    group.set_pose_reference_frame("base_link")  # 根据实际修改
    
    # 运动规划
    group.set_pose_target(target_pose)
    plan = group.plan()
    
    if plan[0]:
        group.execute(plan[1], wait=True)
    else:
        rospy.logerr("Planning failed!")
    
    # 关闭连接
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        move_to_pose()
    except rospy.ROSInterruptException:
        pass