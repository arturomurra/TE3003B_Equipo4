#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
from tf_transformations import euler_from_quaternion
import numpy as np
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import enum

# Define the state machine for the Bug2 algorithm
class StateMachine(enum.Enum):
    LOOK_TOGOAL = 1
    FOLLOW_LINE = 2
    WALL_FOLLOW = 3
    STOP = 4

class Bug2Controller(Node):
    def __init__(self):
        super().__init__('Bug2')
        self.yaw = 0.0
        self.current_state = StateMachine.LOOK_TOGOAL

        self.current_pose = PoseStamped()
        self.start_pose = PoseStamped()
        self.goal = PoseStamped()

        self.cmd_vel = Twist()
        self.hitpoint = None
        self.distance_moved = 0.0

        self.front_distance = 1.0
        self.frontL_distance = 0.0
        self.left_distance = 0.0

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        self.create_subscription(PoseStamped, '/goal', self.goal_callback, 1)
        self.create_subscription(LaserScan, '/filtered_scan', self.scan_callback, 1)
        self.create_subscription(Bool, '/bug2_run', self.run, 1)

    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def look_to_goal(self):
        quaternion = (
            self.current_pose.pose.orientation.x, self.current_pose.pose.orientation.y,
            self.current_pose.pose.orientation.z, self.current_pose.pose.orientation.w)
        euler = euler_from_quaternion(quaternion)
        self.yaw = euler[2]

        angle_to_goal = math.atan2(self.goal.pose.position.y - self.current_pose.pose.position.y,
                                   self.goal.pose.position.x - self.current_pose.pose.position.x)
        angle_error = self.wrap_to_pi(angle_to_goal - self.yaw)

        if np.fabs(angle_error) > np.pi / 90:
            self.cmd_vel.angular.z = 0.2 if angle_error > 0 else -0.2
        else:
            self.cmd_vel.angular.z = 0.0
            self.current_state = StateMachine.FOLLOW_LINE

        self.cmd_vel_pub.publish(self.cmd_vel)

    def move_to_goal(self):
        if self.front_distance < 0.3:
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel.angular.z = 0.0
            self.hitpoint = self.current_pose.pose.position
            self.current_state = StateMachine.WALL_FOLLOW
        else:
            self.cmd_vel.linear.x = 0.05
            self.cmd_vel.angular.z = 0.0

        self.cmd_vel_pub.publish(self.cmd_vel)

    def follow_wall(self):
        # Proportional controller for wall following
        desired_distance_from_wall = 0.3  # Desired distance to maintain from the wall
        Kp = 1.0  # Proportional gain for the controller

        closestGoalLine_x = self.current_pose.pose.position.x
        closestGoalLine_y = self.line_slope_m * self.current_pose.pose.position.x + self.line_slope_b

        self.distance_moved = math.sqrt((self.current_pose.pose.position.x - self.hitpoint.x) ** 2 +
                                        (self.current_pose.pose.position.y - self.hitpoint.y) ** 2)
        distance_to_line = math.sqrt((closestGoalLine_x - self.current_pose.pose.position.x) ** 2 +
                                     (closestGoalLine_y - self.current_pose.pose.position.y) ** 2)

        if distance_to_line < 0.1 and self.distance_moved > 0.5:
            distance_to_goal = math.sqrt((self.goal.pose.position.x - self.current_pose.pose.position.x) ** 2 +
                                         (self.goal.pose.position.y - self.current_pose.pose.position.y) ** 2)
            hitpoint_distance_to_goal = math.sqrt((self.goal.pose.position.x - self.hitpoint.x) ** 2 +
                                                  (self.goal.pose.position.y - self.hitpoint.y) ** 2)

            if hitpoint_distance_to_goal > distance_to_goal:
                self.cmd_vel.linear.x = 0.0
                self.cmd_vel.angular.z = 0.0
                self.current_state = StateMachine.LOOK_TOGOAL
                return

        distance_error = desired_distance_from_wall - self.left_distance
        self.cmd_vel.linear.x = 0.05
        self.cmd_vel.angular.z = Kp * distance_error

        self.cmd_vel_pub.publish(self.cmd_vel)

    def goal_callback(self, msg):
        self.goal = msg
        self.start_pose = self.current_pose

        self.line_slope_m = (self.goal.pose.position.y - self.start_pose.pose.position.y) / (
                self.goal.pose.position.x - self.start_pose.pose.position.x) if self.goal.pose.position.x != self.start_pose.pose.position.x else float('inf')
        self.line_slope_b = self.start_pose.pose.position.y - (self.line_slope_m * self.start_pose.pose.position.x)
        self.current_state = StateMachine.LOOK_TOGOAL

    def odom_callback(self, msg):
        self.current_pose.pose = msg.pose.pose

    def scan_callback(self, msg):
        data = np.array(msg.ranges)
        self.front_distance = np.min(data[180:540])  # Adjust based on your lidar's range and resolution
        self.frontL_distance = np.min(data[540:900])
        self.left_distance = np.min(data[0:180])  # Left side distance
        self.get_logger().info("Scan data updated.")

    def stop(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)

    def run(self, msg):
        if self.current_state is StateMachine.LOOK_TOGOAL:
            self.look_to_goal()
        elif self.current_state is StateMachine.FOLLOW_LINE:
            self.move_to_goal()
        elif self.current_state is StateMachine.WALL_FOLLOW:
            self.follow_wall()
        elif self.current_state is StateMachine.STOP:
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel.angular.z = 0.0
            self.get_logger().info("Found goal!")

        self.cmd_vel_pub.publish(self.cmd_vel)
        goal_distance = math.sqrt((self.goal.pose.position.x - self.current_pose.pose.position.x) ** 2 +
                                  (self.goal.pose.position.y - self.current_pose.pose.position.y) ** 2)

        if goal_distance < 0.15:
            self.current_state = StateMachine.STOP

def main(args=None):
    rclpy.init(args=args)
    bug = Bug2Controller()
    rclpy.spin(bug)
    bug.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

