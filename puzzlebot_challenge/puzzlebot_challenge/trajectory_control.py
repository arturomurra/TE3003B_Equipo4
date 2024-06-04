import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray, Int32, Bool, String
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
from puzzlebot_msgs.msg import Arucoinfo, ArucoArray, LandmarkList, Landmark
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import tf2_ros
from tf2_geometry_msgs import do_transform_point
import enum
from math import sin, cos, inf, pi
from numpy import linspace

# Definición de los estados posibles de la máquina de estados
class StateMachine(enum.Enum):
    FIND_LANDMARK = 1
    WANDER = 2
    GO_TO_TARGET = 3
    HANDLE_OBJECT = 4
    STOP = 5

# Clase principal que controla la trayectoria del robot
class TrajectoryControl(Node):
    def __init__(self):
        super().__init__('Trajectory_Control')

        # Suscripciones a diferentes tópicos
        self.odometry_sub = self.create_subscription(Odometry, '/odom', self.odometry_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/filtered_scan', self.lidar_callback, 10)
        self.aruco_sub = self.create_subscription(ArucoArray, '/aruco_info', self.aruco_callback, 10)
        self.arrived_sub = self.create_subscription(Bool, '/arrived', self.arrived_callback, 10)
        self.handled_sub = self.create_subscription(Bool, '/handled_aruco', self.handled_callback, 10)
        self.object_sub = self.create_subscription(String, '/object_state', self.object_state_callback, 10)

        # Publicadores a diferentes tópicos
        self.velocity_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal', 1)
        self.landmarks_pub = self.create_publisher(LandmarkList, '/landmarks', 1)
        self.bug_pub = self.create_publisher(Bool, '/bug2_run', 1)
        self.handle_run_pub = self.create_publisher(Bool, '/handle_run', 1)
        self.handle_pub = self.create_publisher(Int32, '/handle', 1)
        self.aruco_goal_pub = self.create_publisher(String, '/aruco_goal', 1)
        
        # Inicialización de la pose y ángulo actual
        self.current_pose = PoseStamped()
        self.current_pose.header.frame_id = "world"
        self.current_pose.pose.position.x = 0.0
        self.current_pose.pose.position.y = 0.0
        self.current_pose.pose.orientation.x = 0.0
        self.current_pose.pose.orientation.y = 0.0
        self.current_pose.pose.orientation.z = 0.0
        self.current_pose.pose.orientation.w = 1.0
        self.current_angle = 0.0

        # Estado inicial de la máquina de estados
        self.current_state = StateMachine.FIND_LANDMARK
        
        # Área de descarga y objetivo
        self.discharge_area = PoseStamped()
        self.discharge_area.header.frame_id = "world"
        self.discharge_area.pose.position.x = 0.0
        self.discharge_area.pose.position.y = 0.0

        self.goal = PoseStamped()
        self.goal.header.frame_id = "world"
        self.goal.pose.position.x = 2.0
        self.goal.pose.position.y = 2.0

        # Otras variables
        self.cmd_vel = None
    
        self.cube_id = '2'  # ID del cubo que el robot está buscando
        self.object_state = ""  # Estado del objeto que se está manipulando

        self.aruco_info = ArucoArray()  # Información de los marcadores Aruco detectados
        self.aruco_info.length = 0
        self.aruco_info.aruco_array = []

        self.target_area_id = None
        self.distances = []  # Distancias medidas por el LIDAR
        self.distance_covered = 0.0

        self.extent = 1.0  # Define la variable extent

        # Timer para actualizar la pose
        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.run)
        self.arrived = None
        self.carga = False  # Estado de carga del objeto

        # Creación del diccionario de Arucos
        self.station = Arucoinfo()
        self.station_on_sight = False

        self.goal_ids = {
            'A': '1',
            'B': '8',
            'C': '3'
        }

        self.desired_goal = 'A'  # Objetivo deseado (id del marcador Aruco)

        # Configurar las coordenadas de cada punto
        self.convergence_point = PoseStamped()
        self.convergence_point.pose.position.x = 2.4
        self.convergence_point.pose.position.y = 2.2
        self.convergence_point.pose.position.z = 0.0
        self.convergence_point.header.frame_id = 'world'

    # Callback para la odometría, actualiza la pose y el ángulo actuales del robot
    def odometry_callback(self, msg: Odometry):
        self.current_pose.pose = msg.pose.pose
        orientation_euler = euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
        self.current_angle = orientation_euler[2]

    # Callback para la detección de Aruco, actualiza la información de los Arucos detectados
    def aruco_callback(self, msg):
        self.aruco_info = msg
        for aruco in self.aruco_info.aruco_array:
            if aruco.id == self.goal_ids[self.desired_goal] or aruco.id == self.cube_id:
                self.aruco_info.aruco_array[0] = aruco

    # Callback para el LIDAR, actualiza las distancias medidas
    def lidar_callback(self, msg):
        self.distances = np.array(msg.ranges)

    # Callback para manejar el estado de la carga del Aruco
    def handled_callback(self, msg):
        self.carga = msg.data

    # Callback para manejar el estado del objeto
    def object_state_callback(self, msg):
        self.object_state = msg.data

    # Transforma la posición del cubo desde el marco del robot al marco del mundo
    def transform_cube_position(self, aruco_point):
        rotation_matrix = np.array([
            [cos(self.current_angle), -sin(self.current_angle)],
            [sin(self.current_angle), cos(self.current_angle)]
        ])

        puzzlebot_coords = np.array([aruco_point.z, aruco_point.x])
        world_coords = rotation_matrix.dot(puzzlebot_coords)

        aruco_x = self.current_pose.pose.position.x + world_coords[0]
        aruco_y = self.current_pose.pose.position.y + world_coords[1]

        return aruco_x, aruco_y

    # Callback para manejar la llegada al objetivo
    def arrived_callback(self, msg):
        self.arrived = msg.data

    # Método principal que se ejecuta periódicamente para controlar el comportamiento del robot
    def run(self):
        self.get_logger().info(f"Pose to goal: {np.sqrt((self.goal.pose.position.x - self.current_pose.pose.position.x)**2 + (self.goal.pose.position.y - self.current_pose.pose.position.y)**2)}")
        self.get_logger().info(f"State: {self.current_state}")
        
        # Estado de búsqueda de marcadores
        if self.current_state is StateMachine.FIND_LANDMARK:
            if self.aruco_info.length > 0:
                self.get_logger().info(f"Condition: {self.aruco_info.aruco_array[0].id == self.goal_ids[self.desired_goal]}, aruco: {self.aruco_info.aruco_array[0].id}")
                if self.aruco_info.aruco_array[0].id == self.cube_id:
                    # Si se encuentra el cubo, se detiene el giro y se cambia al estado de WANDER
                    stop_spin_msg = Twist()
                    self.velocity_pub.publish(stop_spin_msg)
                    self.current_state = StateMachine.WANDER
                elif self.object_state == 'lifted' and self.arrived and self.aruco_info.aruco_array[0].id == self.goal_ids[self.desired_goal]:
                    # Si el objeto ha sido levantado y se ha llegado al destino, se cambia al estado de HANDLE_OBJECT
                    stop_spin_msg = Twist()
                    self.velocity_pub.publish(stop_spin_msg)
                    self.handle_pub.publish(Int32(data=1))
                    self.object_state = 'dropped'
                    self.current_state = StateMachine.HANDLE_OBJECT
            else:
                # Si no se detecta ningún Aruco, el robot sigue girando
                spin_msg = Twist()
                spin_msg.angular.z = -0.05  # Velocidad angular en radianes por segundo
                spin_msg.linear.x = 0.0
                self.velocity_pub.publish(spin_msg)

        # Estado de exploración
        elif self.current_state is StateMachine.WANDER:
            if self.aruco_info.length > 0:
                if self.aruco_info.aruco_array[0].id == self.cube_id:
                    # Si se detecta el cubo, se establece como objetivo y se cambia al estado de GO_TO_TARGET
                    self.goal.pose.position.x, self.goal.pose.position.y = self.transform_cube_position(self.aruco_info.aruco_array[0].point.point)
                    self.goal_pub.publish(self.goal)
                    self.current_state = StateMachine.GO_TO_TARGET

        # Estado de ir al objetivo
        elif self.current_state is StateMachine.GO_TO_TARGET:
            if self.aruco_info.length > 0 and self.carga == False:
                if self.aruco_info.aruco_array[0].id == self.cube_id and self.aruco_info.aruco_array[0].point.point.z < 0.5:
                    # Si se está cerca del cubo, se detiene el robot y se cambia al estado de HANDLE_OBJECT
                    stop_spin_msg = Twist()
                    self.velocity_pub.publish(stop_spin_msg)
                    self.handle_pub.publish(Int32(data=0))
                    self.object_state = 'lifted'
                    self.current_state = StateMachine.HANDLE_OBJECT
                else:
                    # Si no se está cerca del cubo, se sigue el camino utilizando el algoritmo Bug2
                    self.bug_pub.publish(Bool(data=True))
            elif self.object_state == 'lifted' and self.arrived:
                # Si el objeto ha sido levantado y se ha llegado al destino, se vuelve al estado de FIND_LANDMARK
                self.current_state = StateMachine.FIND_LANDMARK
            else:
                # Si no se cumplen las condiciones anteriores, se sigue el camino utilizando el algoritmo Bug2
                self.bug_pub.publish(Bool(data=True))

        # Estado de manejo del objeto
        elif self.current_state is StateMachine.HANDLE_OBJECT:
            if self.carga:
                if self.object_state == "lifted":
                    # Si el objeto está levantado, se establece el punto de convergencia como objetivo y se cambia al estado de GO_TO_TARGET
                    self.goal = self.convergence_point
                    self.goal_pub.publish(self.goal)
                    self.current_state = StateMachine.GO_TO_TARGET
                    self.carga = False
                elif self.object_state == "dropped":
                    # Si el objeto ha sido soltado, se establece el área de descarga como objetivo y se cambia al estado de STOP
                    self.goal.pose.position.x = 0.0
                    self.goal.pose.position.y = 0.0
                    self.goal_pub.publish(self.goal)
                    self.current_state = StateMachine.STOP
                else:
                    # Si no se cumplen las condiciones anteriores, se publica el estado de ejecución de manejo del objeto
                    self.handle_run_pub.publish(Bool(data=True))
            else:
                self.handle_run_pub.publish(Bool(data=True))

        # Estado de detención
        elif self.current_state is StateMachine.STOP:
            # Detiene el robot
            stop = Twist()
            stop.linear.x = 0.0
            stop.angular.z = 0.0
            self.velocity_pub.publish(stop)

# Función principal para inicializar y ejecutar el nodo
def main(args=None):
    rclpy.init(args=args)
    slam_node = TrajectoryControl()
    rclpy.spin(slam_node)
    slam_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

