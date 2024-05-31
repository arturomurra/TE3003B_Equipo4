import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped
import numpy as np

class FastSLAM:
    def __init__(self):
        self.map_resolution = 0.05  # Resolución del mapa en metros/celda
        self.map_size_x = 100  # Tamaño del mapa en celdas en dirección x
        self.map_size_y = 100  # Tamaño del mapa en celdas en dirección y
        self.map_origin_x = -self.map_size_x * self.map_resolution / 2.0
        self.map_origin_y = -self.map_size_y * self.map_resolution / 2.0
        self.map_data = np.zeros((self.map_size_x, self.map_size_y), dtype=np.int8)

    def update(self, scan_msg):
        # Actualizar el mapa utilizando los datos del escaneo láser
        # Aquí implementarías tu algoritmo de SLAM

        # Por simplicidad, en este ejemplo simplemente colocamos obstáculos en la posición del robot
        for i, range_value in enumerate(scan_msg.ranges):
            if range_value < scan_msg.range_max:
                angle = scan_msg.angle_min + i * scan_msg.angle_increment
                x = int((scan_msg.ranges[i] * np.cos(angle)) / self.map_resolution - self.map_origin_x)
                y = int((scan_msg.ranges[i] * np.sin(angle)) / self.map_resolution - self.map_origin_y)
                if 0 <= x < self.map_size_x and 0 <= y < self.map_size_y:
                    self.map_data[x, y] = 100  # Marcamos el obstáculo en el mapa

        return self.generate_map_message()

    def generate_map_message(self):
        map_msg = OccupancyGrid()
        map_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        map_msg.header.frame_id = "map"
        map_msg.info.map_load_time = map_msg.header.stamp
        map_msg.info.resolution = self.map_resolution
        map_msg.info.width = self.map_size_x
        map_msg.info.height = self.map_size_y
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.x = 0.0
        map_msg.info.origin.orientation.y = 0.0
        map_msg.info.origin.orientation.z = 0.0
        map_msg.info.origin.orientation.w = 1.0
        map_msg.data = np.ravel(np.flipud(self.map_data)).tolist()
        return map_msg

class SLAMMappingNode(Node):
    def __init__(self):
        super().__init__('slam_mapping_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.map_publisher = self.create_publisher(
            OccupancyGrid,
            '/map',
            10)
        self.fast_slam = FastSLAM()

    def scan_callback(self, msg):
        slam_map = self.fast_slam.update(msg)
        self.map_publisher.publish(slam_map)

def main(args=None):
    rclpy.init(args=args)
    slam_mapping_node = SLAMMappingNode()
    rclpy.spin(slam_mapping_node)
    slam_mapping_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
