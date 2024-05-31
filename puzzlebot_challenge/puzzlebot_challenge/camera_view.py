#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/video_source/raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        if len(msg.data) > 0:
            # Si los datos están presentes en el mensaje
            if msg.encoding == "compressed":
                # La imagen está comprimida
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
            else:
                # La imagen no está comprimida
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            cv2.imshow("Image Window", cv_image)
            cv2.waitKey(1)
        else:
            # Si los datos no están presentes en el mensaje
            print("El mensaje sensor/Image no contiene datos.")

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
