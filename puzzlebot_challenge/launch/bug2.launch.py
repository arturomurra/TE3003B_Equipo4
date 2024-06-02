from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command


def generate_launch_description():
    localisation = Node(
            package = 'puzzlebot_challenge',
            executable = 'odometry',
            name = 'localisation'
    )
    trayectory = Node(
        package='puzzlebot_challenge',
        executable='trajectory_control',
        name='trajectory_control'
    )

    aruco = Node(
        package='puzzlebot_challenge',
        executable='aruco',
        name='aruco'
    )
    
    bug2 = Node(
        package='puzzlebot_challenge',
        executable='bug2',
        name='Bug2'
    )


    return LaunchDescription([
        localisation,
        trayectory,
        bug2,
        #aruco,
    ])

if __name__ == '__main__':
    generate_launch_description()
