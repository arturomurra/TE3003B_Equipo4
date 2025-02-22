from setuptools import find_packages, setup
from glob import glob


package_name = 'puzzlebot_challenge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*')),
        ('share/' + package_name + '/urdf', glob('urdf/*')),
        ('share/' + package_name + '/models', glob('models/*')),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arturo',
    maintainer_email='arturo.murra@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'puzzlebot_kinematics = puzzlebot_challenge.pose_sim:main',
            'odometry = puzzlebot_challenge.localisation:main',
            'joint_state_publisher = puzzlebot_challenge.JointStatePublisher:main',
            'velocity_control = puzzlebot_challenge.velocity_control:main',
            'image_stream = puzzlebot_challenge.image_stream:main',
            'aruco = puzzlebot_challenge.aruco_identification:main',
            'pose_sim = puzzlebot_challenge.pose_sim:main',
            'trajectory_control = puzzlebot_challenge.trajectory_control:main',
            'obstacule_sim=puzzlebot_challenge.obstacule_sim:main',
            'rplidar = puzzlebot_challenge.filter_scan:main',
            'oval_pose = puzzlebot_challenge.oval_pose_publisher:main',
            'tf_broadcaster = puzzlebot_challenge.tf_broadcaster:main',
            'bug2 = puzzlebot_challenge.bug2:main',
            'handle_object = puzzlebot_challenge.handle_object:main',
            'landmark = puzzlebot_challenge.landmark_detection:main',
        ],
    },
)
