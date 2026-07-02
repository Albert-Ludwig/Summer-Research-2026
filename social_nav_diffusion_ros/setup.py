from setuptools import find_packages, setup

package_name = 'social_nav_diffusion_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/raw_eval.yaml', 'config/guarded_eval.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'social_nav_diffusion_node = social_nav_diffusion_ros.social_nav_diffusion_node:main',
            'policy_cmd_vel_node = social_nav_diffusion_ros.policy_cmd_vel_node:main',
            'nav2_goal_to_pose_bridge = social_nav_diffusion_ros.nav2_goal_to_pose_bridge:main',
        ],
    },
)
