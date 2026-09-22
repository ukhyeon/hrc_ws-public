from setuptools import find_packages, setup

package_name = 'hrc_body_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    package_data={
        package_name: [
            'models/backbones/hrnet/models/*.yaml',
            'models/cliff_hr48/*.npz',
            'models/cliff_hr48/*.npy',
            'models/cliff_hr48/*.pth',
            'models/cliff_hr48/*.pt',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robotics',
    maintainer_email='treyu1231@naver.com',
    description='Human sensing and dynamics estimation for human-robot collaboration',
    license='Apache-2.0 AND MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sensing_start = hrc_body_tracking.sensing_node:main',
            'human_dynamics = hrc_body_tracking.human_dynamics_node:main',
            'human_dynamics_debug = hrc_body_tracking.human_dynamics_node_for_validation:main',
        ],
    },
)
