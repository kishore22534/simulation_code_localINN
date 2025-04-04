# Instructions to set up Volta Robot

In order to setup the volta robot software on your raspberry pi, you will need to compile certain libraries into your package.

First create a workspace in your home directory:
```
mkdir catkin_ws
```
then create a source folder to place your code
```
cd catkin_ws
mkdir src
cd src
```
Further you will need to install the volta package from Github:
```
git clone https://github.com/botsync/volta.git
```
From the `catkin_ws` directory, install all the required dependencies using:
```
rosdep install --from-paths src --ignore-src -r -y
```
Next build the workspace using
```
catkin_make
```
There is one package that is not made available to the public, we will need to include this package by downloading it from [here](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/namanmenezes_iisc_ac_in/EtVGzVGxR9tNq2eBJcaEoPIBnIPwa0orSKFJMVX2Yc-kIg?e=PmAthZ)
Note: The above file is private and must not be shared at any cost outside the usage of this course.

Extract the zip file into the source folder of your workspace under the volta folder. Your file setup should now look like:
```
catkin_ws
└── src
    ├── CMakeLists.txt -> /opt/ros/noetic/share/catkin/cmake/toplevel.cmake
    └── volta
        ├── autostart.sh
        ├── LICENSE
        ├── README.md
        ├── volta_base
        ├── volta_control
        ├── volta_description
        ├── volta_hardware
        ├── volta_localization
        ├── volta_msgs
        ├── volta_navigation
        ├── volta_rules
        └── volta_teleoperator
```
Run `catkin_make` again 

## Updating the udev rules
In order to connect to the volta you need to add some rules so that ubuntu knows where to find the ports that you are mentioning.

To update the rules change directory to the udev folder:
```
cd /etc/udev/rules.d/
```

Then create a file with these new udev rules:
```
sudo nano volta.rules
```

Add the following rules to the system 
```
KERNEL=="ttyUSB*", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", MODE:="0777", SYMLINK+="mcu"
KERNEL=="ttyUSB*", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE:="0777", SYMLINK+="rplidar"
KERNEL=="ttyUSB*", ATTRS{idVendor}=="0000", ATTRS{idProduct}=="0000", MODE:="0777", SYMLINK+="imu"
KERNEL=="ttyUSB*", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b3a", MODE:="0777", SYMLINK+="camera
```

Save and Exit (Ctrl s + Ctrl x)

In order to get the system to use the updated rules you will need to reboot the system

## Running Teleoperation

In order to run teleop on the system you can run the following

```
roslaunch volta_base bringup.launch
```

On another terminal you can run

```
rosrun teleop_twist_keyboard teleop_twist_keyboard.py 
```
