# Volta Simulation Instructions


Create a workspace on your desktop system (with Ubuntu and ROS Noetic installed)

    mkdir cps_280_ws
    cd cps_280_ws
    mkdir src
    cd src

Clone the volta workspace 

```
git clone https://github.com/botsync/volta.git 
cd volta/
git checkout noetic-devel
cd ..
git clone https://github.com/botsync/volta_simulation.git
cd volta_simulation/
git checkout noetic-devel 
```

NOTE: If you are using ROS Melodic then checkout the `melodic-devel`

This repository must also be cloned within the `cps_280_ws`

```
git clone https://github.com/airl-iisc/CPS280.git
```

## Build the simulation

From the `cps_280_ws` root folder
```
cd ~/cps_280_ws
```

Install dependencies
```
rosdep install --from-paths src --ignore-src -r -y
```

Run 
```
catkin_make
```
or
```
catkin build
```

## Launching gazebo simulation
```
roslaunch final_assignment gazebo.launch
```
## Launching the volta robot
```
roslaunch final_assignment simulation.launch
```

# Milestone 1
Subscribe to the robot camera feed and print the color of the box in front of the robot.

You can launch the robot in front of the green box using
```
roslaunch final_assignment simulation.launch green:=true
```
