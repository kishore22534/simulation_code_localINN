# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "local_inn: 0 messages, 2 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(local_inn_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication.srv" NAME_WE)
add_custom_target(_local_inn_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "local_inn" "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication.srv" ""
)

get_filename_component(_filename "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication_camera.srv" NAME_WE)
add_custom_target(_local_inn_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "local_inn" "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication_camera.srv" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(local_inn
  "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/local_inn
)
_generate_srv_cpp(local_inn
  "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication_camera.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/local_inn
)

### Generating Module File
_generate_module_cpp(local_inn
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/local_inn
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(local_inn_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(local_inn_generate_messages local_inn_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication.srv" NAME_WE)
add_dependencies(local_inn_generate_messages_cpp _local_inn_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication_camera.srv" NAME_WE)
add_dependencies(local_inn_generate_messages_cpp _local_inn_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(local_inn_gencpp)
add_dependencies(local_inn_gencpp local_inn_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS local_inn_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(local_inn
  "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/local_inn
)
_generate_srv_eus(local_inn
  "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication_camera.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/local_inn
)

### Generating Module File
_generate_module_eus(local_inn
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/local_inn
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(local_inn_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(local_inn_generate_messages local_inn_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication.srv" NAME_WE)
add_dependencies(local_inn_generate_messages_eus _local_inn_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication_camera.srv" NAME_WE)
add_dependencies(local_inn_generate_messages_eus _local_inn_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(local_inn_geneus)
add_dependencies(local_inn_geneus local_inn_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS local_inn_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(local_inn
  "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/local_inn
)
_generate_srv_lisp(local_inn
  "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication_camera.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/local_inn
)

### Generating Module File
_generate_module_lisp(local_inn
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/local_inn
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(local_inn_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(local_inn_generate_messages local_inn_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication.srv" NAME_WE)
add_dependencies(local_inn_generate_messages_lisp _local_inn_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication_camera.srv" NAME_WE)
add_dependencies(local_inn_generate_messages_lisp _local_inn_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(local_inn_genlisp)
add_dependencies(local_inn_genlisp local_inn_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS local_inn_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(local_inn
  "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/local_inn
)
_generate_srv_nodejs(local_inn
  "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication_camera.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/local_inn
)

### Generating Module File
_generate_module_nodejs(local_inn
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/local_inn
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(local_inn_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(local_inn_generate_messages local_inn_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication.srv" NAME_WE)
add_dependencies(local_inn_generate_messages_nodejs _local_inn_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication_camera.srv" NAME_WE)
add_dependencies(local_inn_generate_messages_nodejs _local_inn_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(local_inn_gennodejs)
add_dependencies(local_inn_gennodejs local_inn_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS local_inn_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(local_inn
  "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/local_inn
)
_generate_srv_py(local_inn
  "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication_camera.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/local_inn
)

### Generating Module File
_generate_module_py(local_inn
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/local_inn
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(local_inn_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(local_inn_generate_messages local_inn_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication.srv" NAME_WE)
add_dependencies(local_inn_generate_messages_py _local_inn_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/local_inn/srv/pose_communication_camera.srv" NAME_WE)
add_dependencies(local_inn_generate_messages_py _local_inn_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(local_inn_genpy)
add_dependencies(local_inn_genpy local_inn_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS local_inn_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/local_inn)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/local_inn
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(local_inn_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/local_inn)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/local_inn
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(local_inn_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/local_inn)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/local_inn
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(local_inn_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/local_inn)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/local_inn
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(local_inn_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/local_inn)
  install(CODE "execute_process(COMMAND \"/home/siva/miniconda3/envs/proj/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/local_inn\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/local_inn
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(local_inn_generate_messages_py std_msgs_generate_messages_py)
endif()
