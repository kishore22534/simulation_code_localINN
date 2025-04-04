# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "volta_msgs: 4 messages, 0 services")

set(MSG_I_FLAGS "-Ivolta_msgs:/home/siva/cps_280_ws/src/volta/volta_msgs/msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(volta_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/RPM.msg" NAME_WE)
add_custom_target(_volta_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "volta_msgs" "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/RPM.msg" ""
)

get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Diagnostics.msg" NAME_WE)
add_custom_target(_volta_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "volta_msgs" "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Diagnostics.msg" ""
)

get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Table.msg" NAME_WE)
add_custom_target(_volta_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "volta_msgs" "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Table.msg" ""
)

get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/BMS.msg" NAME_WE)
add_custom_target(_volta_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "volta_msgs" "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/BMS.msg" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/RPM.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/volta_msgs
)
_generate_msg_cpp(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Diagnostics.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/volta_msgs
)
_generate_msg_cpp(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Table.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/volta_msgs
)
_generate_msg_cpp(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/BMS.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/volta_msgs
)

### Generating Services

### Generating Module File
_generate_module_cpp(volta_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/volta_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(volta_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(volta_msgs_generate_messages volta_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/RPM.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_cpp _volta_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Diagnostics.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_cpp _volta_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Table.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_cpp _volta_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/BMS.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_cpp _volta_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(volta_msgs_gencpp)
add_dependencies(volta_msgs_gencpp volta_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS volta_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/RPM.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/volta_msgs
)
_generate_msg_eus(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Diagnostics.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/volta_msgs
)
_generate_msg_eus(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Table.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/volta_msgs
)
_generate_msg_eus(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/BMS.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/volta_msgs
)

### Generating Services

### Generating Module File
_generate_module_eus(volta_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/volta_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(volta_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(volta_msgs_generate_messages volta_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/RPM.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_eus _volta_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Diagnostics.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_eus _volta_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Table.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_eus _volta_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/BMS.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_eus _volta_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(volta_msgs_geneus)
add_dependencies(volta_msgs_geneus volta_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS volta_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/RPM.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/volta_msgs
)
_generate_msg_lisp(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Diagnostics.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/volta_msgs
)
_generate_msg_lisp(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Table.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/volta_msgs
)
_generate_msg_lisp(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/BMS.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/volta_msgs
)

### Generating Services

### Generating Module File
_generate_module_lisp(volta_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/volta_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(volta_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(volta_msgs_generate_messages volta_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/RPM.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_lisp _volta_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Diagnostics.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_lisp _volta_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Table.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_lisp _volta_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/BMS.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_lisp _volta_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(volta_msgs_genlisp)
add_dependencies(volta_msgs_genlisp volta_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS volta_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/RPM.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/volta_msgs
)
_generate_msg_nodejs(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Diagnostics.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/volta_msgs
)
_generate_msg_nodejs(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Table.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/volta_msgs
)
_generate_msg_nodejs(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/BMS.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/volta_msgs
)

### Generating Services

### Generating Module File
_generate_module_nodejs(volta_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/volta_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(volta_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(volta_msgs_generate_messages volta_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/RPM.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_nodejs _volta_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Diagnostics.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_nodejs _volta_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Table.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_nodejs _volta_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/BMS.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_nodejs _volta_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(volta_msgs_gennodejs)
add_dependencies(volta_msgs_gennodejs volta_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS volta_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/RPM.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/volta_msgs
)
_generate_msg_py(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Diagnostics.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/volta_msgs
)
_generate_msg_py(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Table.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/volta_msgs
)
_generate_msg_py(volta_msgs
  "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/BMS.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/volta_msgs
)

### Generating Services

### Generating Module File
_generate_module_py(volta_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/volta_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(volta_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(volta_msgs_generate_messages volta_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/RPM.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_py _volta_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Diagnostics.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_py _volta_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/Table.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_py _volta_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/siva/cps_280_ws/src/volta/volta_msgs/msg/BMS.msg" NAME_WE)
add_dependencies(volta_msgs_generate_messages_py _volta_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(volta_msgs_genpy)
add_dependencies(volta_msgs_genpy volta_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS volta_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/volta_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/volta_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(volta_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/volta_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/volta_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(volta_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/volta_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/volta_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(volta_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/volta_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/volta_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(volta_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/volta_msgs)
  install(CODE "execute_process(COMMAND \"/home/siva/miniconda3/envs/proj/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/volta_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/volta_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(volta_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
