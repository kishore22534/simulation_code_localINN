; Auto-generated. Do not edit!


(cl:in-package local_inn-srv)


;//! \htmlinclude pose_communication-request.msg.html

(cl:defclass <pose_communication-request> (roslisp-msg-protocol:ros-message)
  ((lidar_data
    :reader lidar_data
    :initarg :lidar_data
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass pose_communication-request (<pose_communication-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <pose_communication-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'pose_communication-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name local_inn-srv:<pose_communication-request> is deprecated: use local_inn-srv:pose_communication-request instead.")))

(cl:ensure-generic-function 'lidar_data-val :lambda-list '(m))
(cl:defmethod lidar_data-val ((m <pose_communication-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader local_inn-srv:lidar_data-val is deprecated.  Use local_inn-srv:lidar_data instead.")
  (lidar_data m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <pose_communication-request>) ostream)
  "Serializes a message object of type '<pose_communication-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'lidar_data))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'lidar_data))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <pose_communication-request>) istream)
  "Deserializes a message object of type '<pose_communication-request>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'lidar_data) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'lidar_data)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<pose_communication-request>)))
  "Returns string type for a service object of type '<pose_communication-request>"
  "local_inn/pose_communicationRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'pose_communication-request)))
  "Returns string type for a service object of type 'pose_communication-request"
  "local_inn/pose_communicationRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<pose_communication-request>)))
  "Returns md5sum for a message object of type '<pose_communication-request>"
  "5aa28f834219f856593a6cf0bb2c70d5")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'pose_communication-request)))
  "Returns md5sum for a message object of type 'pose_communication-request"
  "5aa28f834219f856593a6cf0bb2c70d5")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<pose_communication-request>)))
  "Returns full string definition for message of type '<pose_communication-request>"
  (cl:format cl:nil "# Request~%float32[] lidar_data  # LIDAR scan data (array of floats)~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'pose_communication-request)))
  "Returns full string definition for message of type 'pose_communication-request"
  (cl:format cl:nil "# Request~%float32[] lidar_data  # LIDAR scan data (array of floats)~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <pose_communication-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'lidar_data) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <pose_communication-request>))
  "Converts a ROS message object to a list"
  (cl:list 'pose_communication-request
    (cl:cons ':lidar_data (lidar_data msg))
))
;//! \htmlinclude pose_communication-response.msg.html

(cl:defclass <pose_communication-response> (roslisp-msg-protocol:ros-message)
  ((pose
    :reader pose
    :initarg :pose
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass pose_communication-response (<pose_communication-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <pose_communication-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'pose_communication-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name local_inn-srv:<pose_communication-response> is deprecated: use local_inn-srv:pose_communication-response instead.")))

(cl:ensure-generic-function 'pose-val :lambda-list '(m))
(cl:defmethod pose-val ((m <pose_communication-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader local_inn-srv:pose-val is deprecated.  Use local_inn-srv:pose instead.")
  (pose m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <pose_communication-response>) ostream)
  "Serializes a message object of type '<pose_communication-response>"
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'pose))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <pose_communication-response>) istream)
  "Deserializes a message object of type '<pose_communication-response>"
  (cl:setf (cl:slot-value msg 'pose) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'pose)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<pose_communication-response>)))
  "Returns string type for a service object of type '<pose_communication-response>"
  "local_inn/pose_communicationResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'pose_communication-response)))
  "Returns string type for a service object of type 'pose_communication-response"
  "local_inn/pose_communicationResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<pose_communication-response>)))
  "Returns md5sum for a message object of type '<pose_communication-response>"
  "5aa28f834219f856593a6cf0bb2c70d5")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'pose_communication-response)))
  "Returns md5sum for a message object of type 'pose_communication-response"
  "5aa28f834219f856593a6cf0bb2c70d5")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<pose_communication-response>)))
  "Returns full string definition for message of type '<pose_communication-response>"
  (cl:format cl:nil "# Response~%float32[3] pose  # x, y, theta~%~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'pose_communication-response)))
  "Returns full string definition for message of type 'pose_communication-response"
  (cl:format cl:nil "# Response~%float32[3] pose  # x, y, theta~%~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <pose_communication-response>))
  (cl:+ 0
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'pose) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <pose_communication-response>))
  "Converts a ROS message object to a list"
  (cl:list 'pose_communication-response
    (cl:cons ':pose (pose msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'pose_communication)))
  'pose_communication-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'pose_communication)))
  'pose_communication-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'pose_communication)))
  "Returns string type for a service object of type '<pose_communication>"
  "local_inn/pose_communication")