; Auto-generated. Do not edit!


(cl:in-package local_inn-srv)


;//! \htmlinclude pose_communication_camera-request.msg.html

(cl:defclass <pose_communication_camera-request> (roslisp-msg-protocol:ros-message)
  ((image_data
    :reader image_data
    :initarg :image_data
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass pose_communication_camera-request (<pose_communication_camera-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <pose_communication_camera-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'pose_communication_camera-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name local_inn-srv:<pose_communication_camera-request> is deprecated: use local_inn-srv:pose_communication_camera-request instead.")))

(cl:ensure-generic-function 'image_data-val :lambda-list '(m))
(cl:defmethod image_data-val ((m <pose_communication_camera-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader local_inn-srv:image_data-val is deprecated.  Use local_inn-srv:image_data instead.")
  (image_data m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <pose_communication_camera-request>) ostream)
  "Serializes a message object of type '<pose_communication_camera-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'image_data))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'image_data))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <pose_communication_camera-request>) istream)
  "Deserializes a message object of type '<pose_communication_camera-request>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'image_data) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'image_data)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<pose_communication_camera-request>)))
  "Returns string type for a service object of type '<pose_communication_camera-request>"
  "local_inn/pose_communication_cameraRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'pose_communication_camera-request)))
  "Returns string type for a service object of type 'pose_communication_camera-request"
  "local_inn/pose_communication_cameraRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<pose_communication_camera-request>)))
  "Returns md5sum for a message object of type '<pose_communication_camera-request>"
  "c4fc5fe5d8e49992c022543ae4cff221")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'pose_communication_camera-request)))
  "Returns md5sum for a message object of type 'pose_communication_camera-request"
  "c4fc5fe5d8e49992c022543ae4cff221")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<pose_communication_camera-request>)))
  "Returns full string definition for message of type '<pose_communication_camera-request>"
  (cl:format cl:nil "# Request~%float32[] image_data  # camera scan data but using only pose as of now (array of floats)~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'pose_communication_camera-request)))
  "Returns full string definition for message of type 'pose_communication_camera-request"
  (cl:format cl:nil "# Request~%float32[] image_data  # camera scan data but using only pose as of now (array of floats)~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <pose_communication_camera-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'image_data) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <pose_communication_camera-request>))
  "Converts a ROS message object to a list"
  (cl:list 'pose_communication_camera-request
    (cl:cons ':image_data (image_data msg))
))
;//! \htmlinclude pose_communication_camera-response.msg.html

(cl:defclass <pose_communication_camera-response> (roslisp-msg-protocol:ros-message)
  ((pose
    :reader pose
    :initarg :pose
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass pose_communication_camera-response (<pose_communication_camera-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <pose_communication_camera-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'pose_communication_camera-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name local_inn-srv:<pose_communication_camera-response> is deprecated: use local_inn-srv:pose_communication_camera-response instead.")))

(cl:ensure-generic-function 'pose-val :lambda-list '(m))
(cl:defmethod pose-val ((m <pose_communication_camera-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader local_inn-srv:pose-val is deprecated.  Use local_inn-srv:pose instead.")
  (pose m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <pose_communication_camera-response>) ostream)
  "Serializes a message object of type '<pose_communication_camera-response>"
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'pose))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <pose_communication_camera-response>) istream)
  "Deserializes a message object of type '<pose_communication_camera-response>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<pose_communication_camera-response>)))
  "Returns string type for a service object of type '<pose_communication_camera-response>"
  "local_inn/pose_communication_cameraResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'pose_communication_camera-response)))
  "Returns string type for a service object of type 'pose_communication_camera-response"
  "local_inn/pose_communication_cameraResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<pose_communication_camera-response>)))
  "Returns md5sum for a message object of type '<pose_communication_camera-response>"
  "c4fc5fe5d8e49992c022543ae4cff221")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'pose_communication_camera-response)))
  "Returns md5sum for a message object of type 'pose_communication_camera-response"
  "c4fc5fe5d8e49992c022543ae4cff221")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<pose_communication_camera-response>)))
  "Returns full string definition for message of type '<pose_communication_camera-response>"
  (cl:format cl:nil "# Response~%float32[3] pose  # x, y, theta~%~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'pose_communication_camera-response)))
  "Returns full string definition for message of type 'pose_communication_camera-response"
  (cl:format cl:nil "# Response~%float32[3] pose  # x, y, theta~%~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <pose_communication_camera-response>))
  (cl:+ 0
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'pose) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <pose_communication_camera-response>))
  "Converts a ROS message object to a list"
  (cl:list 'pose_communication_camera-response
    (cl:cons ':pose (pose msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'pose_communication_camera)))
  'pose_communication_camera-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'pose_communication_camera)))
  'pose_communication_camera-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'pose_communication_camera)))
  "Returns string type for a service object of type '<pose_communication_camera>"
  "local_inn/pose_communication_camera")