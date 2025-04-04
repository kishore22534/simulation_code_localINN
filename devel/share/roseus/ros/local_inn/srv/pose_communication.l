;; Auto-generated. Do not edit!


(when (boundp 'local_inn::pose_communication)
  (if (not (find-package "LOCAL_INN"))
    (make-package "LOCAL_INN"))
  (shadow 'pose_communication (find-package "LOCAL_INN")))
(unless (find-package "LOCAL_INN::POSE_COMMUNICATION")
  (make-package "LOCAL_INN::POSE_COMMUNICATION"))
(unless (find-package "LOCAL_INN::POSE_COMMUNICATIONREQUEST")
  (make-package "LOCAL_INN::POSE_COMMUNICATIONREQUEST"))
(unless (find-package "LOCAL_INN::POSE_COMMUNICATIONRESPONSE")
  (make-package "LOCAL_INN::POSE_COMMUNICATIONRESPONSE"))

(in-package "ROS")





(defclass local_inn::pose_communicationRequest
  :super ros::object
  :slots (_lidar_data ))

(defmethod local_inn::pose_communicationRequest
  (:init
   (&key
    ((:lidar_data __lidar_data) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _lidar_data __lidar_data)
   self)
  (:lidar_data
   (&optional __lidar_data)
   (if __lidar_data (setq _lidar_data __lidar_data)) _lidar_data)
  (:serialization-length
   ()
   (+
    ;; float32[] _lidar_data
    (* 4    (length _lidar_data)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _lidar_data
     (write-long (length _lidar_data) s)
     (dotimes (i (length _lidar_data))
       (sys::poke (elt _lidar_data i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _lidar_data
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _lidar_data (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _lidar_data i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(defclass local_inn::pose_communicationResponse
  :super ros::object
  :slots (_pose ))

(defmethod local_inn::pose_communicationResponse
  (:init
   (&key
    ((:pose __pose) (make-array 3 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _pose __pose)
   self)
  (:pose
   (&optional __pose)
   (if __pose (setq _pose __pose)) _pose)
  (:serialization-length
   ()
   (+
    ;; float32[3] _pose
    (* 4    3)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[3] _pose
     (dotimes (i 3)
       (sys::poke (elt _pose i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[3] _pose
   (dotimes (i (length _pose))
     (setf (elt _pose i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;;
   self)
  )

(defclass local_inn::pose_communication
  :super ros::object
  :slots ())

(setf (get local_inn::pose_communication :md5sum-) "5aa28f834219f856593a6cf0bb2c70d5")
(setf (get local_inn::pose_communication :datatype-) "local_inn/pose_communication")
(setf (get local_inn::pose_communication :request) local_inn::pose_communicationRequest)
(setf (get local_inn::pose_communication :response) local_inn::pose_communicationResponse)

(defmethod local_inn::pose_communicationRequest
  (:response () (instance local_inn::pose_communicationResponse :init)))

(setf (get local_inn::pose_communicationRequest :md5sum-) "5aa28f834219f856593a6cf0bb2c70d5")
(setf (get local_inn::pose_communicationRequest :datatype-) "local_inn/pose_communicationRequest")
(setf (get local_inn::pose_communicationRequest :definition-)
      "# Request
float32[] lidar_data  # LIDAR scan data (array of floats)

---
# Response
float32[3] pose  # x, y, theta



")

(setf (get local_inn::pose_communicationResponse :md5sum-) "5aa28f834219f856593a6cf0bb2c70d5")
(setf (get local_inn::pose_communicationResponse :datatype-) "local_inn/pose_communicationResponse")
(setf (get local_inn::pose_communicationResponse :definition-)
      "# Request
float32[] lidar_data  # LIDAR scan data (array of floats)

---
# Response
float32[3] pose  # x, y, theta



")



(provide :local_inn/pose_communication "5aa28f834219f856593a6cf0bb2c70d5")


