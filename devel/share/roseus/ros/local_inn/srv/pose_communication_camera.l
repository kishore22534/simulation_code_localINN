;; Auto-generated. Do not edit!


(when (boundp 'local_inn::pose_communication_camera)
  (if (not (find-package "LOCAL_INN"))
    (make-package "LOCAL_INN"))
  (shadow 'pose_communication_camera (find-package "LOCAL_INN")))
(unless (find-package "LOCAL_INN::POSE_COMMUNICATION_CAMERA")
  (make-package "LOCAL_INN::POSE_COMMUNICATION_CAMERA"))
(unless (find-package "LOCAL_INN::POSE_COMMUNICATION_CAMERAREQUEST")
  (make-package "LOCAL_INN::POSE_COMMUNICATION_CAMERAREQUEST"))
(unless (find-package "LOCAL_INN::POSE_COMMUNICATION_CAMERARESPONSE")
  (make-package "LOCAL_INN::POSE_COMMUNICATION_CAMERARESPONSE"))

(in-package "ROS")





(defclass local_inn::pose_communication_cameraRequest
  :super ros::object
  :slots (_image_data ))

(defmethod local_inn::pose_communication_cameraRequest
  (:init
   (&key
    ((:image_data __image_data) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _image_data __image_data)
   self)
  (:image_data
   (&optional __image_data)
   (if __image_data (setq _image_data __image_data)) _image_data)
  (:serialization-length
   ()
   (+
    ;; float32[] _image_data
    (* 4    (length _image_data)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _image_data
     (write-long (length _image_data) s)
     (dotimes (i (length _image_data))
       (sys::poke (elt _image_data i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _image_data
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _image_data (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _image_data i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(defclass local_inn::pose_communication_cameraResponse
  :super ros::object
  :slots (_pose ))

(defmethod local_inn::pose_communication_cameraResponse
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

(defclass local_inn::pose_communication_camera
  :super ros::object
  :slots ())

(setf (get local_inn::pose_communication_camera :md5sum-) "c4fc5fe5d8e49992c022543ae4cff221")
(setf (get local_inn::pose_communication_camera :datatype-) "local_inn/pose_communication_camera")
(setf (get local_inn::pose_communication_camera :request) local_inn::pose_communication_cameraRequest)
(setf (get local_inn::pose_communication_camera :response) local_inn::pose_communication_cameraResponse)

(defmethod local_inn::pose_communication_cameraRequest
  (:response () (instance local_inn::pose_communication_cameraResponse :init)))

(setf (get local_inn::pose_communication_cameraRequest :md5sum-) "c4fc5fe5d8e49992c022543ae4cff221")
(setf (get local_inn::pose_communication_cameraRequest :datatype-) "local_inn/pose_communication_cameraRequest")
(setf (get local_inn::pose_communication_cameraRequest :definition-)
      "# Request
float32[] image_data  # camera scan data but using only pose as of now (array of floats)

---
# Response
float32[3] pose  # x, y, theta



")

(setf (get local_inn::pose_communication_cameraResponse :md5sum-) "c4fc5fe5d8e49992c022543ae4cff221")
(setf (get local_inn::pose_communication_cameraResponse :datatype-) "local_inn/pose_communication_cameraResponse")
(setf (get local_inn::pose_communication_cameraResponse :definition-)
      "# Request
float32[] image_data  # camera scan data but using only pose as of now (array of floats)

---
# Response
float32[3] pose  # x, y, theta



")



(provide :local_inn/pose_communication_camera "c4fc5fe5d8e49992c022543ae4cff221")


