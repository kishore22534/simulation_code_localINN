
(cl:in-package :asdf)

(defsystem "local_inn-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "pose_communication" :depends-on ("_package_pose_communication"))
    (:file "_package_pose_communication" :depends-on ("_package"))
    (:file "pose_communication_camera" :depends-on ("_package_pose_communication_camera"))
    (:file "_package_pose_communication_camera" :depends-on ("_package"))
  ))