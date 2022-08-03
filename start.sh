#!/bin/bash


GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_cpu  --calculator_graph_config_file=mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt > bocvv.log 2>&1 &

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/upper_body_pose_tracking/upper_body_pose_tracking_cpu  --calculator_graph_config_file=mediapipe/graphs/pose_tracking/upper_body_pose_tracking_cpu.pbtxt > bocvv.log 2>&1 &

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_cpu  --calculator_graph_config_file=mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt > bocvv.log 2>&1 &

:<<!
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt > bocvv.log 2>&1 &

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_tracking/object_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/tracking/object_detection_tracking_desktop_live.pbtxt > bocvv.log 2>&1 &
!
