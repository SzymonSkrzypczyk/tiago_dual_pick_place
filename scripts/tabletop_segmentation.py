#!/usr/bin/env python3

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from moveit_commander import PlanningSceneInterface, roscpp_initialize, roscpp_shutdown

import pcl
import pcl.pcl_visualization
import numpy as np


class TabletopSegmentation:
    def __init__(self):
        roscpp_initialize([])
        rospy.init_node('tabletop_segmentation_node', anonymous=True)
        self.scene = PlanningSceneInterface()
        rospy.sleep(2)

        # Subscribe to TIAGo depth camera pointcloud
        rospy.Subscriber("/xtion/depth_registered/points", PointCloud2, self.cloud_cb)
        rospy.loginfo("Subscribed to pointcloud topic.")
        rospy.spin()

    def cloud_cb(self, cloud_msg):
        # Convert ROS -> PCL
        points_list = []
        for p in pc2.read_points(cloud_msg, skip_nans=True):
            points_list.append([p[0], p[1], p[2]])
        cloud = pcl.PointCloud()
        cloud.from_list(points_list)

        # 1. Plane segmentation (remove table)
        seg = cloud.make_segmenter_normals(ksearch=50)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(0.01)
        indices, model = seg.segment()

        cloud_objects = cloud.extract(indices, negative=True)

        # 2. Clustering objects
        tree = cloud_objects.make_kdtree()
        ec = cloud_objects.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(0.02)  # 2cm
        ec.set_MinClusterSize(100)
        ec.set_MaxClusterSize(25000)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()

        rospy.loginfo("Found %d clusters (objects)" % len(cluster_indices))

        # Clear old objects
        for i in range(5):
            self.scene.remove_world_object("Box_%d" % i)

        # 3. Add clusters as boxes
        for i, indices in enumerate(cluster_indices):
            pts = np.array([cloud_objects[j] for j in indices])
            centroid = np.mean(pts, axis=0)
            min_pt = np.min(pts, axis=0)
            max_pt = np.max(pts, axis=0)
            size = max_pt - min_pt

            pose = PoseStamped()
            pose.header.frame_id = cloud_msg.header.frame_id
            pose.pose.position.x = centroid[0]
            pose.pose.position.y = centroid[1]
            pose.pose.position.z = centroid[2]
            pose.pose.orientation.w = 1.0

            name = "Box_%d" % i
            self.scene.add_box(name, pose, size=size)
            rospy.loginfo("Added object %s at %s" % (name, str(centroid)))

if __name__ == "__main__":
    try:
        TabletopSegmentation()
    except rospy.ROSInterruptException:
        pass
    roscpp_shutdown()
