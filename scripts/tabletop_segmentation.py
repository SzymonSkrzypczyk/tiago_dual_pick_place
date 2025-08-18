#!/usr/bin/env python3

import rospy
import ros_numpy
import numpy as np
import open3d as o3d

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from moveit_commander import PlanningSceneInterface


class TabletopSegmentation:
    def __init__(self):
        rospy.init_node("tabletop_segmentation_node", anonymous=True)
        self.scene = PlanningSceneInterface()
        rospy.sleep(2.0)

        rospy.Subscriber("/xtion/depth_registered/points", PointCloud2, self.cloud_cb, queue_size=1)
        rospy.loginfo("Subscribed to depth camera point cloud")

        rospy.spin()

    def cloud_cb(self, cloud_msg):
        pc = ros_numpy.point_cloud2.pointcloud2_to_array(cloud_msg)

        points = np.zeros((pc.shape[0], 3), dtype=np.float32)
        points[:, 0] = pc['x']
        points[:, 1] = pc['y']
        points[:, 2] = pc['z']

        mask = ~np.isnan(points).any(axis=1)
        points = points[mask]

        if points.shape[0] == 0:
            rospy.logwarn("No valid points in cloud")
            return

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(points)
        cloud_o3d = cloud_o3d.voxel_down_sample(voxel_size=0.01)

        _, inliers = cloud_o3d.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )
        objects_cloud = cloud_o3d.select_by_index(inliers, invert=True)

        labels = np.array(objects_cloud.cluster_dbscan(
            eps=0.03, min_points=50, print_progress=False))

        if labels.max() == -1:
            rospy.logwarn("No clusters found")
            return

        rospy.loginfo(f"Detected {labels.max() + 1} objects")

        for i in range(5):
            self.scene.remove_world_object(f"Box_{i}")

        for i in range(labels.max() + 1):
            cluster = objects_cloud.select_by_index(np.where(labels == i)[0])
            pts = np.asarray(cluster.points)

            centroid = np.mean(pts, axis=0)
            min_pt = np.min(pts, axis=0)
            max_pt = np.max(pts, axis=0)
            size = max_pt - min_pt

            pose = PoseStamped()
            pose.header.frame_id = cloud_msg.header.frame_id
            pose.pose.position.x = float(centroid[0])
            pose.pose.position.y = float(centroid[1])
            pose.pose.position.z = float(centroid[2])
            pose.pose.orientation.w = 1.0

            name = f"Box_{i}"
            self.scene.add_box(name, pose, size=tuple(size))
            rospy.loginfo(f"Added {name} at {centroid} with size {size}")


if __name__ == "__main__":
    try:
        TabletopSegmentation()
    except rospy.ROSInterruptException:
        pass
