#!/usr/bin/env python3

from pathlib import Path
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import rospkg

rospack = rospkg.RosPack()
package_path = rospack.get_path('tiago_dual_pick_place')
model_path = Path(package_path) / "models"

model = cv2.dnn.readNet(
    str(model_path / "MobileNetSSD_deploy.caffemodel"), 
    str(model_path / "MobileNetSSD_deploy.prototxt.txt")
)

class CameraViewer:
    def __init__(self):
        rospy.init_node('camera_viewer', anonymous=True)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/xtion/rgb/image_raw', Image, self.callback)
        
        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            h, w = cv_image.shape[:2]
            blob = cv2.dnn.blobFromImage(image=cv_image, scalefactor=0.007843, size=(300, 300), mean=(127.5, 127.5, 127.5))
            model.setInput(blob)
            outputs = model.forward()

            for i in range(outputs.shape[2]):
                confidence = outputs[0, 0, i, 2]

                if confidence > 0.6:
                    box = outputs[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")

                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow('Camera Feed', cv_image)
            cv2.waitKey(1) 
        except Exception as e:
            rospy.logerr("Error converting image: {}".format(e))

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    viewer = CameraViewer()
    viewer.run()