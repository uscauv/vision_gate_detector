#!/usr/bin/env python

import rospy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_common.msg import Target, Targets
import gate_detector


class GateDetectorNode():
    def __init__(self):
        self.targets_pub = rospy.Publisher('/vision/gate_detector/targets', Targets, queue_size=10)
        self.target_pub = rospy.Publisher('/vision/gate_detector/target', Target, queue_size=10)
        self.bin_pub = rospy.Publisher('/vision/gate_detector/img/bin', Image, queue_size=10)
        self.result_img_pub = rospy.Publisher('/vision/gate_detector/img/result', Image, queue_size=10)
        rospy.init_node('gate_detector')

        self.bridge = CvBridge()

        rospy.Subscriber('/camera/bottom', Image, self.on_image)

        self.bridge = CvBridge()

    def on_image(self, img_msg):
        rospy.loginfo('got image')
        img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')

        intermediary_imgs = {}

        thresh = rospy.get_param('/vision/gate_detector/threshold',
                                 {'hue_min': 125, 'hue_max': 175,
                                  'sat_min': 240, 'sat_max': 255,
                                  'val_min': 215, 'val_max': 255})
        print(thresh)
        results = gate_detector.find(img, hue_min=thresh['hue_min'], hue_max=thresh['hue_max'],
                                                 sat_min=thresh['sat_min'], sat_max=thresh['sat_max'],
                                                 val_min=thresh['val_min'], val_max=thresh['val_max'],
                                                 output_images=intermediary_imgs)

        # publish intermediary images from the direction marker detector
        self.bin_pub.publish(self.bridge.cv2_to_imgmsg(intermediary_imgs['bin'], "mono8"))
        self.result_img_pub.publish(self.bridge.cv2_to_imgmsg(intermediary_imgs['result'], "bgr8"))

        # construct and publish the Target messages if there were detected targets
        if len(results) > 0:
            targets_msg = Targets()
            for rect in results:
                target = Target()
                target.header.stamp = rospy.Time.now()
                target.header.frame_id = 'gate_detector'
                target.x = rect[0][0]
                target.y = rect[0][1]
                target.width = rect[1][0]
                target.height = rect[1][1]
                target.angle = rect[2]
                targets_msg.targets.append(target)

            targets_msg.header.stamp = rospy.Time.now()
            self.target_pub.publish(targets_msg.targets[0])
            self.targets_pub.publish(targets_msg)


if __name__ == '__main__':
    node = GateDetectorNode()
    rospy.loginfo('created')
    rospy.spin()