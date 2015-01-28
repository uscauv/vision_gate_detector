import cv2

import numpy as np

import vision_common


def hull_score(hull):
    """
    Give a score to a convex hull based on how likely it is to be a qualification gate element.
    :param hull: convex hull to test
    :return: Score based on the ratio of side lengths and a minimum area
    """
    rect = cv2.minAreaRect(hull)
    shorter_side = min(rect[1])
    longer_side = max(rect[1])

    # the orange tape marker is 3 inches by 4 feet so the ratio of long : short side = 16
    ratio_score = 1 / (abs((longer_side / shorter_side) - 16) + 0.001)  # add 0.001 to prevent NaN

    score = ratio_score + cv2.contourArea(hull)

    # cut off minimum area at 500 px^2
    if cv2.contourArea(hull) < 500:
        return 0

    return score


def score_pair(left, right):
    # calculate the space in pixels that should be between each side
    # the actual dimension is 10 feet, so the space should be 2.5x the height (longer dimension)
    desired_space = max(left[1][0], left[1][1]) * 2.5

    # score based on the distance between the two sides
    # left.x - right.x
    space_score = abs(abs(left[0][0] - right[0][0]) - desired_space)
    space_score = 100 - .5 * abs(space_score)

    # score based on the parallel-ity of the two sides
    parallel_score = abs(vision_common.angle(left) - vision_common.angle(right))
    parallel_score = 100 - abs(parallel_score)

    # score based on the similarity in sizes
    sameness_score = abs(max(left[1][0], left[1][1]) - max(right[1][0], right[1][1]))
    sameness_score = 100 - abs(sameness_score)

    # score based on being located in the same spot on the Y-axis
    same_y_score = abs(left[1][1] - right[1][1])
    same_y_score = 100 - abs(same_y_score)

    return np.mean([space_score, parallel_score, same_y_score, sameness_score])


class GateDetector:
    def __init__(self):
        pass

    def find(self, img):
        """
        Detect the qualification gate.
        :param img: HSV image from the bottom camera
        :return: tuple of location of the center of the gate in a "targeting" coordinate system: origin is at center of image, axes range [-1, 1]
        """

        img = np.copy(img)

        # TODO: get rid of these magic numbers
        bin = vision_common.hsv_threshold(img, 20, 175, 0, 255, 0, 255)

        canny = vision_common.canny(bin, 50)

        # find contours after first processing it with Canny edge detection
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hulls = vision_common.convex_hulls(contours)
        cv2.drawContours(bin, hulls, -1, 255)

        cv2.imshow('bin', bin)

        hulls.sort(key=hull_score)

        if len(hulls) < 2:
            return ()

        # get the two highest scoring candidates
        left = cv2.minAreaRect(hulls[0])
        right = cv2.minAreaRect(hulls[1])

        # if we got left and right mixed up, switch them
        if right[0][0] < left[0][0]:
            left, right = right, left

        confidence = score_pair(left, right)
        if confidence < 80:
            return 0, 0

        # draw hulls in Blaze Orange
        cv2.drawContours(img, hulls, -1, (0, 102, 255), -1)
        # draw green outlines so we know it actually detected it
        cv2.drawContours(img, hulls, -1, (0, 255, 0), 2)

        cv2.imshow('img', img)

        return np.mean([left[0][0], right[0][0]]), np.mean([left[0][1], right[0][1]])


img = cv2.imread('sample.jpg', cv2.IMREAD_COLOR)
detector = GateDetector()
loc = detector.find(img)

cv2.waitKey(0)
cv2.destroyAllWindows()