import numpy as np

import cv2

import vision_util


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
    parallel_score = abs(vision_util.angle(left) - vision_util.angle(right))
    parallel_score = 100 - abs(parallel_score)

    # score based on the similarity in sizes
    sameness_score = abs(max(left[1][0], left[1][1]) - max(right[1][0], right[1][1]))
    sameness_score = 100 - abs(sameness_score)

    # score based on being located in the same spot on the Y-axis
    same_y_score = abs(left[1][1] - right[1][1])
    same_y_score = 100 - abs(same_y_score)

    return np.mean([space_score, parallel_score, same_y_score, sameness_score])


def find(img, hue_min=20, hue_max=175, sat_min=0, sat_max=255, val_min=0, val_max=255):
    """
    Detect the qualification gate.
    :param img: HSV image from the bottom camera
    :return: tuple of location of the center of the gate in a "targeting" coordinate system: origin is at center of image, axes range [-1, 1]
    """

    img = np.copy(img)

    bin = vision_util.hsv_threshold(img, hue_min, hue_max, sat_min, sat_max, val_min, val_max)

    canny = vision_util.canny(bin, 50)

    # find contours after first processing it with Canny edge detection
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hulls = vision_util.convex_hulls(contours)
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

    center_actual = (np.mean([left[0][0], right[0][0]]), np.mean([left[0][1], right[0][1]]))
    # shape[0] is the number of rows because matrices are dumb
    center = (center_actual[0] / img.shape[1], center_actual[1] / img.shape[0])
    # convert to the targeting system of [-1, 1]
    center = ((center[0] * 2) - 1, (center[1] * 2) - 1)

    return center