import cv2 as cv
import numpy as np
import easygui
from math import sqrt, cos, sin, pi, atan2, acos, degrees
from functools import partial
from random import randint


ORIGINAL_IMAGE_WINDOW = 'Original Image'
SEGMENTED_HAND_WINDOW = 'Segmented Hand'
FINAL_SEGMENTED_HAND_WINDOW = 'Final Segmented Hand'


class HSVValues:

    def __init__(self):
        self.lower_hue = 0
        self.upper_hue = 180
        self.lower_saturation = 0
        self.upper_saturation = 255
        self.lower_value = 0
        self.upper_value = 255


class TrackBar:

    def __init__(self, obj, attribute_name, lower_value, upper_value, track_bar_name, window_name, event_handler):
        self.obj = obj
        self.attribute_name = attribute_name
        self.lower_value = lower_value
        self.upper_value = upper_value
        self.track_bar_name = track_bar_name
        self.window_name = window_name
        self.event_handler = event_handler
        cv.createTrackbar(track_bar_name, window_name, lower_value, upper_value, self.on_change)

    def on_change(self, value):
        setattr(self.obj, self.attribute_name, value)
        self.event_handler()


def on_change(img, hsv_values):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_bin = cv.inRange(img_hsv, (hsv_values.lower_hue, hsv_values.lower_saturation, hsv_values.lower_value),
                         (hsv_values.upper_hue, hsv_values.upper_saturation, hsv_values.upper_value))
    cv.imshow(SEGMENTED_HAND_WINDOW, img_bin)


def segment_hand(img, hsv_values):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_bin = cv.inRange(img_hsv, (hsv_values.lower_hue, hsv_values.lower_saturation, hsv_values.lower_value),
                         (hsv_values.upper_hue, hsv_values.upper_saturation, hsv_values.upper_value))
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(img_bin)
    hand_label = max([(label, stats[label - 1, cv.CC_STAT_AREA]) for label in range(1, num_labels + 1)],
                     key=lambda x: x[1])[0]
    segmented_hand = np.zeros(shape=labels.shape, dtype=np.uint8)
    for row in range(0, segmented_hand.shape[0]):
        for col in range(0, segmented_hand.shape[1]):
            if labels[row, col] == hand_label:
                segmented_hand[row, col] = 255
    return segmented_hand


def get_palm_point(segmented_hand):
    distance_transform = cv.distanceTransform(segmented_hand, cv.DIST_L2, 3)
    point = np.unravel_index(np.argmax(distance_transform, axis=None), distance_transform.shape)
    return point[1], point[0]


def distance(p1, p2):
    return sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def get_maximum_radius(palm_point, contour):
    distances = [distance(palm_point, point) for point in contour]
    return min(distances)


def get_wrist_points(palm_mask_points):
    max_dist = 0
    max_index = 0
    for index in range(0, len(palm_mask_points) - 1):
        p1 = palm_mask_points[index]
        p2 = palm_mask_points[index + 1]
        if distance(p1, p2) > max_dist:
            max_dist = distance(p1, p2)
            max_index = index
    return palm_mask_points[max_index], palm_mask_points[max_index + 1]


def get_sampled_points(palm_point, radius):
    sampled_points = []
    for angle in range(0, 360):
        x = radius * cos(angle * pi / 180.0) + palm_point[0]
        y = radius * sin(angle * pi / 180.0) + palm_point[1]
        sampled_points.append((int(x), int(y)))
    return sampled_points


def get_palm_mask_points(sampled_points, contour):
    palm_mask_points = []
    for sampled_point in sampled_points:
        distances = [(distance(sampled_point, point), point) for point in contour]
        closest_point = min(distances, key=lambda t: t[0])[1]
        palm_mask_points.append((closest_point[0], closest_point[1]))
    return palm_mask_points


def get_rotation_angle(palm_point, middle_wrist_point):
    x = palm_point[0] - middle_wrist_point[0]
    y = palm_point[1] - middle_wrist_point[1]
    hand_angle = atan2(y, x)
    rotation_angle = (hand_angle + pi / 2) * 180.0 / pi
    return rotation_angle


def get_rotation_matrix(palm_point, middle_wrist_point):
    rotation_angle = get_rotation_angle(palm_point, middle_wrist_point)
    rotation_matrix = cv.getRotationMatrix2D(palm_point, rotation_angle, 1)
    return rotation_matrix


def transform_point(point, angle, center):
    angle = 2 * pi - angle * pi / 180.0
    point = np.array([point[0], point[1], 1.0], dtype=np.float32)
    translation_matrix = np.array([[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [-center[0], -center[1], 1.0]], dtype=np.float32)
    rotation_matrix = np.array([[cos(angle), sin(angle), 0.0],
                                [-sin(angle), cos(angle), 0.0],
                                [0.0, 0.0, 1.0]], dtype=np.float32)
    inverse_translation_matrix = np.array([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [center[0], center[1], 1.0]], dtype=np.float32)
    point = point.dot(translation_matrix)
    point = point.dot(rotation_matrix)
    point = point.dot(inverse_translation_matrix)
    return np.array((point[0], point[1]), dtype=np.float32)


def get_fingers(segmented_hand):
    contours, _ = cv.findContours(segmented_hand, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    centers = [(int(cv.minAreaRect(contour)[0][0]), int(cv.minAreaRect(contour)[0][1])) for contour in contours]
    return list(zip(contours, centers))


def get_thumb_index(fingers, palm_point, first_wrist_point, second_wrist_point):
    wrist_vector = (second_wrist_point[0] - first_wrist_point[0], second_wrist_point[1] - first_wrist_point[1])
    wrist_vector = np.array(wrist_vector, dtype=np.float32)
    for index, (_, finger_center) in enumerate(fingers):
        finger_vector = (finger_center[0] - palm_point[0], finger_center[1] - palm_point[1])
        finger_vector = np.array(finger_vector, dtype=np.float32)
        cos_angle = wrist_vector.dot(finger_vector) / (np.linalg.norm(wrist_vector) * np.linalg.norm(finger_vector))
        angle = degrees(acos(cos_angle))
        if (angle < 90 and angle < 50) or (angle > 90 and (180.0 - angle) < 50):
            return index
    return None


def get_palm_line(segmented_hand, fingers, thumb_index):
    if thumb_index is not None:
        thumb_contour, _ = fingers[thumb_index]
        cv.drawContours(segmented_hand, [thumb_contour], -1, 0, -1)
    kernel = np.ones((5, 5), dtype=np.uint8)
    segmented_hand = cv.morphologyEx(segmented_hand, cv.MORPH_OPEN, kernel)
    for palm_line in range(segmented_hand.shape[0] - 1, -1, -1):
        copy_segmented_hand = np.copy(segmented_hand)
        for row in range(palm_line, segmented_hand.shape[0]):
            for col in range(0, segmented_hand.shape[1]):
                copy_segmented_hand[row, col] = 0
        num_labels, _, _, _ = cv.connectedComponentsWithStats(copy_segmented_hand)
        if num_labels >= 3:
            x_min = None
            x_max = None
            for col in range(0, segmented_hand.shape[1]):
                if segmented_hand[palm_line, col] != 0:
                    if x_min is None:
                        x_min = col
                    x_max = col
            return palm_line - 1


def get_finger_line(segmented_hand, fingers, thumb_index):
    for contour, _ in fingers:
        cv.drawContours(segmented_hand, [contour], -1, 0, -1)
    kernel = np.ones((5, 5), dtype=np.uint8)
    segmented_hand = cv.morphologyEx(segmented_hand, cv.MORPH_OPEN, kernel)
    global_x_min = None
    global_x_max = None
    palm_line = None
    stop = segmented_hand.shape[0]
    if thumb_index is not None:
        thumb_contour, center = fingers[thumb_index]
        rect = cv.minAreaRect(thumb_contour)
        box = cv.boxPoints(rect)
        box = sorted(box, key=lambda t: t[1])
        box = [np.array(point, dtype=np.int32) for point in box]
        stop = box[2][1]
    for row in range(0, stop):
        local_x_min = None
        local_x_max = None
        for col in range(0, segmented_hand.shape[1]):
            if segmented_hand[row, col] != 0:
                if local_x_min is None:
                    local_x_min = col
                local_x_max = col
        if global_x_max is None:
            global_x_max = local_x_max
            global_x_min = local_x_min
            palm_line = row
        else:
            if local_x_max is not None:
                if (local_x_max - local_x_min) > (global_x_max - global_x_min):
                    global_x_max = local_x_max
                    global_x_min = local_x_min
                    palm_line = row
    return (global_x_min, palm_line), (global_x_max, palm_line)


def get_horizontal_line_intersection(first_point, second_point, line):
    m = (second_point[1] - first_point[1]) / (second_point[0] - first_point[0])
    return int((line - first_point[1] + m * first_point[0]) / m), line


def get_fingers_status(fingers, thumb_index, first_finger_point, second_finger_point):
    quarter_length = (second_finger_point[0] - first_finger_point[0]) // 5
    fingers_status = [False, False, False, False, False]
    fingers_status[0] = thumb_index is not None
    for finger_index in range(0, len(fingers)):
        if finger_index != thumb_index:
            finger = fingers[finger_index]
            contour, center = finger
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            box = reversed(sorted(box, key=lambda t: t[1]))
            box = [np.array(point) for point in box]
            bottom_center = (box[0] + box[1]) / 2
            palm_intersection = get_horizontal_line_intersection(center, bottom_center, first_finger_point[1])
            length = int(palm_intersection[0]) - first_finger_point[0]
            finger_index = length // quarter_length
            if finger_index < 5:
                fingers_status[finger_index] = True
            else:
                fingers_status[4] = True
    return fingers_status


def draw_fingers_line(first_finger_point, second_finger_point, color_image):
    quarter_length = (second_finger_point[0] - first_finger_point[0]) // 5
    first_quartile = (first_finger_point[0] + quarter_length, first_finger_point[1])
    second_quartile = (first_finger_point[0] + 2 * quarter_length, first_finger_point[1])
    third_quartile = (first_finger_point[0] + 3 * quarter_length, first_finger_point[1])
    fourth_quartile = (first_finger_point[0] + 4 * quarter_length, first_finger_point[1])
    points = [first_finger_point, first_quartile, second_quartile, third_quartile, fourth_quartile, second_finger_point]
    for index in range(0, len(points) - 1):
        first_point = points[index]
        second_point = points[index + 1]
        red = randint(0, 255)
        green = randint(0, 255)
        blue = randint(0, 255)
        color = (blue, green, red)
        cv.line(color_image, first_point, second_point, color, 2)


def get_hand_attributes(segmented_hand):
    palm_point = get_palm_point(segmented_hand)
    contours, _ = cv.findContours(segmented_hand, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    contour = contours[0]
    contour = [point[0] for point in contour]
    maximum_radius = get_maximum_radius(palm_point, contour)
    radius = 1.2 * maximum_radius
    sampled_points = get_sampled_points(palm_point, radius)
    palm_mask_points = get_palm_mask_points(sampled_points, contour)
    first_wrist_point, second_wrist_point = get_wrist_points(palm_mask_points)
    middle_wrist_point = ((first_wrist_point[0] + second_wrist_point[0]) / 2.0,
                          (first_wrist_point[1] + second_wrist_point[1]) / 2.0)
    rotation_angle = get_rotation_angle(palm_point, middle_wrist_point)
    rotation_matrix = get_rotation_matrix(palm_point, middle_wrist_point)
    segmented_hand = cv.warpAffine(segmented_hand, rotation_matrix, (segmented_hand.shape[0], segmented_hand.shape[1]))
    first_wrist_point = np.array(first_wrist_point)
    second_wrist_point = np.array(second_wrist_point)
    palm_mask_points = [np.array(palm_mask_point) for palm_mask_point in palm_mask_points]
    first_wrist_point = transform_point(first_wrist_point, rotation_angle, palm_point)
    second_wrist_point = transform_point(second_wrist_point, rotation_angle, palm_point)
    palm_mask_points = [transform_point(palm_mask_point, rotation_angle, palm_point)
                        for palm_mask_point in palm_mask_points]
    palm_mask_points = np.array(palm_mask_points, dtype=np.int32)
    minimum_row = segmented_hand.shape[0]
    if first_wrist_point[1] < minimum_row:
        minimum_row = first_wrist_point[1]
    if second_wrist_point[1] < minimum_row:
        minimum_row = second_wrist_point[1]
    for row in range(int(minimum_row), segmented_hand.shape[0]):
        for col in range(0, segmented_hand.shape[1]):
            segmented_hand[row, col] = 0
    segmented_hand_no_arm = np.copy(segmented_hand)
    cv.fillPoly(segmented_hand, [palm_mask_points], 0)
    kernel = np.ones((5, 5), dtype=np.uint8)
    segmented_hand = cv.morphologyEx(segmented_hand, cv.MORPH_OPEN, kernel)
    fingers = get_fingers(segmented_hand)
    thumb_index = get_thumb_index(fingers, palm_point, first_wrist_point, second_wrist_point)
    first_finger_point, second_finger_point = get_finger_line(segmented_hand_no_arm, fingers, thumb_index)
    fingers_status = get_fingers_status(fingers, thumb_index, first_finger_point, second_finger_point)
    print(fingers_status)
    color_image = cv.cvtColor(segmented_hand, cv.COLOR_GRAY2BGR)
    cv.circle(color_image, palm_point, 5, (0, 0, 255), 2)
    cv.circle(color_image, palm_point, int(maximum_radius), (0, 255, 0))
    cv.circle(color_image, palm_point, int(radius), (255, 0, 0), 1)
    cv.circle(color_image, (first_wrist_point[0], first_wrist_point[1]), 3, (0, 0, 255), 3)
    cv.circle(color_image, (second_wrist_point[0], second_wrist_point[1]), 3, (0, 0, 255), 3)
    cv.line(color_image, (first_wrist_point[0], first_wrist_point[1]), (second_wrist_point[0], second_wrist_point[1]),
            (0, 255, 255), 3)
    draw_fingers_line(first_finger_point, second_finger_point, color_image)
    cv.imshow(FINAL_SEGMENTED_HAND_WINDOW, color_image)


def main():
    image_name = easygui.fileopenbox()
    img = cv.imread(image_name)
    hsv_values = HSVValues()
    cv.namedWindow(ORIGINAL_IMAGE_WINDOW)
    cv.imshow(ORIGINAL_IMAGE_WINDOW, img)
    lower_hue_bar = TrackBar(hsv_values, 'lower_hue', 0, 180, 'Lower Hue', ORIGINAL_IMAGE_WINDOW,
                             partial(on_change, img, hsv_values))
    upper_hue_bar = TrackBar(hsv_values, 'upper_hue', 0, 180, 'Upper Hue', ORIGINAL_IMAGE_WINDOW,
                             partial(on_change, img, hsv_values))
    lower_saturation_bar = TrackBar(hsv_values, 'lower_saturation', 0, 255, 'Lower Saturation', ORIGINAL_IMAGE_WINDOW,
                                    partial(on_change, img, hsv_values))
    upper_saturation_bar = TrackBar(hsv_values, 'upper_saturation', 0, 255, 'Upper Saturation', ORIGINAL_IMAGE_WINDOW,
                                    partial(on_change, img, hsv_values))
    lower_value_bar = TrackBar(hsv_values, 'lower_value', 0, 255, 'Lower Value', ORIGINAL_IMAGE_WINDOW,
                               partial(on_change, img, hsv_values))
    upper_value_bar = TrackBar(hsv_values, 'upper_value', 0, 255, 'Upper Value', ORIGINAL_IMAGE_WINDOW,
                               partial(on_change, img, hsv_values))
    while True:
        if cv.waitKey(1) & 0xFF == ord('s'):
            break
    cv.destroyAllWindows()
    segmented_hand = segment_hand(img, hsv_values)
    get_hand_attributes(segmented_hand)
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()