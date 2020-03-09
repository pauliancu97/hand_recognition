import cv2 as cv
import numpy as np
import easygui
import math
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

CANNY_FIRST_THRESHOLD_TRACK_BAR_NAME = 'Canny First Threshold'
CANNY_SECOND_THRESHOLD_TRACK_BAR_NAME = 'Canny Second Threshold'
THRESHOLD_BLACK_TRACK_BAR_NAME = 'Th Black'
N_TRACK_BAR_NAME = 'N'
THRESHOLD_TRACK_BAR_NAME = 'Threshold'
WINDOW_NAME = 'Control'
SALIENT_EDGES_WINDOW_NAME = 'Salient Edges'
FINGER_WINDOW_NAME = 'Fingers'
RADIUS_PALM_TRACK_BAR_NAME = 'Radius Palm'
SEGMENTED_HAND_WINDOW = 'Segmented Hand'
SEGMENTATION_VALUES_WINDOW = 'Segmentation Values'
LOWER_HUE_VALUE_TRACK_BAR = 'Lower Hue Value'
UPPER_HUE_VALUE_TRACK_BAR = 'Upper Hue Value'
LOWER_SATURATION_VALUE_TRACK_BAR = 'Lower Saturation Value'
UPPER_SATURATION_VALUE_TRACK_BAR = 'Upper Saturation Value'


class HandValues:

    def __init__(self, finger_center, hand_center, hand_orientation, wrist_center, p_array, finger_tips_angles,
                 finger_tips):
        self.finger_center = finger_center
        self.hand_center = hand_center
        self.hand_orientation = hand_orientation
        self.wrist_center = (int(wrist_center[0]), int(wrist_center[1]))
        self.p_array = p_array
        self.finger_tips_angles = finger_tips_angles
        self.finger_tips = finger_tips


class Values:

    def __init__(self, image, canny_first_threshold, canny_second_threshold, black_threshold, n, threshold, radius_palm):
        self.image = image
        self.canny_first_threshold = canny_first_threshold
        self.canny_second_threshold = canny_second_threshold
        self.black_threshold = black_threshold
        self.n = n
        self.threshold = threshold
        self.radius_palm = radius_palm


class TrackBar:

    def __init__(self, track_bar_name, window_name, track_bar_value_name, track_bar_value, count, values_object):
        self.track_bar_name = track_bar_name
        self.window_name = window_name
        self.track_bar_value_name = track_bar_value_name
        self.track_bar_value = track_bar_value
        self.values_object = values_object
        self.count = count

    def init(self):
        cv.createTrackbar(self.track_bar_name, self.window_name, self.track_bar_value, self.count, self.on_change)

    def on_change(self, value):
        setattr(self.values_object, self.track_bar_value_name, value)
        image = np.copy(self.values_object.image)
        canny_first_threshold = self.values_object.canny_first_threshold
        canny_second_threshold = self.values_object.canny_second_threshold
        black_threshold = self.values_object.black_threshold
        n = self.values_object.n
        threshold = self.values_object.threshold
        radius_palm = self.values_object.radius_palm
        salient_edges = get_salient_edges(image, canny_first_threshold, canny_second_threshold, black_threshold)
        cv.imshow(SALIENT_EDGES_WINDOW_NAME, salient_edges)
        finger_image = get_finger_image(image, canny_first_threshold, canny_second_threshold, black_threshold, n,
                                        threshold)
        cv.normalize(finger_image, finger_image, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
        cv.imshow(FINGER_WINDOW_NAME, finger_image)
        hand_values = get_hand_values(finger_image, image, radius_palm)
        circle_mask = np.zeros(image.shape, dtype=image.dtype)
        cv.circle(circle_mask, hand_values.finger_center, int(2.5 * radius_palm), 255, -1)
        image = cv.bitwise_and(circle_mask, image)
        color_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        cv.circle(color_image, hand_values.finger_center, 3, (0, 255, 0))
        cv.circle(color_image, hand_values.hand_center, 3, (0, 0, 255))
        cv.circle(color_image, hand_values.wrist_center, 3, (255, 0, 0))
        cv.imshow('centers', color_image)


class Plotter:

    def __init__(self, values_object):
        self.values_object = values_object

    def plot(self):
        image = np.copy(self.values_object.image)
        canny_first_threshold = self.values_object.canny_first_threshold
        canny_second_threshold = self.values_object.canny_second_threshold
        black_threshold = self.values_object.black_threshold
        n = self.values_object.n
        threshold = self.values_object.threshold
        radius_palm = self.values_object.radius_palm
        finger_image = get_finger_image(image, canny_first_threshold, canny_second_threshold, black_threshold, n,
                                        threshold)
        cv.normalize(finger_image, finger_image, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
        hand_values = get_hand_values(finger_image, image, radius_palm)
        hand_center = hand_values.hand_center
        wrist_center = hand_values.wrist_center
        rotation_matrix = get_rotation_matrix(hand_center, wrist_center)
        finger_image = cv.warpAffine(finger_image, rotation_matrix, (finger_image.shape[0], finger_image.shape[1]))
        for row in range(wrist_center[1], finger_image.shape[0]):
            for col in range(0, finger_image.shape[1]):
                finger_image[row, col] = 0
        color_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        color_image = cv.warpAffine(color_image, rotation_matrix, (color_image.shape[0], color_image.shape[1]))
        for finger_tip in hand_values.finger_tips:
            cv.circle(color_image, (int(finger_tip[0]), int(finger_tip[1])), 3, (255, 0, 0,), 1)
        cv.imshow('finger tips', color_image)
        cv.imshow('rotated', finger_image)
        plt.plot(hand_values.p_array)
        plt.show()


class SegmentationValues:

    def __init__(self):
        self.lower_hue = 0
        self.upper_hue = 255
        self.lower_saturation = 0
        self.upper_saturation = 255


class SegmentationValueTrackBar:

    def __init__(self, window_name, track_bar_name, value_object, prop, image, value, count):
        self.window_name = window_name
        self.track_bar_name = track_bar_name
        self.value_object = value_object
        self.prop = prop
        self.image = image
        self.value = value
        self.count = count

    def init(self):
        cv.createTrackbar(self.track_bar_name, self.window_name, self.value, self.count, self.on_change)

    def on_change(self, value):
        setattr(self.value_object, self.prop, value)
        lower_hue = self.value_object.lower_hue
        upper_hue = self.value_object.upper_hue
        lower_saturation = self.value_object.lower_saturation
        upper_saturation = self.value_object.upper_saturation
        lower = (lower_hue, lower_saturation, 0)
        upper = (upper_hue, upper_saturation, 255)
        image_hsv = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
        image_gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        segmented = cv.inRange(image_hsv, lower, upper)
        image_selected = cv.bitwise_and(image_gray, segmented)
        cv.imshow('crop', image_selected)
        cv.imshow(SEGMENTED_HAND_WINDOW, segmented)


def get_rotation_angle(palm_point, middle_wrist_point):
    x = palm_point[0] - middle_wrist_point[0]
    y = palm_point[1] - middle_wrist_point[1]
    hand_angle = math.atan2(y, x)
    rotation_angle = math.degrees(hand_angle + math.pi / 2)
    return rotation_angle


def get_rotation_matrix(palm_point, middle_wrist_point):
    rotation_angle = get_rotation_angle(palm_point, middle_wrist_point)
    rotation_matrix = cv.getRotationMatrix2D(palm_point, rotation_angle, 1)
    return rotation_matrix


def transform_point(point, angle, center):
    angle = 2 * math.pi - angle * math.pi / 180.0
    point = np.array([point[0], point[1], 1.0], dtype=np.float32)
    translation_matrix = np.array([[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [-center[0], -center[1], 1.0]], dtype=np.float32)
    rotation_matrix = np.array([[math.cos(angle), math.sin(angle), 0.0],
                                [-math.sin(angle), math.cos(angle), 0.0],
                                [0.0, 0.0, 1.0]], dtype=np.float32)
    inverse_translation_matrix = np.array([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [center[0], center[1], 1.0]], dtype=np.float32)
    point = point.dot(translation_matrix)
    point = point.dot(rotation_matrix)
    point = point.dot(inverse_translation_matrix)
    return np.array((point[0], point[1]), dtype=np.float32)


def get_center_of_mass(image):
    moments = cv.moments(image)
    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])
    return x, y


def find_local_max(array, width, threshold):
    local_max_list = []
    for index in range(width, array.shape[0] - width):
        is_max = True
        average = 0.0
        for position in range(index - width, index + width + 1):
            if array[position] > array[index]:
                is_max = False
            average += array[position]
        average = average / (2 * width + 1)
        if is_max and array[index] >= (average + threshold):
            local_max_list.append((index, array[index][0]))
    return local_max_list


def get_angle(first_point, second_point):
    x_diff = second_point[0] - first_point[0]
    y_diff = second_point[1] - first_point[1]
    angle = math.degrees(math.atan2(y_diff, x_diff))
    if angle < 0.0:
        angle += 360
    return angle


def get_hand_values(finger_image, hand_image, radius_palm):
    finger_center = get_center_of_mass(finger_image)
    circle_mask = np.zeros(hand_image.shape, dtype=hand_image.dtype)
    cv.circle(circle_mask, finger_center, int(2.5 * radius_palm), 255, -1)
    hand_image = cv.GaussianBlur(hand_image, (3, 3), 0)
    _, binary_hand_image = cv.threshold(hand_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    extracted_hand_image = cv.bitwise_and(binary_hand_image, circle_mask)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    extracted_hand_image = cv.erode(extracted_hand_image, kernel)
    hand_center = get_center_of_mass(extracted_hand_image)
    hand_orientation = np.array((finger_center[0] - hand_center[0], finger_center[1] - hand_center[1]),
                                dtype=np.float32)
    length = (finger_center[0] - hand_center[0]) * (finger_center[0] - hand_center[0]) \
             + (finger_center[1] - hand_center[1]) * (finger_center[1] - hand_center[1])
    length = math.sqrt(length)
    hand_orientation = hand_orientation * (1 / length)
    wrist_center = np.array(hand_center, dtype=np.float32) - radius_palm * hand_orientation
    rotation_matrix = get_rotation_matrix(hand_center, wrist_center)
    finger_image = cv.warpAffine(finger_image, rotation_matrix, (finger_image.shape[0], finger_image.shape[1]))
    binary_hand_image = cv.warpAffine(binary_hand_image, rotation_matrix, (binary_hand_image.shape[0], binary_hand_image.shape[1]))
    for row in range(int(wrist_center[1]), finger_image.shape[0]):
        for col in range(0, finger_image.shape[1]):
            finger_image[row, col] = 0
    first_radius = 0.5 * radius_palm
    second_radius = 2.5 * radius_palm
    p_matrix = np.zeros((361, int(second_radius - first_radius + 1)), dtype=np.int32)
    for row in range(0, finger_image.shape[0]):
        for col in range(0, finger_image.shape[1]):
            if finger_image[row, col] != 0:
                length = (col - wrist_center[0]) * (col - wrist_center[0]) \
                         + (row - wrist_center[1]) * (row - wrist_center[1])
                length = math.sqrt(length)
                if (length >= first_radius) and (length <= second_radius):
                    x_diff = col - wrist_center[0]
                    y_diff = row - wrist_center[1]
                    angle = math.degrees(math.atan2(y_diff, x_diff))
                    if angle < 0.0:
                        angle += 360
                    p_matrix_row = int(angle)
                    p_matrix_col = int(length - first_radius)
                    p_matrix[p_matrix_row, p_matrix_col] += 1
    p_array = np.zeros(361, dtype=np.float32)
    global_sum = 0.0
    for row in range(0, 361):
        accumulator = 0.0
        for col in range(0, int(second_radius - first_radius + 1)):
            accumulator += col * p_matrix[row, col]
        p_array[row] = accumulator
        global_sum += accumulator
    p_array = p_array * (1.0 / global_sum)
    p_array = gaussian_filter1d(p_array, 3)
    local_max, _ = find_peaks(p_array)
    local_max = local_max[:5]
    finger_tips_angles = local_max
    contour = []
    for row in range(0, binary_hand_image.shape[0]):
        for col in range(0, binary_hand_image.shape[1]):
            if binary_hand_image[row, col] != 0:
                contour.append(np.array([col, row], dtype=np.float32))
    get_distance = lambda point: math.sqrt((point[0] - wrist_center[0]) * (point[0] - wrist_center[0]) +
                                           (point[1] - wrist_center[1]) * (point[1] - wrist_center[1]))
    finger_tips = []
    for finger_tip_angle in finger_tips_angles:
        potential_finger_tips = [point for point in contour]
        potential_finger_tips = [point for point in potential_finger_tips
                                 if int(get_angle(wrist_center, point)) == finger_tip_angle]
        finger_tip = max(potential_finger_tips, key=lambda point: get_distance(point))
        finger_tips.append(finger_tip)
    return HandValues(finger_center, hand_center, hand_orientation, wrist_center, p_array, finger_tips_angles,
                      finger_tips)


def get_salient_edges(gray_image, canny_first_threshold, canny_second_threshold, black_threshold):
    canny = cv.Canny(gray_image, canny_first_threshold, canny_second_threshold)
    contours, _ = cv.findContours(gray_image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    contours_image = np.zeros(gray_image.shape, dtype=gray_image.dtype)
    contours = contours[0]
    for point in contours:
        point = point[0]
        contours_image[point[1], point[0]] = gray_image[point[1], point[0]]
    _, black_image = cv.threshold(gray_image, black_threshold, 255, cv.THRESH_BINARY_INV)
    and_image = cv.bitwise_and(canny, black_image)
    salient_edges = cv.bitwise_or(contours_image, and_image)
    for col in range(0, salient_edges.shape[1]):
        salient_edges[1, col] = 0
        salient_edges[salient_edges.shape[0] - 2, col] = 0
    return salient_edges


def get_convolution_operator(n):
    first_radius = n
    second_radius = 2 * n
    third_radius = 3 * n
    operator = np.zeros((6 * n + 1, 6 * n + 1), dtype=np.int32)
    for row in range(0, 6 * n + 1):
        for col in range(0, 6 * n + 1):
            dist = math.sqrt((col - 3 * n) * (col - 3 * n) + (row - 3 * n) * (row - 3 * n))
            if dist <= first_radius:
                operator[row, col] = -1
            elif (dist > second_radius) and (dist <= third_radius):
                operator[row, col] = 1
            else:
                operator[row, col] = 0
    return operator


def get_finger_image(gray_image, canny_first_threshold, canny_second_threshold, black_threshold, n, threshold):
    salient_edges = get_salient_edges(gray_image, canny_first_threshold, canny_second_threshold, black_threshold)
    convolution_operator = get_convolution_operator(n)
    salient_edges_normalized = np.zeros(salient_edges.shape, dtype=np.float32)
    for row in range(0, salient_edges.shape[0]):
        for col in range(0, salient_edges.shape[1]):
            salient_edges_normalized[row, col] = salient_edges[row, col] / 255.0
    salient_edges = salient_edges_normalized
    convolution_operator = convolution_operator.astype(np.float32)
    convolution_operator = cv.flip(convolution_operator, -1)
    convolution = cv.filter2D(salient_edges, -1, convolution_operator)
    _, convolution = cv.threshold(convolution, threshold, 1.0, cv.THRESH_BINARY)
    _, binary_hand = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    for row in range(0, convolution.shape[0]):
        for col in range(0, convolution.shape[1]):
            if binary_hand[row, col] == 0:
                convolution[row, col] = 0
    return convolution


def segment_hand(image):
    thresh = image
    np.savetxt('segmented.txt', thresh)
    num_of_labels, labels, stats, _ = cv.connectedComponentsWithStats(thresh, 4, cv.CV_32S)
    hand_label = max([(label, stats[label, cv.CC_STAT_AREA]) for label in range(0, num_of_labels)], key=lambda x: x[1])[0]
    hand_label += 1
    hand_image = np.zeros(labels.shape, dtype=np.uint8)
    for row in range(0, hand_image.shape[0]):
        for col in range(0, hand_image.shape[1]):
            if labels[row, col] == hand_label:
                hand_image[row, col] = 255
    return hand_image


def main():
    path = easygui.fileopenbox()
    image = cv.imread(path, cv.IMREAD_COLOR)
    segmentation_values = SegmentationValues()
    cv.namedWindow(SEGMENTATION_VALUES_WINDOW)
    lower_hue_segmentation_value_track_bar = SegmentationValueTrackBar(SEGMENTATION_VALUES_WINDOW,
                                                                       LOWER_HUE_VALUE_TRACK_BAR, segmentation_values,
                                                                       'lower_hue', image, 0, 180)
    lower_hue_segmentation_value_track_bar.init()
    upper_hue_segmentation_value_track_bar = SegmentationValueTrackBar(SEGMENTATION_VALUES_WINDOW,
                                                                       UPPER_HUE_VALUE_TRACK_BAR, segmentation_values,
                                                                       'upper_hue', image, 180, 180)
    upper_hue_segmentation_value_track_bar.init()
    lower_sat_segmentation_value_track_bar = SegmentationValueTrackBar(SEGMENTATION_VALUES_WINDOW,
                                                                       LOWER_SATURATION_VALUE_TRACK_BAR, segmentation_values,
                                                                       'lower_saturation', image, 0, 255)
    lower_sat_segmentation_value_track_bar.init()
    upper_sat_segmentation_value_track_bar = SegmentationValueTrackBar(SEGMENTATION_VALUES_WINDOW,
                                                                       UPPER_SATURATION_VALUE_TRACK_BAR, segmentation_values,
                                                                       'upper_saturation', image, 255, 255)
    upper_sat_segmentation_value_track_bar.init()
    while True:
        if cv.waitKey(1) & 0xFF == ord('s'):
            break
    lower = (segmentation_values.lower_hue, segmentation_values.lower_saturation, 0)
    upper = (segmentation_values.upper_hue, segmentation_values.upper_saturation, 255)
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image_gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hand_gray_image = cv.inRange(image_hsv, lower, upper)
    hand_gray_image = segment_hand(hand_gray_image)
    hand_gray_image = cv.bitwise_and(image_gray_scale, hand_gray_image)
    hand_gray_image = cv.blur(hand_gray_image, (3, 3))
    cv.imshow('segmented hand', hand_gray_image)
    cv.namedWindow(WINDOW_NAME)
    values = Values(hand_gray_image, 100, 255, 150, 1, 6, 30)
    TrackBar(CANNY_FIRST_THRESHOLD_TRACK_BAR_NAME, WINDOW_NAME, 'canny_first_threshold', 100, 255, values).init()
    TrackBar(CANNY_SECOND_THRESHOLD_TRACK_BAR_NAME, WINDOW_NAME, 'canny_second_threshold', 255, 255, values).init()
    TrackBar(THRESHOLD_BLACK_TRACK_BAR_NAME, WINDOW_NAME, 'black_threshold', 150, 255, values).init()
    TrackBar(N_TRACK_BAR_NAME, WINDOW_NAME, 'n', 1, 30, values).init()
    TrackBar(THRESHOLD_TRACK_BAR_NAME, WINDOW_NAME, 'threshold', 6, 50, values).init()
    TrackBar(THRESHOLD_TRACK_BAR_NAME, WINDOW_NAME, 'threshold', 6, 50, values).init()
    TrackBar(RADIUS_PALM_TRACK_BAR_NAME, WINDOW_NAME, 'radius_palm', 30, 200, values).init()
    salient_edges = get_salient_edges(hand_gray_image, 100, 255, 150)
    finger_image = get_finger_image(hand_gray_image, 100, 255, 150, 1, 6)
    cv.normalize(finger_image, finger_image, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    cv.imshow(SALIENT_EDGES_WINDOW_NAME, salient_edges)
    cv.imshow(FINGER_WINDOW_NAME, finger_image)
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv.waitKey(1) & 0xFF == ord('p'):
            plotter = Plotter(values)
            plotter.plot()


if __name__ == '__main__':
    main()