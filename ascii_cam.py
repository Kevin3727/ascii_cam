import cv2
import numpy as np


def get_frame():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
        return cv2.flip(frame, 1)
    return np.array()


def rbg_to_bw(frame):
    return (frame[:, :, 0] + frame[:, :, 1] + frame[:, :, 2]) / 3.


def resize(frame, size=24):
    y_size = size
    x_size = int(2 * size * frame.shape[1] / frame.shape[0])
    result = np.empty([y_size, x_size])

    x_depth = int(frame.shape[1] / x_size)
    y_depth = int(frame.shape[0] / y_size)

    for i in range(y_size):
        for j in range(x_size):
            result[i, j] = round(
                frame[(y_depth * i):(y_depth * (i + 1)),
                (x_depth * j):(x_depth * (j + 1))].mean())

    return result


def ascii_map(pixel):
    if pixel < 85:
        return '#'
    if pixel < 170:
        return '='
    return '.'


vect_map = np.vectorize(ascii_map)


def image_to_ascii(frame):
    return vect_map(frame)


def recalibrate(frame):
    max_ = frame.max()

    vect_ = np.vectorize(lambda x: 255 * x / max_)
    return vect_(frame)


def get_image():
    frame = get_frame()
    frame = rbg_to_bw(frame)
    frame = resize(frame)
    frame = recalibrate(frame)
    return frame


def print_image(frame):
    ascii_frame = image_to_ascii(frame)
    for row in ascii_frame:
        print(''.join(row))


if __name__ == '__main__':
    print_image(get_image())
