import argparse

import cv2
import numpy as np


def get_frame():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
        return cv2.flip(frame, 1)
    return np.array()


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


def resize_color(frame):
    recalibration_factor = 255 / frame.max()

    b_ = resize(frame[:, :, 0], size=24)
    vect_ = np.vectorize(lambda x: recalibration_factor * x)
    b_ = vect_(b_)

    g_ = resize(frame[:, :, 1], size=24)
    #     vect_ = np.vectorize(lambda x: 128 * x / g_.mean())
    g_ = vect_(g_)

    r_ = resize(frame[:, :, 2], size=24)
    #     vect_ = np.vectorize(lambda x: 128 * x / r_.mean())
    r_ = vect_(r_)

    #     vect_ = np.vectorize(lambda x: 128 * x / frame.mean())
    #     return vect_(r_), vect_(b_), vect_(g_)
    return r_, g_, b_


def run(background=False, black=False):
    frame = get_frame()
    r_, g_, b_ = resize_color(frame)

    result = np.empty_like(r_, dtype=object)

    if background:
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = "\033[48;2;" + str(int(r_[i, j])) + ';' + str(
                    int(g_[i, j])) + ';' + str(int(b_[i, j])) + "m \033[0m"
    elif black:
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = "\033[48;2;0;0;0m\033[38;2;" + str(
                    int(r_[i, j])) + ';' + str(
                    int(g_[i, j])) + ';' + str(int(b_[i, j])) + "m#\033[0m"
    else:
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = "\033[38;2;" + str(
                    int(r_[i, j])) + ';' + str(
                    int(g_[i, j])) + ';' + str(int(b_[i, j])) + "m#\033[0m"

    for row in result:
        for item in row:
            print(item, end='')
        print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""ASCII cam""")

    parser.add_argument('-bg', '--background',
                        action='store_true',
                        default=False)

    parser.add_argument('-b', '--black',
                        action='store_true',
                        default=False)

    parser.add_argument('-m', type=int)

    args = parser.parse_args()

    if args.m:
        for _ in range(args.m):
            run(background=args.background,
                black=args.black)
    else:
        run(background=args.background,
            black=args.black)
