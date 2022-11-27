#!/usr/bin/env python
import sys
import os
import argparse
import cv2
import numpy as np

ENABLE_INTROSPECTION = False
KERNEL_SIZE = 11

DRAW_BUFFER = None
INFO_BUFFER = []

class CursorPosition:
    def __init__(self):
        self.x = 0
        self.y = 0

CURSOR = CursorPosition()

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='cv_viewer',
        description='This simple CLI tool allows one to introspect an image')
    parser.add_argument('image_file', help='Image file to open/introspect')
    parser.add_argument('-I', '--input-enhancer', default='')
    return parser.parse_args()

class PutTextAdaptor:
    def __init__(self, img, start):
        self._img = img
        self._start = start
        self._offset = 0

    def __call__(self, text):
        pos = (self._start[0], self._start[1] + int(self._offset * 30))
        self._offset += 1

        cv2.putText(self._img,
            text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA)

    def add_blank(self, offset):
        self._offset += offset

def create_text_window(redraw=False):
    window = 'cv_viewer help'
    if redraw:
        cv2.destroyWindow(window)
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(window, 640, 480)
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    putText = PutTextAdaptor(img, (5, 25))
    putText('This is a little help')
    putText('ESC : closes the viewer')
    putText('h : open this help window')
    putText('s : show/hide kernel at cursor (now: {k}x{k})'.format(k=KERNEL_SIZE))
    putText('+ : increases kernel size under cursor')
    putText('- : decreases kernel size under cursor')
    putText.add_blank(2)
    putText('Cool features:')
    putText('i : open info window')
    putText('r : reload user code (outputs into info window)')
    cv2.imshow(window, img)

def create_info_window(redraw=False):
    window = 'cv_viewer info'
    if redraw:
        cv2.destroyWindow(window)
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(window, 640, 480)
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    putText = PutTextAdaptor(img, (5, 25))
    putText('User code output:')
    global INFO_BUFFER
    for line in INFO_BUFFER:
        putText(line)

    cv2.imshow(window, img)

def introspection_main(event, x, y, flags, img):
    global ENABLE_INTROSPECTION
    global DRAW_BUFFER
    if not ENABLE_INTROSPECTION:
        return

    # clean up the draw buffer first to avoid previous frame artifacts
    DRAW_BUFFER = img.copy()

    half = int(KERNEL_SIZE / 2)
    cv2.rectangle(DRAW_BUFFER,
        (x - half, y - half), (x + half, y + half),
        (0, 0, 255), 2)

    global CURSOR
    CURSOR.x = x
    CURSOR.y = y

class InfoOutput:
    def __init__(self):
        self._buf = []

    def commit(self):
        global INFO_BUFFER
        import copy
        INFO_BUFFER = copy.deepcopy(self._buf)
        self._buf.clear()

    def add(self, *objects, sep=' '):
        import io
        output = io.StringIO()
        print(*objects, file=output, end='')
        line = output.getvalue()
        output.close()
        self._buf.append(line)

def compute_statistics_on_image(img, output_buffer: InfoOutput):
    def clip(x, dim):
        return np.clip(x, 0, img.shape[dim])

    global CURSOR
    x = CURSOR.x
    y = CURSOR.y
    global KERNEL_SIZE
    half = int(KERNEL_SIZE / 2)

    y_f, y_l = clip(y - half, 0), clip(y + half + 1, 0)
    x_f, x_l = clip(x - half, 1), clip(x + half + 1, 1)
    roi = img[y_f:y_l, x_f:x_l]

    # mean value:
    mean = np.mean(roi, axis=(0, 1))
    mean = [int(e) for e in mean]
    output_buffer.add('Window mean:', mean)


def handle_key(key, user_code_path):
    global ENABLE_INTROSPECTION
    global KERNEL_SIZE

    if key == ord('s'):
        ENABLE_INTROSPECTION = not ENABLE_INTROSPECTION

    if key == ord('+'):
        size = KERNEL_SIZE
        KERNEL_SIZE = np.clip(size + 2, 3, 151)
        create_text_window()
    if key == ord('-'):
        size = KERNEL_SIZE
        KERNEL_SIZE = np.clip(size - 2, 3, 151)
        create_text_window()

    if key == ord('h'):
        create_text_window(redraw=True)

    if key == ord('i'):
        create_info_window(redraw=True)

    # if key == ord('r') and user_code_path:
    #     import importlib
    #     importlib.reload()


def main():
    global ENABLE_INTROSPECTION
    global DRAW_BUFFER

    args = parse_arguments()

    original = cv2.imread(args.image_file)
    # Note: .copy() changes the data pointer! (so we need global variable)
    DRAW_BUFFER = original.copy()

    window = 'cv_viewer'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1920, 1080)
    cv2.setMouseCallback(window, introspection_main, original)

    create_text_window()

    while True:
        cv2.imshow(window, DRAW_BUFFER)
        key = cv2.waitKey(30)
        if key == 27: # ESC
            break
        handle_key(key, args.input_enhancer)
        if not ENABLE_INTROSPECTION:
            DRAW_BUFFER = original.copy()
        else:
            adaptor = InfoOutput()
            compute_statistics_on_image(original, adaptor)
            adaptor.commit()
            create_info_window()


if __name__ == '__main__':
    main()
    sys.exit(0)
