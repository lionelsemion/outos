import pickle
from typing import Dict, List, Tuple, Union
import numpy.typing as npt
import cv2
import numpy as np
from utils import Vec, Line
from pyinputplus import inputInt, inputFilename


class Track:
    _segments: List[Line]
    width: int
    height: int
    size: int
    window_name: str
    curves: List[Tuple[Tuple[Vec, Vec, Vec, Vec], List[Line]]]
    incomplete_curve: Tuple[List[Vec], List[bool]]
    first_curve: int
    changed: bool
    image: npt.NDArray[np.uint8]
    background: npt.NDArray[np.uint8]

    @property
    def segments(self):
        segs = self._segments
        for points, lines in self.curves:
            segs += lines
        return segs

    def __init__(self, width, height, size, segments=None) -> None:
        self.window_name = "yay"

        cv2.namedWindow(self.window_name)

        self.changed = True
        self.curves = []
        self.incomplete_curve = ([None, None, None, None], [False, False, False, False])

        self.first_curve = 0

        self.width, self.height, self.size = width, height, size
        self._segments = [] if segments is None else segments

        img = np.zeros(
            ((self.height - 1) * self.size + 1, (self.width - 1) * self.size + 1, 3),
            np.uint8,
        )
        for row in range(height):
            for col in range(width):
                img = cv2.circle(img, (col * size, row * size), 2, (64, 64, 64), -1)
        self.background = img

    def draw(self, editing=False):
        if self.changed:
            img = np.copy(self.background)

            for line in self.segments:
                img = line.draw(img, scale=size)

            self.image = np.copy(img)
        else:
            img = np.copy(self.image)

        self.changed = False
        if editing:
            control_color = (64, 64, 64)
            for points, lines in self.curves:
                img = Line(points[0], points[2]).draw(
                    img, color=control_color, scale=size
                )
                img = Line(points[1], points[3]).draw(
                    img, color=control_color, scale=size
                )

            if (
                self.incomplete_curve[0][0] is not None
                and self.incomplete_curve[0][1] is not None
            ):
                p0, p3 = self.incomplete_curve[0][0:2]
                p1 = (
                    self.incomplete_curve[0][2]
                    if self.incomplete_curve[0][2] is not None
                    else p0
                )
                p2 = (
                    self.incomplete_curve[0][3]
                    if self.incomplete_curve[0][3] is not None
                    else p3
                )

                img = Line(p0, p1).draw(img, color=control_color, scale=size)
                img = Line(p3, p2).draw(img, color=control_color, scale=size)

                for line in self.bezier(p0, p3, p1, p2):
                    img = line.draw(img, color=(255, 128, 128), scale=size)

        return img

    def show(self, editing=False):
        cv2.imshow(self.window_name, self.draw(editing=editing))

    def new_curve(self, points: Tuple[Vec, Vec, Vec, Vec]):
        self.curves.append(
            (
                tuple(points),
                self.bezier(*points),
            )
        )
        self.changed = True

    def bezier(self, p0: Vec, p3: Vec, p1: Vec, p2: Vec, segments=None):
        if segments is None:
            chord = (p3 - p0).rho
            cont_net = (p0 - p1).rho + (p2 - p1).rho + (p3 - p2).rho

            segments = max(int((cont_net + chord) / 2), 1)

        p0 = p0.xy
        p1 = p1.xy
        p2 = p2.xy
        p3 = p3.xy

        t = np.arange(1, segments + 1) / segments
        t = np.stack((t, t), axis=1)

        q0 = p0 * (1 - t) + p1 * t
        q1 = p1 * (1 - t) + p2 * t
        q2 = p2 * (1 - t) + p3 * t
        r0 = q0 * (1 - t) + q1 * t
        r1 = q1 * (1 - t) + q2 * t
        points = r0 * (1 - t) + r1 * t

        segs = []
        last_point = p0
        for new_point in points:
            segs.append(Line(Vec(*last_point), Vec(*new_point)))
            last_point = new_point

        return segs

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.incomplete_curve = (
                [None, None, None, None],
                [False, False, False, False],
            )
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.incomplete_curve[1][self.incomplete_curve[1].index(False)] = True
            if False not in self.incomplete_curve[1]:
                self.new_curve(self.incomplete_curve[0])
                self.incomplete_curve = (
                    [None, None, None, None],
                    [False, False, False, False],
                )
            if True not in self.incomplete_curve[1]:
                if len(self.curves) > 0:
                    last_curve = self.curves[-1][0]
                    self.incomplete_curve = (
                        [
                            last_curve[1],
                            None,
                            2 * last_curve[1] - last_curve[3],
                            None,
                        ],
                        [True, False, True, False],
                    )
        elif event == cv2.EVENT_MOUSEMOVE:
            self.incomplete_curve[0][self.incomplete_curve[1].index(False)] = Vec(
                x / self.size, y / self.size
            )

    def edit_mode(self):
        self.show(True)
        cv2.setMouseCallback(self.window_name, self.mouse_event)
        while key := cv2.waitKey(1):  # (key := cv2.waitKey(1)): # != 32:  # 113:#q
            self.show(True)
            if key == 115:  # s
                try:
                    with open(
                        name := inputFilename(
                            "How do you want your track to be called? "
                        )
                        + ".pickle",
                        "xb",
                    ) as f:
                        pickle.dump(track.export(), f)
                        print(f"Track saved as {name}")
                except FileExistsError:
                    print("File already exists.")
            if key == 106:  # j
                if len(self.curves) > self.first_curve:
                    first_curve = self.curves[self.first_curve][0]
                    last_curve = self.curves[-1][0]
                    join_curve = (
                        last_curve[1],
                        first_curve[0],
                        2 * last_curve[1] - last_curve[3],
                        2 * first_curve[0] - first_curve[2],
                    )
                    self.new_curve(join_curve)
                    self.incomplete_curve = (
                        [None, None, None, None],
                        [False, False, False, False],
                    )
                    self.first_curve = len(self.curves)
        cv2.setMouseCallback(self.window_name, lambda *args: None)

    def export(self):
        return {
            "segments": [(line.vec1.xy, line.vec2.xy) for line in self.segments],
            "width": self.width,
            "height": self.height,
            "size": self.size,
            "curves": [tuple(p.xy for p in curve[0]) for curve in self.curves],
        }


if __name__ == "__main__":
    width = 60  # inputInt("Width? ", min=1, default=60, limit=1)
    height = 40  # inputInt("Height? ", min=1, default=40, limit=1)
    size = int(min(1200 / width, 800 / height))

    track = Track(width, height, size)

    track.edit_mode()

    track.show()
    cv2.waitKey()
