import pickle
from typing import Any, List, Literal, Tuple, TypedDict
import cv2
import cv2.barcode
import numpy as np
from utils import Vec, Line
from pyinputplus import inputFilename


class SerializedTrack(TypedDict):
    segments: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    width: int
    height: int
    size: int
    curves: List[
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    ]


class Track:
    state: Literal[
        "choosing_start",
        "choosing_goal",
        "drawing_line",
        "idle",
        "calculating",
        "solved",
    ]
    start: Vec
    goal: Vec
    explored_states: List[
        Tuple[
            Tuple[int, int],
            Tuple[int, int],
        ]
    ]
    state_before: List[int]
    last_explored_state: int

    def __init__(self, serialized: SerializedTrack):
        self.window_name = "Track Solver"
        self.width, self.height = serialized["width"], serialized["height"]
        self.size = serialized["size"]
        self.segments = [
            Line(Vec(*segment[0]), Vec(*segment[1]))
            for segment in serialized["segments"]
        ]
        self.state = "choosing_start"
        self.start = None
        self.goal = None

        self.explored_states = []
        self.state_before = []

        self.last_explored_state = 0

        self.background = np.zeros(
            ((self.height - 1) * self.size + 1, (self.width - 1) * self.size + 1, 3),
            np.uint8,
        )
        self.update_background()

    def update_background(self):
        self.background = np.zeros(
            ((self.height - 1) * self.size + 1, (self.width - 1) * self.size + 1, 3),
            np.uint8,
        )
        for row in range(self.height):
            for col in range(self.width):
                self.background = cv2.circle(
                    self.background,
                    (col * self.size, row * self.size),
                    2,
                    (64, 64, 64),
                    -1,
                )
        for line in self.segments:
            self.background = line.draw(self.background, scale=self.size)

        for state, last_state_id in zip(self.explored_states, self.state_before):
            self.background = Line(
                Vec(*state[0]), Vec(*self.explored_states[last_state_id][1])
            ).draw(self.background, scale=self.size, color=(63, 63, 63))

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state == "choosing_start":
                self.start = round(Vec(x / self.size, y / self.size))
                self.state = "choosing_goal"
            elif self.state == "choosing_goal":
                self.goal = round(Vec(x / self.size, y / self.size))
                self.explored_states.append((self.start.xy, (0, 0)))
                self.state_before.append(0)
                self.state = "idle"
            elif self.state == "drawing_line":
                self.segments.append(self.drawing_line)
                self.update_background()
                self.drawing_line = None
                self.state = (
                    "choosing_start"
                    if self.start is None
                    else ("choosing_goal" if self.goal is None else "idle")
                )
        if event == cv2.EVENT_RBUTTONDOWN:
            if self.state in ("choosing_start", "choosing_goal", "idle"):
                self.drawing_line = Line(
                    Vec(x / self.size, y / self.size), Vec(x / self.size, y / self.size)
                )
                self.state = "drawing_line"
        if event == cv2.EVENT_MOUSEMOVE:
            if self.state == "drawing_line":
                self.drawing_line.vec2 = Vec(x / self.size, y / self.size)

    def add_states(self):
        if (
            self.state != "solved"
            and len(self.explored_states) - 1 >= self.last_explored_state
        ):
            last_state = self.explored_states[self.last_explored_state]
            x, y = last_state[0]
            vx, vy = last_state[1]

            for dvx, dvy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                new_state = ((x + vx + dvx, y + vy + dvy), (vx + dvx, vy + dvy))
                intersects = False
                for line in self.segments:
                    intersects = intersects or line.intersects(
                        Line(Vec(*last_state[0]), Vec(*new_state[0]))
                    )
                if new_state not in self.explored_states and not intersects:
                    self.background = Line(
                        Vec(*new_state[0]), Vec(*last_state[0])
                    ).draw(self.background, scale=self.size, color=(63, 63, 63))
                    self.explored_states.append(new_state)
                    self.state_before.append(self.last_explored_state)
                    if new_state[0] == self.goal.xy:
                        self.state = "solved"
                        return

            self.last_explored_state += 1

    def loop(self):
        cv2.imshow(self.window_name, self.background)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.mouse_event)
        while 1:
            frame = self.background.copy()

            if self.state == "drawing_line":
                frame = self.drawing_line.draw(frame, scale=self.size)

            if self.start is not None:
                frame = cv2.circle(
                    frame, (self.size * self.start).xy, 3, (255, 127, 127), thickness=-1
                )

            if self.goal is not None:
                frame = cv2.circle(
                    frame, (self.size * self.goal).xy, 3, (127, 127, 255), thickness=-1
                )

            if self.state == "calculating":
                for _ in range(10):
                    self.add_states()

            if self.state == "solved":
                p = self.goal
                i = len(self.explored_states) - 1
                while p != self.start:
                    new_p = Vec(*self.explored_states[i][0])
                    frame = Line(p, new_p).draw(frame, (0, 255, 0), scale=self.size)
                    i = self.state_before[i]
                    p = new_p

            # frame = cv2.putText(
            #     frame,
            #     self.state,
            #     (50, 50),
            #     cv2.FONT_HERSHEY_COMPLEX,
            #     1,
            #     (255, 255, 255),
            # )

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

            if key == 32 and self.state == "idle":
                self.state = "calculating"

            # if key != -1:
            #     print(key)

        cv2.setMouseCallback(self.window_name, lambda *_: None)


with open(
    inputFilename("What track should be solved? ") + ".pickle",
    "rb",
) as file:
    track = Track(pickle.load(file))
track.loop()
