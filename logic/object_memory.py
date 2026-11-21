import time
import numpy as np


class LogicalObject:
    def __init__(self, logical_id, position):
        now = time.time()

        self.logical_id = logical_id

        # Spatial memory
        self.last_position = position
        self.prev_position = position
        self.last_seen_time = now

        # Person presence memory
        self.last_person_near_time = now

        # Stationary tracking
        self.stationary_since = None

        # State machine
        self.state = "ATTENDED"
        self.state_enter_time = now

        # Event handling
        self.event_triggered = False
        self.resolved = False

    def update_position(self, position):
        self.prev_position = self.last_position
        self.last_position = position
        self.last_seen_time = time.time()

    def movement_distance(self):
        return np.linalg.norm(
            np.array(self.last_position) - np.array(self.prev_position)
        )

    def reset_stationary(self):
        self.stationary_since = None


class ObjectMemory:
    def __init__(self, distance_threshold=60, time_threshold=2.0):
        self.objects = []
        self.distance_threshold = distance_threshold
        self.time_threshold = time_threshold
        self.next_id = 1

    def match_or_create(self, position):
        now = time.time()

        for obj in self.objects:
            dist = np.linalg.norm(
                np.array(position) - np.array(obj.last_position)
            )

            if dist < self.distance_threshold and (now - obj.last_seen_time) < self.time_threshold:
                obj.update_position(position)
                return obj

        new_obj = LogicalObject(self.next_id, position)
        self.objects.append(new_obj)
        self.next_id += 1
        return new_obj
