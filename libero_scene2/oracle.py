#!/usr/bin/env python3
"""Privileged closed-loop pick-and-place oracle for LIBERO Scene 2."""

from __future__ import annotations

import numpy as np


def contacts(inner, object_name: str) -> tuple[bool, bool]:
    gripper = inner.robots[0].gripper
    left = gripper.important_geoms["left_fingerpad"]
    right = gripper.important_geoms["right_fingerpad"]
    left = [left] if isinstance(left, str) else list(left)
    right = [right] if isinstance(right, str) else list(right)
    object_geoms = list(inner.objects_dict[object_name].contact_geoms)
    return (
        bool(inner.check_contact(left, object_geoms)),
        bool(inner.check_contact(right, object_geoms)),
    )


def task_objects(inner) -> tuple[str, str]:
    goals = inner.parsed_problem["goal_state"]
    if len(goals) != 1 or len(goals[0]) != 3 or goals[0][0].lower() != "in":
        raise ValueError(f"Expected one In goal, got {goals}")
    return goals[0][1], goals[0][2]


class BasketOracle:
    def __init__(self, *, inner, object_name: str, site_name: str,
                 grasp_offset: np.ndarray, position_gain: float,
                 max_action: float, hover_height: float,
                 basket_clearance: float,
                 reference_eef_quat: np.ndarray | None = None,
                 orientation_gain: float = 1.0,
                 max_rotation_action: float = 0.55) -> None:
        self.inner = inner
        self.object_name = object_name
        self.site_name = site_name
        self.grasp_offset = np.asarray(grasp_offset, dtype=np.float64)
        self.position_gain = float(position_gain)
        self.max_action = float(max_action)
        self.hover_height = float(hover_height)
        self.basket_clearance = float(basket_clearance)
        self.reference_eef_quat = (
            None
            if reference_eef_quat is None
            else np.asarray(reference_eef_quat, dtype=np.float64)
        )
        self.orientation_gain = float(orientation_gain)
        self.max_rotation_action = float(max_rotation_action)
        obj_model = self.inner.objects_dict[self.object_name]
        object_height = float(obj_model.top_offset[2] - obj_model.bottom_offset[2])
        # Tall objects need extra clearance over the basket rim.  The reference
        # clearance was tuned for the approximately 8 cm boxes in this scene.
        self.transport_clearance = max(
            0.28,
            self.basket_clearance + max(0.0, object_height - 0.08),
        )
        self.phase = "hover_object"
        self.hold_steps = 0
        self.consecutive_grasp_steps = 0
        self.grasp_reference_xy = None
        self.transport_eef_z = None

    @staticmethod
    def _quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
        """Convert a robosuite xyzw quaternion to a rotation matrix."""
        x, y, z, w = np.asarray(quaternion, dtype=np.float64)
        norm = np.linalg.norm([x, y, z, w])
        if norm < 1e-12:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = np.asarray([x, y, z, w]) / norm
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
        """Return the shortest axis-angle vector for a rotation matrix."""
        trace = float(np.trace(rotation))
        angle = float(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
        if angle < 1e-7:
            return 0.5 * np.asarray(
                [
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                ]
            )
        if np.pi - angle < 1e-4:
            # The skew part vanishes near pi. Recover the axis from R + I.
            symmetric = (rotation + np.eye(3)) / 2.0
            axis = np.sqrt(np.maximum(np.diag(symmetric), 0.0))
            largest = int(np.argmax(axis))
            if axis[largest] > 1e-7:
                for index in range(3):
                    if index != largest:
                        axis[index] = symmetric[largest, index] / axis[largest]
            axis /= max(np.linalg.norm(axis), 1e-12)
            return axis * angle
        axis = np.asarray(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        ) / (2.0 * np.sin(angle))
        return axis * angle

    def _rotation_action(self, observation) -> tuple[np.ndarray, float]:
        if self.reference_eef_quat is None:
            return np.zeros(3, dtype=np.float32), 0.0
        current = self._quat_to_matrix(observation["robot0_eef_quat"])
        desired = self._quat_to_matrix(self.reference_eef_quat)
        # Robosuite applies the commanded delta as R_delta @ R_current and
        # scales a unit rotation action to 0.5 radians.
        rotvec = self._matrix_to_rotvec(desired @ current.T)
        action = np.clip(
            rotvec * (self.orientation_gain / 0.5),
            -self.max_rotation_action,
            self.max_rotation_action,
        )
        return action.astype(np.float32), float(np.linalg.norm(rotvec))

    def _positions(self, observation):
        eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        obj = np.asarray(
            self.inner.object_states_dict[self.object_name].get_geom_state()["pos"],
            dtype=np.float64,
        )
        site = np.asarray(
            self.inner.object_states_dict[self.site_name].get_geom_state()["pos"],
            dtype=np.float64,
        )
        return eef, obj, site

    def _move(
        self,
        observation,
        eef: np.ndarray,
        target: np.ndarray,
        gripper: float,
        action_limit: float | None = None,
    ) -> np.ndarray:
        limit = self.max_action if action_limit is None else min(
            self.max_action, float(action_limit)
        )
        translation = np.clip(
            (target - eef) * self.position_gain,
            -limit,
            limit,
        )
        action = np.zeros(7, dtype=np.float32)
        action[:3] = translation.astype(np.float32)
        action[3:6], _ = self._rotation_action(observation)
        action[6] = np.float32(gripper)
        return action

    def act(self, observation):
        eef, obj, site = self._positions(observation)
        left, right = contacts(self.inner, self.object_name)
        grasped = bool(left and right)
        contained = bool(
            self.inner.object_states_dict[self.site_name].check_contain(
                self.inner.object_states_dict[self.object_name]
            )
        )
        target = eef.copy()
        gripper = -1.0

        # Predicate-level progress has priority over the geometric phase.  An
        # object can naturally leave the fingers as it contacts the basket;
        # once it is contained, retrying the grasp would undo a valid result.
        if contained and not grasped:
            self.phase = "retreat"
            target = eef.copy()
            target[2] = site[2] + self.transport_clearance
            gripper = -1.0
        elif contained and grasped:
            self.phase = "release"
            target = eef.copy()
            gripper = -1.0
            self.hold_steps += 1
        elif self.phase == "hover_object":
            target = obj + self.grasp_offset
            target[2] = max(target[2] + self.hover_height, obj[2] + self.hover_height)
            if np.linalg.norm(target - eef) < 0.018:
                self.phase = "descend"
        elif self.phase == "descend":
            target = obj + self.grasp_offset
            # Contact can prevent OSC from reaching the nominal Cartesian
            # target exactly.  One fingerpad touching is sufficient evidence
            # that closing is now safer than pushing indefinitely.
            xy_error = np.linalg.norm((target - eef)[:2])
            z_error = abs(float(target[2] - eef[2]))
            if (
                left
                or right
                or np.linalg.norm(target - eef) < 0.012
                or (xy_error < 0.015 and z_error < 0.040)
            ):
                self.phase = "close"
                self.hold_steps = 0
        elif self.phase == "close":
            target = obj + self.grasp_offset
            gripper = 1.0
            self.hold_steps += 1
            self.consecutive_grasp_steps = (
                self.consecutive_grasp_steps + 1 if grasped else 0
            )
            if self.consecutive_grasp_steps >= 10:
                self.phase = "lift"
                self.grasp_reference_xy = eef[:2].copy()
                self.transport_eef_z = site[2] + self.transport_clearance
            elif self.hold_steps >= 18 and not grasped:
                self.phase = "hover_object"
                self.hold_steps = 0
                self.consecutive_grasp_steps = 0
        elif self.phase == "lift":
            gripper = 1.0
            target = eef.copy()
            if self.grasp_reference_xy is not None:
                target[:2] = self.grasp_reference_xy
            target[2] = self.transport_eef_z
            if not grasped:
                self.phase = "hover_object"
            elif abs(target[2] - eef[2]) < 0.018:
                self.phase = "transport"
        elif self.phase == "transport":
            gripper = 1.0
            target = site + self.grasp_offset
            target[2] = self.transport_eef_z
            if not grasped:
                self.phase = "hover_object"
            elif np.linalg.norm(target - eef) < 0.022:
                self.phase = "lower"
        elif self.phase == "lower":
            gripper = 1.0
            target = site + self.grasp_offset
            target[2] = site[2] + self.grasp_offset[2]
            if not grasped:
                self.phase = "hover_object"
            elif np.linalg.norm(target - eef) < 0.018 or contained:
                self.phase = "release"
                self.hold_steps = 0
        elif self.phase == "release":
            target = eef.copy()
            gripper = -1.0
            self.hold_steps += 1
            if self.hold_steps >= 10:
                self.phase = "retreat"
        elif self.phase == "retreat":
            target = eef.copy()
            target[2] = site[2] + self.basket_clearance
            gripper = -1.0
        else:
            raise RuntimeError(self.phase)

        info = {
            "phase": self.phase,
            "eef": eef.tolist(),
            "object": obj.tolist(),
            "site": site.tolist(),
            "target": target.tolist(),
            "left_contact": left,
            "right_contact": right,
            "grasped": grasped,
            "contained": contained,
            "eef_target_distance": float(np.linalg.norm(target - eef)),
            "transport_clearance": float(self.transport_clearance),
        }
        _, orientation_error = self._rotation_action(observation)
        info["orientation_error_rad"] = orientation_error
        # Use a gentler vertical lift to let the fingers settle around tall,
        # narrow objects.  Once airborne, a faster horizontal move reduces the
        # duration over which the object can creep inside the gripper.
        action_limit = 0.25 if self.phase == "lift" else None
        return self._move(observation, eef, target, gripper, action_limit), info

