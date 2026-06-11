"""Clean Snake rules engine used by DQN, simulation, and manual play."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
import os
from typing import Any, Dict, List, Optional, Tuple

from DQN.src.utils.snake_config import SNAKE_DEFAULT_GRID_SIZE, resolve_snake_grid_size


Point = Tuple[int, int]


class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


@dataclass
class StepResult:
    state: Any
    reward: float
    done: bool
    info: Dict[str, Any]


class SnakeLogic:
    """Classic Snake logic with a stable 43-value DQN state vector."""

    GRID_WIDTH = SNAKE_DEFAULT_GRID_SIZE
    GRID_HEIGHT = SNAKE_DEFAULT_GRID_SIZE
    STATE_SIZE = 43

    ACTION_STRAIGHT = 0
    ACTION_TURN_RIGHT = 1
    ACTION_TURN_LEFT = 2

    FOOD_REWARD = 12.0
    WALL_COLLISION_REWARD = -10.0
    SELF_COLLISION_REWARD = -30.0
    STEP_REWARD = -0.01
    TOWARD_FOOD_REWARD = 0.14
    AWAY_FROM_FOOD_REWARD = -0.08
    CYCLE_FOLLOW_REWARD = 0.0
    CYCLE_LEAVE_REWARD = 0.0
    OPEN_SPACE_REWARD = 0.18
    LOW_SPACE_REWARD = -0.50
    TAIL_ACCESS_REWARD = 0.08
    TAIL_BLOCKED_REWARD = -0.50
    TRAP_REWARD = -0.90
    BOARD_FILLED_REWARD = 300.0
    STAGNATION_REWARD = -8.0

    _DIR_VECTORS: Tuple[Point, ...] = (
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
    )

    def __init__(self, grid_size: Optional[int] = None, rng: Optional[random.Random] = None) -> None:
        resolved_grid_size = resolve_snake_grid_size(grid_size)
        self.GRID_WIDTH = resolved_grid_size
        self.GRID_HEIGHT = resolved_grid_size
        self.rng = rng or random.Random()
        self.max_stagnation_steps = max(100, self.GRID_WIDTH * self.GRID_HEIGHT * 6)

        self.body: List[Point] = []
        self.foods: List[Point] = []
        self.target_food_count = 1
        self.food: Optional[Point] = None
        self.direction = int(Direction.RIGHT)
        self.next_direction = int(Direction.RIGHT)
        self.score = 0
        self.step_count = 0
        self.steps_since_food = 0
        self._body_set: set[Point] = set()
        self._food_set: set[Point] = set()
        self._cycle: List[Point] = self._build_hamiltonian_cycle()
        self._cycle_index: Dict[Point, int] = {
            point: index for index, point in enumerate(self._cycle)
        }
        self.use_cycle_shaping = os.environ.get("SNAKE_CYCLE_SHAPING", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "ja",
            "on",
        )
        self.use_hamiltonian_start = os.environ.get("SNAKE_HAMILTONIAN_START", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "ja",
            "on",
        )
        self.space_reward_every = self._resolve_space_reward_interval()

    def reset(self) -> List[float]:
        """Start a fresh Snake episode."""
        self.body = self._make_initial_body()
        self.direction = self._direction_between(self.body[1], self.body[0])
        self.next_direction = self.direction
        self._body_set = set(self.body)
        self.foods = []
        self._food_set = set()
        self.food = None
        self.score = 0
        self.step_count = 0
        self.steps_since_food = 0
        self._spawn_food()
        return self._get_state()

    def step(self, action: int) -> StepResult:
        """Advance the game by one DQN action.

        Action convention:
        - 0 = keep going straight
        - 1 = turn right relative to the current direction
        - 2 = turn left relative to the current direction
        """
        if not self.body:
            self.reset()
        self.step_count += 1

        action = self._normalize_action(action)
        if action == self.ACTION_TURN_RIGHT:
            self.next_direction = (self.direction + 1) % 4
        elif action == self.ACTION_TURN_LEFT:
            self.next_direction = (self.direction - 1) % 4
        else:
            self.next_direction = self.direction

        self.direction = self.next_direction
        head = self.body[0]
        new_head = self._move_point(head, Direction(self.direction))
        previous_distance = self._distance_to_food(head)
        will_eat = new_head in self._food_set

        done = False
        reward = self.STEP_REWARD
        info: Dict[str, Any] = {"food_eaten": False}

        if self._is_wall_collision(new_head):
            done = True
            reward = self.WALL_COLLISION_REWARD
            info["reason"] = "wall_collision"
        elif self._is_self_collision(new_head, will_eat=will_eat):
            done = True
            length_ratio = len(self.body) / max(1, self.GRID_WIDTH * self.GRID_HEIGHT)
            reward = self.SELF_COLLISION_REWARD - (20.0 * length_ratio)
            info["reason"] = "self_collision"
        else:
            self._move_snake(new_head, grow=will_eat)
            if will_eat:
                reward = self.FOOD_REWARD
                self.score += 10
                self.steps_since_food = 0
                self._remove_food(new_head)
                if len(self.body) >= self.GRID_WIDTH * self.GRID_HEIGHT:
                    done = True
                    reward += self.BOARD_FILLED_REWARD
                    info["reason"] = "board_filled"
                    info["board_filled"] = True
                else:
                    reward += self._space_control_reward(new_head, force=True)
                    self._spawn_food()
                info["food_eaten"] = True
            else:
                self.steps_since_food += 1
                reward += self._food_progress_reward(new_head, previous_distance)
                reward += self._space_control_reward(new_head)
                if self.use_cycle_shaping:
                    reward += self._cycle_follow_reward(head, new_head)
                if self.steps_since_food >= self.max_stagnation_steps:
                    done = True
                    reward += self.STAGNATION_REWARD
                    info["reason"] = "stagnation_timeout"

        self._sync_primary_food()
        return StepResult(state=self._get_state(), reward=reward, done=done, info=info)

    def step_towards(self, direction: int | Direction) -> StepResult:
        """Advance using an absolute direction, useful for keyboard play."""
        return self.step(self.action_for_direction(direction))

    def action_for_direction(self, direction: int | Direction) -> int:
        """Convert an absolute direction into the relative DQN action space."""
        desired = int(direction) % 4
        if desired == self.direction:
            return self.ACTION_STRAIGHT
        if desired == (self.direction + 1) % 4:
            return self.ACTION_TURN_RIGHT
        if desired == (self.direction - 1) % 4:
            return self.ACTION_TURN_LEFT
        return self.ACTION_STRAIGHT

    def expert_action(self) -> int:
        """Return a safe Hamiltonian-cycle action that can fill an even board."""
        if not self.body:
            self.reset()

        next_point = self._cycle_successor(self.body[0])
        if next_point is None:
            return self.ACTION_STRAIGHT
        desired_direction = self._direction_between(self.body[0], next_point)
        return self.action_for_direction(desired_direction)

    def expert_direction(self) -> Direction:
        """Return the absolute direction used by the Hamiltonian solver."""
        if not self.body:
            self.reset()

        next_point = self._cycle_successor(self.body[0])
        if next_point is None:
            return Direction(self.direction)
        return Direction(self._direction_between(self.body[0], next_point))

    def get_state(self) -> List[float]:
        return self._get_state()

    def action_space(self) -> Tuple[int, ...]:
        return (self.ACTION_STRAIGHT, self.ACTION_TURN_RIGHT, self.ACTION_TURN_LEFT)

    def _normalize_action(self, action: int) -> int:
        try:
            value = int(action)
        except (TypeError, ValueError):
            return self.ACTION_STRAIGHT
        return value if value in self.action_space() else self.ACTION_STRAIGHT

    def _make_initial_body(self) -> List[Point]:
        if self.use_hamiltonian_start and len(self._cycle) >= 3:
            head_index = self.rng.randrange(len(self._cycle))
            return [
                self._cycle[(head_index - offset) % len(self._cycle)]
                for offset in range(3)
            ]

        direction = Direction(self.rng.choice(tuple(Direction)))
        dx, dy = self._DIR_VECTORS[int(direction)]
        head_x = self._start_axis(self.GRID_WIDTH, dx)
        head_y = self._start_axis(self.GRID_HEIGHT, dy)
        return [(head_x - i * dx, head_y - i * dy) for i in range(3)]

    def _start_axis(self, size: int, delta: int) -> int:
        lower = 2 if delta > 0 else 0
        upper = size - 3 if delta < 0 else size - 1
        return min(max(size // 2, lower), upper)

    def _move_point(self, point: Point, direction: Direction) -> Point:
        dx, dy = self._DIR_VECTORS[int(direction)]
        return (point[0] + dx, point[1] + dy)

    def _direction_between(self, start: Point, end: Point) -> int:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        for direction, vector in enumerate(self._DIR_VECTORS):
            if vector == (dx, dy):
                return direction
        return int(Direction.RIGHT)

    def _move_snake(self, new_head: Point, *, grow: bool) -> None:
        self.body.insert(0, new_head)
        self._body_set.add(new_head)
        if grow:
            return

        tail = self.body.pop()
        if tail != new_head:
            self._body_set.discard(tail)

    def _spawn_food(self) -> None:
        occupied = set(self._body_set)
        occupied.update(self._food_set)
        board_cells = self.GRID_WIDTH * self.GRID_HEIGHT
        if len(occupied) >= board_cells:
            self._sync_primary_food()
            return

        for _ in range(128):
            candidate = (
                self.rng.randrange(self.GRID_WIDTH),
                self.rng.randrange(self.GRID_HEIGHT),
            )
            if candidate not in occupied:
                self.foods.append(candidate)
                self._food_set.add(candidate)
                self._sync_primary_food()
                return

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                candidate = (x, y)
                if candidate not in occupied:
                    self.foods.append(candidate)
                    self._food_set.add(candidate)
                    self._sync_primary_food()
                    return

    def _remove_food(self, point: Point) -> None:
        self._food_set.discard(point)
        self.foods = [food for food in self.foods if food != point]
        self._sync_primary_food()

    def _sync_primary_food(self) -> None:
        self.food = self.foods[0] if self.foods else None

    def _is_wall_collision(self, point: Point) -> bool:
        x, y = point
        return x < 0 or x >= self.GRID_WIDTH or y < 0 or y >= self.GRID_HEIGHT

    def _is_self_collision(self, point: Point, *, will_eat: bool) -> bool:
        if will_eat:
            return point in self._body_set
        return point in self._body_set and point != self.body[-1]

    def _would_collide(self, direction: int) -> bool:
        if not self.body:
            return False
        next_point = self._move_point(self.body[0], Direction(direction % 4))
        if self._is_wall_collision(next_point):
            return True
        return next_point in self._body_set and next_point != self.body[-1]

    def _build_hamiltonian_cycle(self) -> List[Point]:
        """Build a cycle for boards where at least one dimension is even."""
        if self.GRID_WIDTH < 2 or self.GRID_HEIGHT < 2:
            return []

        def row_cycle(width: int, height: int) -> List[Point]:
            cycle: List[Point] = [(0, 0)]
            for y in range(height):
                if y % 2 == 0:
                    x_range = range(1, width)
                else:
                    x_range = range(width - 1, 0, -1)
                for x in x_range:
                    cycle.append((x, y))

            for y in range(height - 1, 0, -1):
                cycle.append((0, y))
            return cycle

        if self.GRID_HEIGHT % 2 == 0:
            cycle = row_cycle(self.GRID_WIDTH, self.GRID_HEIGHT)
        elif self.GRID_WIDTH % 2 == 0:
            cycle = [(y, x) for x, y in row_cycle(self.GRID_HEIGHT, self.GRID_WIDTH)]
        else:
            return []

        return cycle if len(cycle) == self.GRID_WIDTH * self.GRID_HEIGHT else []

    def _cycle_successor(self, point: Point) -> Optional[Point]:
        if not self._cycle:
            return None
        index = self._cycle_index.get(point)
        if index is None:
            return None
        return self._cycle[(index + 1) % len(self._cycle)]

    def _cycle_follow_reward(self, old_head: Point, new_head: Point) -> float:
        successor = self._cycle_successor(old_head)
        if successor is None:
            return 0.0
        return self.CYCLE_FOLLOW_REWARD if new_head == successor else self.CYCLE_LEAVE_REWARD

    def _nearest_food(self, head: Point) -> Optional[Point]:
        if not self.foods:
            return None
        return min(self.foods, key=lambda food: self._manhattan(head, food))

    def _distance_to_food(self, head: Point) -> Optional[int]:
        food = self._nearest_food(head)
        return None if food is None else self._manhattan(head, food)

    def _food_progress_reward(self, new_head: Point, previous_distance: Optional[int]) -> float:
        if previous_distance is None:
            return 0.0
        next_distance = self._distance_to_food(new_head)
        if next_distance is None:
            return 0.0
        if next_distance < previous_distance:
            return self.TOWARD_FOOD_REWARD
        if next_distance > previous_distance:
            return self.AWAY_FROM_FOOD_REWARD
        return 0.0

    def _safe_move_count(self, head: Point, body: List[Point]) -> int:
        if not body:
            return 0
        body_without_tail = self._body_set if body is self.body else set(body[:-1])
        tail = body[-1]
        safe_moves = 0
        for direction in Direction:
            candidate = self._move_point(head, direction)
            if self._is_wall_collision(candidate):
                continue
            if candidate in body_without_tail and candidate != tail:
                continue
            safe_moves += 1
        return safe_moves

    def _resolve_space_reward_interval(self) -> int:
        raw = os.environ.get("SNAKE_SPACE_REWARD_EVERY", "").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)

        board_cells = self.GRID_WIDTH * self.GRID_HEIGHT
        if board_cells <= 256:
            return 1
        if board_cells <= 1024:
            return 4
        if board_cells <= 4096:
            return 16
        return 64

    def _space_control_reward(self, head: Point, *, force: bool = False) -> float:
        board_cells = max(1, self.GRID_WIDTH * self.GRID_HEIGHT)
        length_ratio = len(self.body) / board_cells
        safe_moves = self._safe_move_count(head, self.body)

        if not force and self.step_count % self.space_reward_every != 0:
            if safe_moves <= 1:
                return self.TRAP_REWARD * (0.4 + length_ratio)
            if safe_moves == 2 and length_ratio > 0.35:
                return -0.08
            return 0.0

        free_cells = max(1, board_cells - len(self.body) + 1)
        reachable = self._reachable_space_count(head)
        reachable_ratio = reachable / free_cells

        reward = self.OPEN_SPACE_REWARD * reachable_ratio
        if reachable_ratio < 0.35:
            reward += self.LOW_SPACE_REWARD * (1.0 - reachable_ratio)

        if self._can_reach_tail(head):
            reward += self.TAIL_ACCESS_REWARD * (0.5 + length_ratio)
        else:
            reward += self.TAIL_BLOCKED_REWARD * (0.5 + length_ratio)

        if safe_moves <= 1:
            reward += self.TRAP_REWARD * (0.4 + length_ratio)
        elif safe_moves == 2 and length_ratio > 0.35:
            reward -= 0.08

        return reward

    def _reachable_space_count(self, start: Point) -> int:
        if self._is_wall_collision(start):
            return 0

        blockers = set(self.body[1:-1])
        queue = [start]
        seen = {start}
        count = 0

        while queue:
            point = queue.pop()
            count += 1
            for direction in Direction:
                candidate = self._move_point(point, direction)
                if candidate in seen or self._is_wall_collision(candidate) or candidate in blockers:
                    continue
                seen.add(candidate)
                queue.append(candidate)

        return count

    def _can_reach_tail(self, start: Point) -> bool:
        if not self.body:
            return True
        tail = self.body[-1]
        if start == tail:
            return True

        blockers = set(self.body[1:-1])
        queue = [start]
        seen = {start}

        while queue:
            point = queue.pop()
            for direction in Direction:
                candidate = self._move_point(point, direction)
                if candidate == tail:
                    return True
                if candidate in seen or self._is_wall_collision(candidate) or candidate in blockers:
                    continue
                seen.add(candidate)
                queue.append(candidate)

        return False

    def _get_state(self) -> List[float]:
        if not self.body:
            return [0.0] * self.STATE_SIZE

        head_x, head_y = self.body[0]
        nearest_food = self._nearest_food((head_x, head_y))
        food_x, food_y = nearest_food if nearest_food is not None else (head_x, head_y)

        direction_flags = {
            Direction.RIGHT: (0.0, 0.0, 0.0, 1.0),
            Direction.DOWN: (0.0, 1.0, 0.0, 0.0),
            Direction.LEFT: (0.0, 0.0, 1.0, 0.0),
            Direction.UP: (1.0, 0.0, 0.0, 0.0),
        }
        dir_up, dir_down, dir_left, dir_right = direction_flags[Direction(self.direction)]

        max_x = max(1, self.GRID_WIDTH - 1)
        max_y = max(1, self.GRID_HEIGHT - 1)
        wall_left = head_x / max_x
        wall_right = (self.GRID_WIDTH - 1 - head_x) / max_x
        wall_up = head_y / max_y
        wall_down = (self.GRID_HEIGHT - 1 - head_y) / max_y

        board_cells = max(1, self.GRID_WIDTH * self.GRID_HEIGHT)
        length_ratio = len(self.body) / board_cells
        free_ratio = 1.0 - length_ratio
        safe_moves_norm = self._safe_move_count((head_x, head_y), self.body) / 4.0
        tail_x, tail_y = self.body[-1]
        max_manhattan = max(1, self.GRID_WIDTH + self.GRID_HEIGHT - 2)
        tail_distance_norm = self._manhattan((head_x, head_y), (tail_x, tail_y)) / max_manhattan

        return [
            float(self._would_collide(self.direction)),
            float(self._would_collide((self.direction + 1) % 4)),
            float(self._would_collide((self.direction - 1) % 4)),
            dir_up,
            dir_down,
            dir_left,
            dir_right,
            float(food_x < head_x),
            float(food_x > head_x),
            float(food_y < head_y),
            float(food_y > head_y),
            wall_left,
            wall_right,
            wall_up,
            wall_down,
            *self._get_ray_features(head_x, head_y),
            length_ratio,
            free_ratio,
            safe_moves_norm,
            tail_distance_norm,
        ]

    def _get_ray_features(self, head_x: int, head_y: int) -> List[float]:
        directions: Tuple[Point, ...] = (
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        )
        features: List[float] = []
        max_distance = float(max(self.GRID_WIDTH, self.GRID_HEIGHT))

        for dx, dy in directions:
            x = head_x
            y = head_y
            steps = 0
            body_distance = 0.0
            food_distance = 0.0

            while True:
                x += dx
                y += dy
                steps += 1

                if x < 0 or x >= self.GRID_WIDTH or y < 0 or y >= self.GRID_HEIGHT:
                    wall_distance = steps / max_distance
                    features.extend([wall_distance, body_distance, food_distance])
                    break

                if body_distance == 0.0 and (x, y) in self._body_set:
                    body_distance = steps / max_distance
                if food_distance == 0.0 and (x, y) in self._food_set:
                    food_distance = steps / max_distance

        return features

    def _manhattan(self, a: Point, b: Point) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
