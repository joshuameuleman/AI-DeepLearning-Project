"""Core game loop logic for Snake."""

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from DQN.src.utils.snake_config import SNAKE_DEFAULT_GRID_SIZE


@dataclass
class StepResult:
    state: Any
    reward: float
    done: bool
    info: Dict[str, Any]


class SnakeLogic:
    """Pure game logic: rules, state transitions and terminal checks."""

    _DIR_VECTORS: Tuple[Tuple[int, int], ...] = (
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
    )

    GRID_WIDTH = SNAKE_DEFAULT_GRID_SIZE
    GRID_HEIGHT = SNAKE_DEFAULT_GRID_SIZE
    
    def __init__(self):
        env_grid = os.environ.get("SNAKE_GRID_SIZE", "").strip()
        parsed_grid = int(env_grid) if env_grid.isdigit() and int(env_grid) > 3 else None
        grid_size = parsed_grid or self.GRID_WIDTH
        env_stagnation = os.environ.get("SNAKE_STAGNATION_STEPS", "").strip()
        parsed_stagnation = int(env_stagnation) if env_stagnation.isdigit() and int(env_stagnation) > 0 else None

        self.GRID_WIDTH = grid_size
        self.GRID_HEIGHT = grid_size
        board_cells = self.GRID_WIDTH * self.GRID_HEIGHT
        # Allow long episodes so large boards can realistically be filled.
        self.max_stagnation_steps = parsed_stagnation or max(self.GRID_WIDTH * 16, board_cells)
        self.body: List[Tuple[int, int]] = []
        self.foods: List[Tuple[int, int]] = []
        self.target_food_count = 1
        self.food: Optional[Tuple[int, int]] = None
        self.direction = 0  # 0=right, 1=down, 2=left, 3=up (absolute)
        self.next_direction = 0
        self.score = 0
        self.steps_since_food = 0
        self.food_progress_stall_steps = 0
        self.recent_heads: List[Tuple[int, int]] = []
        self.target_food_lock: Optional[Tuple[int, int]] = None
        self._body_set: set[Tuple[int, int]] = set()
        self._food_set: set[Tuple[int, int]] = set()
        
    def reset(self) -> Any:
        """Initialize snake in center, food at random position."""
        center_x = self.GRID_WIDTH // 2
        center_y = self.GRID_HEIGHT // 2
        self.direction = random.randint(0, 3)
        dx, dy = self._DIR_VECTORS[self.direction]

        # Place body behind the head relative to the initial direction.
        self.body = [
            (center_x, center_y),
            (center_x - dx, center_y - dy),
            (center_x - 2 * dx, center_y - 2 * dy),
        ]
        self.next_direction = self.direction
        self.score = 0
        self.steps_since_food = 0
        self.food_progress_stall_steps = 0
        self.recent_heads = []
        self.target_food_lock = None
        self._body_set = set(self.body)
        self._food_set = set()
        min_food, max_food = self._food_count_range()
        self.target_food_count = random.randint(min_food, max_food)
        self.foods = []
        self._refill_foods()
        return self._get_state()

    def _food_choice_key(self, head: Tuple[int, int], food: Tuple[int, int]) -> Tuple[int, int, int, int]:
        hx, hy = head
        fx, fy = food
        tail = self.body[-1] if self.body else head
        dist = abs(fx - hx) + abs(fy - hy)
        # Reuse cached occupancy for the current body to avoid rebuilding large sets per candidate food.
        safe_after_eat = self._safe_move_count((fx, fy), self.body)
        risk_bucket = 0 if safe_after_eat >= 3 else (1 if safe_after_eat == 2 else 2)
        tail_dist = abs(fx - tail[0]) + abs(fy - tail[1])
        return (risk_bucket, dist, -safe_after_eat, -tail_dist)

    def _food_count_range(self) -> Tuple[int, int]:
        if self.GRID_WIDTH >= 256:
            return (16, 20)
        if self.GRID_WIDTH >= 128:
            return (8, 10)
        if self.GRID_WIDTH >= 64:
            return (2, 3)
        return (1, 2)

    def _safe_move_count(self, head: Tuple[int, int], body: List[Tuple[int, int]]) -> int:
        """Count how many immediate moves are safe from a given head/body snapshot."""
        if not body:
            return 0

        if body is self.body:
            body_blocks = self._body_set
        else:
            body_blocks = set(body)

        # We exclude tail because in the next tick it may move away.
        tail = body[-1]
        safe = 0
        for dx, dy in self._DIR_VECTORS:
            nx = head[0] + dx
            ny = head[1] + dy
            if nx < 0 or nx >= self.GRID_WIDTH or ny < 0 or ny >= self.GRID_HEIGHT:
                continue
            if (nx, ny) in body_blocks and (nx, ny) != tail:
                continue
            safe += 1
        return safe

    def _safety_adjustment(self, safe_moves: int) -> float:
        """Reward shaping component based on immediate mobility after a move."""
        # This shaping nudges policy away from dead-end positions.
        if safe_moves <= 1:
            return -0.6
        if safe_moves == 2:
            return -0.15
        return 0.03

    def _sync_primary_food(self) -> None:
        self.food = self.foods[0] if self.foods else None

    def _spawn_food(self, occupied: set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Spawn one food at a random free cell."""
        board_cells = self.GRID_WIDTH * self.GRID_HEIGHT
        if len(occupied) >= board_cells:
            return None

        for _ in range(64):
            x = random.randint(0, self.GRID_WIDTH - 1)
            y = random.randint(0, self.GRID_HEIGHT - 1)
            if (x, y) not in occupied:
                return (x, y)

        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in occupied:
                    return (x, y)
        return None

    def _refill_foods(self) -> None:
        occupied = set(self._body_set)
        occupied.update(self._food_set)
        while len(self.foods) < self.target_food_count:
            next_food = self._spawn_food(occupied)
            if next_food is None:
                break
            self.foods.append(next_food)
            self._food_set.add(next_food)
            occupied.add(next_food)
        self._sync_primary_food()

    def _nearest_food(self, head: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        if not self.foods:
            self.target_food_lock = None
            return None

        # Choose a target food that is not only close, but also safer after eating.
        best_food: Optional[Tuple[int, int]] = None
        best_key: Optional[Tuple[int, int, int, int]] = None
        for candidate in self.foods:
            key = self._food_choice_key(head, candidate)
            if best_key is None or key < best_key:
                best_key = key
                best_food = candidate

        assert best_food is not None
        assert best_key is not None

        # Hysteresis on food targeting: keep current target if it stays reasonably good.
        chosen_food = best_food
        if self.target_food_lock in self.foods:
            locked_food = self.target_food_lock
            locked_key = self._food_choice_key(head, locked_food)
            lock_is_reasonable = (
                locked_key[0] <= best_key[0]
                and locked_key[1] <= best_key[1] + 4
            )
            if lock_is_reasonable:
                chosen_food = locked_food

        self.target_food_lock = chosen_food
        return chosen_food
    
    def _get_state(self) -> List[float]:
        """
        Return state vector with 43 elements:
        - 11 core directional features (legacy)
        - 4 wall distance features
        - 24 ray features (8 directions x 3 signals)
        - 4 global stability features

        Ray signals per direction:
        [distance_to_wall, distance_to_body, distance_to_food]
        """
        head_x, head_y = self.body[0]
        nearest_food = self._nearest_food((head_x, head_y))
        food_x, food_y = nearest_food if nearest_food is not None else (head_x, head_y)
        
        # Direction encoding (one-hot)
        # One-hot encoding: exactly one value is 1.0, others are 0.0.
        # This is common in ML for categorical values like direction.
        dir_encoding = {
            0: (1, 0, 0, 0),  # right: left=0, right=1, up=0, down=0
            1: (0, 1, 0, 0),  # down:  left=0, right=0, up=0, down=1
            2: (0, 0, 1, 0),  # left: left=1, right=0, up=0, down=0
            3: (0, 0, 0, 1),  # up:   left=0, right=0, up=1, down=0
        }
        dir_left, dir_right, dir_up, dir_down = dir_encoding[self.direction]
        
        # Danger detection (collision in 3 directions relative to snake direction)
        danger_straight = float(self._would_collide(self.direction))
        danger_right = float(self._would_collide((self.direction + 1) % 4))
        danger_left = float(self._would_collide((self.direction - 1) % 4))
        
        # Food direction (relative to head)
        food_left = float(food_x < head_x)
        food_right = float(food_x > head_x)
        food_up = float(food_y < head_y)
        food_down = float(food_y > head_y)

        max_x = max(1, self.GRID_WIDTH - 1)
        max_y = max(1, self.GRID_HEIGHT - 1)
        wall_left = head_x / max_x
        wall_right = (self.GRID_WIDTH - 1 - head_x) / max_x
        wall_up = head_y / max_y
        wall_down = (self.GRID_HEIGHT - 1 - head_y) / max_y

        ray_features = self._get_ray_features(head_x, head_y)

        board_cells = self.GRID_WIDTH * self.GRID_HEIGHT
        length_ratio = len(self.body) / max(1, board_cells)
        free_ratio = 1.0 - length_ratio
        safe_moves = self._safe_move_count((head_x, head_y), self.body)
        safe_moves_norm = safe_moves / 4.0
        tail_x, tail_y = self.body[-1]
        max_manhattan = max(1, self.GRID_WIDTH + self.GRID_HEIGHT - 2)
        tail_distance_norm = (abs(head_x - tail_x) + abs(head_y - tail_y)) / max_manhattan
        
        # Final state is a flat list of floats. Order matters for the neural network.
        return [
            danger_straight, danger_right, danger_left,
            dir_up, dir_down, dir_left, dir_right,
            food_left, food_right, food_up, food_down,
            wall_left, wall_right, wall_up, wall_down,
            *ray_features,
            length_ratio, free_ratio, safe_moves_norm, tail_distance_norm,
        ]

    def _get_ray_features(self, head_x: int, head_y: int) -> List[float]:
        """Encode 8-direction vision as normalized distances to wall/body/food."""
        directions = [
            (0, -1),   # N
            (1, -1),   # NE
            (1, 0),    # E
            (1, 1),    # SE
            (0, 1),    # S
            (-1, 1),   # SW
            (-1, 0),   # W
            (-1, -1),  # NW
        ]
        features: List[float] = []
        max_dist = float(max(self.GRID_WIDTH, self.GRID_HEIGHT))

        for dx, dy in directions:
            x = head_x
            y = head_y
            steps = 0
            body_dist = 0.0
            food_dist = 0.0

            while True:
                x += dx
                y += dy
                steps += 1

                if x < 0 or x >= self.GRID_WIDTH or y < 0 or y >= self.GRID_HEIGHT:
                    wall_dist = steps / max_dist
                    features.extend([wall_dist, body_dist, food_dist])
                    break

                if body_dist == 0.0 and (x, y) in self._body_set:
                    body_dist = steps / max_dist
                if food_dist == 0.0 and (x, y) in self._food_set:
                    food_dist = steps / max_dist

        return features
    
    def _would_collide(self, direction: int) -> bool:
        """Check if moving in given direction would cause collision."""
        head_x, head_y = self.body[0]

        # Calculate next head position
        dx, dy = self._DIR_VECTORS[direction]
        
        next_x = head_x + dx
        next_y = head_y + dy
        
        # Wall collision
        if next_x < 0 or next_x >= self.GRID_WIDTH or next_y < 0 or next_y >= self.GRID_HEIGHT:
            return True
        
        # Self collision (body except tail which might move away)
        tail = self.body[-1]
        if (next_x, next_y) in self._body_set and (next_x, next_y) != tail:
            return True
        
        return False
    
    def step(self, action: int) -> StepResult:
        """
        Execute one step.
        Action: 0=straight, 1=right turn, 2=left turn
        """
        # Turn relative to current direction
        # Action space convention used by agent:
        # 0 = keep going straight, 1 = turn right, 2 = turn left.
        if action == 1:  # turn right
            self.next_direction = (self.direction + 1) % 4
        elif action == 2:  # turn left
            self.next_direction = (self.direction - 1) % 4
        else:  # action == 0, go straight
            self.next_direction = self.direction
        
        # Prevent reversing into self (180 degree turn)
        # Only allow if body has 1 segment (start of game)
        if len(self.body) > 1:
            opposite = (self.direction + 2) % 4
            if self.next_direction == opposite:
                self.next_direction = self.direction
        
        self.direction = self.next_direction
        
        # Move head
        head_x, head_y = self.body[0]
        prev_tail = self.body[-1]
        nearest_food = self._nearest_food((head_x, head_y))
        has_food = nearest_food is not None
        food_x, food_y = nearest_food if nearest_food is not None else (head_x, head_y)
        prev_distance = abs(head_x - food_x) + abs(head_y - food_y) if has_food else 0
        dx, dy = self._DIR_VECTORS[self.direction]
        
        new_head = (head_x + dx, head_y + dy)
        will_eat = new_head in self.foods
        
        # Check collisions
        done = False
        reward = 0.0
        info = {}
        
        # Terminal transitions (done=True) are penalized to teach survival.
        # Positive terminal bonus exists only for successful board completion.
        # Wall collision
        if new_head[0] < 0 or new_head[0] >= self.GRID_WIDTH or new_head[1] < 0 or new_head[1] >= self.GRID_HEIGHT:
            done = True
            reward = -50.0
            info["reason"] = "wall_collision"
        elif len(self.body) == self.GRID_WIDTH * self.GRID_HEIGHT and new_head == prev_tail or len(self.body) == self.GRID_WIDTH * self.GRID_HEIGHT and new_head == self.body[-1]:
            # Full-board loop completion: allow stepping onto tail and end with a large bonus.
            self.body.insert(0, new_head)
            self.body.pop()
            self._body_set = set(self.body)
            reward = 120.0
            done = True
            info["reason"] = "full_board_tail_touch"
        # Self collision
        elif new_head in self._body_set and (will_eat or new_head != prev_tail):
            done = True
            # Stronger penalty to discourage self-collisions observed in training metrics.
            reward = -120.0
            info["reason"] = "self_collision"
        else:
            # Valid move
            self.body.insert(0, new_head)
            self._body_set.add(new_head)
            
            # Food eaten path: snake grows (we do NOT pop tail this turn).
            if will_eat:
                self.foods.remove(new_head)
                self._food_set.discard(new_head)
                reward = 18.0
                self.score += 10
                self.steps_since_food = 0
                self.food_progress_stall_steps = 0
                self.recent_heads = []
                self.target_food_lock = None
                self._refill_foods()
                if len(self.body) >= self.GRID_WIDTH * self.GRID_HEIGHT:
                    reward += 500.0
                    done = True
                    info["reason"] = "board_filled"
                    info["board_filled"] = True
                else:
                    safe_moves = self._safe_move_count(new_head, self.body)
                    reward += self._safety_adjustment(safe_moves)
                    # Eating should not be blindly greedy: penalize trap-prone post-food states.
                    if safe_moves <= 1:
                        reward -= 2.5
                        info["post_food_trap_risk"] = True
                    elif safe_moves == 2:
                        reward -= 0.6
                info["foods_remaining"] = len(self.foods)
                info["food_eaten"] = True
            else:
                # No food path: classic snake move = add new head + remove tail.
                removed_tail = self.body.pop()
                # If we moved onto the old tail, the coordinate stays occupied by the new head.
                if removed_tail != new_head:
                    self._body_set.discard(removed_tail)
                # Small time penalty encourages efficient food-seeking.
                # Increase to discourage long aimless loops (helps reduce self-collisions).
                reward = -0.05
                if has_food:
                    next_nearest = self._nearest_food(new_head)
                    if next_nearest is not None:
                        # Distance shaping: move toward food -> small bonus, away -> small penalty.
                        new_distance = abs(new_head[0] - next_nearest[0]) + abs(new_head[1] - next_nearest[1])
                        if new_distance < prev_distance:
                            reward += 0.20
                            self.food_progress_stall_steps = 0
                        elif new_distance > prev_distance:
                            reward -= 0.20
                            self.food_progress_stall_steps += 1
                        else:
                            self.food_progress_stall_steps += 1

                        # Penalize repeated non-progress loops around food (circle behavior).
                        if self.food_progress_stall_steps >= 5:
                            orbit_penalty = min(2.0, 0.12 * (self.food_progress_stall_steps - 4))
                            reward -= orbit_penalty
                            info["orbit_penalty"] = round(orbit_penalty, 3)
                        if self.food_progress_stall_steps >= 20:
                            reward -= 1.0
                            info["orbit_hard_penalty"] = 1.0
                    else:
                        self.food_progress_stall_steps = 0
                else:
                    self.food_progress_stall_steps = 0
                safe_moves = self._safe_move_count(new_head, self.body)
                reward += self._safety_adjustment(safe_moves)
                # Penalize moving into positions with very few safe moves (dead-ends).
                if safe_moves <= 1:
                    reward -= 1.0

                # Penalize local loops: revisiting the same head positions repeatedly
                # without eating often indicates circling around food/body traps.
                self.recent_heads.append(new_head)
                if len(self.recent_heads) > 20:
                    self.recent_heads.pop(0)
                revisit_count = self.recent_heads.count(new_head)
                if revisit_count >= 3:
                    loop_penalty = min(1.6, 0.25 * (revisit_count - 2))
                    reward -= loop_penalty
                    info["loop_penalty"] = round(loop_penalty, 3)

                self.steps_since_food += 1
                info["food_eaten"] = False

        self._sync_primary_food()

        if not done and self.foods and self.steps_since_food >= self.max_stagnation_steps:
            done = True
            reward -= 30.0
            info["reason"] = "stagnation_timeout"
        
        state = self._get_state()
        return StepResult(state=state, reward=reward, done=done, info=info)

    def get_state(self) -> Any:
        """Get current state."""
        return self._get_state()

    def action_space(self) -> Tuple[int, ...]:
        return (0, 1, 2)


# Active clean-slate implementation.
# Kept here so the existing main.py/DQN dynamic loader can keep importing this path.
from Games.Snake.logic.clean_logic import Direction, SnakeLogic, StepResult  # noqa: E402,F401
