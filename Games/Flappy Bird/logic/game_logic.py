"""Core game loop logic for Flappy Bird."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class StepResult:
    state: Any
    reward: float
    done: bool
    info: Dict[str, Any]


class FlappyBirdLogic:
    """Pure game logic: rules, state transitions and terminal checks."""

    SCREEN_WIDTH = 288
    SCREEN_HEIGHT = 512
    BIRD_X = 56
    BIRD_RADIUS = 12

    GRAVITY = 0.42
    FLAP_VELOCITY = -6.2
    MAX_FALL_SPEED = 10.0

    PIPE_WIDTH = 52
    PIPE_SPEED = 3.0
    PIPE_GAP = 140
    PIPE_SPACING = 180
    PIPE_MARGIN = 60
    BONUS_EVERY_PIPES = 50
    BONUS_POINTS = 10
    BONUS_REWARD = 10.0

    def __init__(self) -> None:
        self.bird_y = float(self.SCREEN_HEIGHT * 0.5)
        self.bird_velocity = 0.0
        self.pipes: List[dict[str, float | bool]] = []
        self.score = 0
        self.pipes_passed = 0
        self.steps = 0
        self.done = False

    def _sample_gap_center(self) -> float:
        low = float(self.PIPE_MARGIN + self.PIPE_GAP / 2)
        high = float(self.SCREEN_HEIGHT - self.PIPE_MARGIN - self.PIPE_GAP / 2)
        return random.uniform(low, high)

    def _append_pipe(self, x: float) -> None:
        self.pipes.append({"x": x, "gap_y": self._sample_gap_center(), "passed": False})

    def _ensure_pipes(self) -> None:
        while len(self.pipes) < 3:
            if not self.pipes:
                next_x = float(self.SCREEN_WIDTH + 32)
            else:
                next_x = float(self.pipes[-1]["x"]) + self.PIPE_SPACING
            self._append_pipe(next_x)

    def _next_pipes(self) -> Tuple[dict[str, float | bool], dict[str, float | bool]]:
        candidates = [pipe for pipe in self.pipes if float(pipe["x"]) + self.PIPE_WIDTH >= self.BIRD_X]
        if not candidates:
            candidates = sorted(self.pipes, key=lambda p: float(p["x"]))
        if len(candidates) == 1:
            return candidates[0], candidates[0]
        return candidates[0], candidates[1]

    def _pipe_collision(self, pipe: dict[str, float | bool]) -> bool:
        pipe_x = float(pipe["x"])
        if self.BIRD_X + self.BIRD_RADIUS < pipe_x:
            return False
        if self.BIRD_X - self.BIRD_RADIUS > pipe_x + self.PIPE_WIDTH:
            return False

        gap_center = float(pipe["gap_y"])
        gap_top = gap_center - self.PIPE_GAP / 2
        gap_bottom = gap_center + self.PIPE_GAP / 2
        bird_top = self.bird_y - self.BIRD_RADIUS
        bird_bottom = self.bird_y + self.BIRD_RADIUS
        return bird_top < gap_top or bird_bottom > gap_bottom

    def reset(self) -> Any:
        self.bird_y = float(self.SCREEN_HEIGHT * 0.5)
        self.bird_velocity = 0.0
        self.score = 0
        self.pipes_passed = 0
        self.steps = 0
        self.done = False
        self.pipes = []
        self._ensure_pipes()
        return self._get_state()

    def step(self, action: int) -> StepResult:
        if self.done:
            return StepResult(state=self._get_state(), reward=0.0, done=True, info={"reason": "already_done"})

        # Ignore repeated flap spam while already moving upward fast;
        # this makes early exploration less likely to immediately hit the top bound.
        if int(action) == 1 and self.bird_velocity > -3.0:
            self.bird_velocity = self.FLAP_VELOCITY

        self.bird_velocity = min(self.bird_velocity + self.GRAVITY, self.MAX_FALL_SPEED)
        self.bird_y += self.bird_velocity
        self.steps += 1

        for pipe in self.pipes:
            pipe["x"] = float(pipe["x"]) - self.PIPE_SPEED

        while self.pipes and float(self.pipes[0]["x"]) + self.PIPE_WIDTH < 0:
            self.pipes.pop(0)
        self._ensure_pipes()

        reward = 0.05
        info: Dict[str, Any] = {}

        if self.bird_y - self.BIRD_RADIUS <= 0:
            self.done = True
            reward = -10.0
            info["reason"] = "ceiling_collision"
            return StepResult(state=self._get_state(), reward=reward, done=True, info=info)

        if self.bird_y + self.BIRD_RADIUS >= self.SCREEN_HEIGHT:
            self.done = True
            reward = -10.0
            info["reason"] = "ground_collision"
            return StepResult(state=self._get_state(), reward=reward, done=True, info=info)

        for pipe in self.pipes:
            if self._pipe_collision(pipe):
                self.done = True
                reward = -10.0
                info["reason"] = "pipe_collision"
                return StepResult(state=self._get_state(), reward=reward, done=True, info=info)

        passed_now = 0
        for pipe in self.pipes:
            if not bool(pipe["passed"]) and float(pipe["x"]) + self.PIPE_WIDTH < self.BIRD_X:
                pipe["passed"] = True
                passed_now += 1

        if passed_now > 0:
            prev_milestone = self.pipes_passed // self.BONUS_EVERY_PIPES
            self.pipes_passed += passed_now
            new_milestone = self.pipes_passed // self.BONUS_EVERY_PIPES
            bonus_hits = max(0, new_milestone - prev_milestone)
            bonus_points = bonus_hits * self.BONUS_POINTS

            self.score += passed_now + bonus_points
            reward += 3.0 * passed_now
            if bonus_points > 0:
                reward += self.BONUS_REWARD * bonus_hits
                info["bonus_points"] = bonus_points
                info["bonus_hits"] = bonus_hits
            info["passed_pipes"] = passed_now
            info["pipes_passed_total"] = self.pipes_passed

        next_pipe, _ = self._next_pipes()
        gap_center = float(next_pipe["gap_y"])
        dist_to_center = abs(self.bird_y - gap_center) / max(1.0, float(self.SCREEN_HEIGHT))
        reward += 0.1 * (1.0 - min(1.0, dist_to_center * 2.0))

        info["score"] = self.score
        return StepResult(state=self._get_state(), reward=reward, done=False, info=info)

    def get_state(self) -> Any:
        return self._get_state()

    def _get_state(self) -> List[float]:
        next_pipe, second_pipe = self._next_pipes()

        def _norm_x(pipe: dict[str, float | bool]) -> float:
            return (float(pipe["x"]) - self.BIRD_X) / max(1.0, float(self.SCREEN_WIDTH))

        def _norm_center_delta(pipe: dict[str, float | bool]) -> float:
            return (self.bird_y - float(pipe["gap_y"])) / max(1.0, float(self.SCREEN_HEIGHT))

        next_gap_center = float(next_pipe["gap_y"])
        gap_top = (next_gap_center - self.PIPE_GAP / 2) / float(self.SCREEN_HEIGHT)
        gap_bottom = (next_gap_center + self.PIPE_GAP / 2) / float(self.SCREEN_HEIGHT)

        return [
            self.bird_y / float(self.SCREEN_HEIGHT),
            self.bird_velocity / self.MAX_FALL_SPEED,
            _norm_x(next_pipe),
            gap_top,
            gap_bottom,
            _norm_center_delta(next_pipe),
            _norm_x(second_pipe),
            _norm_center_delta(second_pipe),
        ]

    def action_space(self) -> Tuple[int, ...]:
        return (0, 1)
