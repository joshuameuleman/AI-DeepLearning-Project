# Snake AI with Deep Q-Learning (PyTorch)

An AI project that uses **Reinforcement Learning** and **Deep Learning** with **Deep Q-Networks (DQN)**. The current setup is organized so Snake, Flappy Bird, and 2048 can each run through the same RL pipeline.



---

## What is Reinforcement Learning? (Beginner-Friendly)
Reinforcement Learning (RL) is a way to train an agent by **letting it learn from experience**:

- The agent is placed in an environment (here: the Snake game).
- It chooses actions (move directions).
- It receives rewards (good or bad feedback).
- Over many attempts, it learns which actions lead to higher total reward.

Unlike supervised learning (with “correct answers”), RL learns mainly by **trial and error**.

---

## How the Snake Environment Works
The environment is the game world the agent interacts with:

- The snake moves on a grid.
- Each step, the snake chooses a direction.
- The snake grows when it eats food.
- The episode ends if the snake:
  - hits a wall, or
  - hits its own body.

**Goal:** maximize the total reward by eating food and avoiding collisions.

---

## Core Concepts
### Agent
The **agent** is the “player” controlled by the AI. It uses a neural network to estimate how good each possible move is.

### State
The **state** is what the agent “sees” at the current time step.

A common (simple) state representation for Snake includes:
- Danger straight/right/left (collision risk)
- Current movement direction
- Food location relative to the head (food left/right/up/down)

This turns the game situation into a small vector the model can understand.

### Actions
The **actions** are the moves the agent can take.

Typical action space (3 actions):
- Go straight
- Turn right
- Turn left

(You can also implement 4 actions: up/down/left/right — both are fine.)

### Reward System
The **reward** tells the agent what was good or bad.

A typical reward setup:
- **+10** for eating food
- **-10** for dying (collision)
- Small negative reward like **-0.1** per step (optional) to encourage faster food collection

The exact numbers can vary — what matters is the *idea*: reward food, punish death.

---

## Model Architecture (Neural Network)
DQN uses a neural network to approximate a function $Q(s, a)$:

- Input: the state vector
- Output: one Q-value per possible action (how good each action is)

A common, simple architecture:
- Fully-connected (MLP) network
- 2 hidden layers (e.g. 128–256 units) with ReLU
- Output layer size = number of actions

---

## Training Process (DQN)
### Exploration vs. Exploitation
The agent must balance:

- **Exploration:** try random actions to discover better strategies
- **Exploitation:** use the best-known action based on the model

### Epsilon-Greedy Strategy
A standard approach is **epsilon-greedy**:

- With probability $\varepsilon$, pick a random action (explore)
- Otherwise, pick the action with the highest predicted Q-value (exploit)

During training, $\varepsilon$ usually **decreases over time**, so the agent explores a lot at the start and becomes more confident later.

### Key DQN Ideas (High-Level)
Most DQN Snake projects use:

- **Replay memory:** store experiences $(state, action, reward, next\_state, done)$ and sample random batches
- **Bellman update:** train the network to match a target value
- (Optional) **Target network:** a second network updated less frequently to stabilize learning

---

## Installation
> Recommended: Python 3.10 or 3.11 (best compatibility for PyTorch GPU wheels).

1) Clone the repository
```bash
git clone <YOUR_REPO_URL>
cd AI-DeepLearning-Project
```

2) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3) Install dependencies

- If you have a `requirements.txt`:
```bash
pip install -r requirements.txt
```

- If you **don’t** have one yet, a common baseline is:
```bash
pip install torch numpy pygame matplotlib
```

4) (Optional) Verify PyTorch
```bash
python -c "import torch; print('torch:', torch.__version__)"
```

---

## How to Run
Because project structures differ, use the commands that match your entry scripts:

- Interactive launcher from the project root:
```bash
python main.py
```

- Visualize a game directly:
```bash
python DQN/visualize.py --game snake
```

- Train the agent (common script name: `train.py`):
```bash
python DQN/train.py --game snake --episodes 1
```

- Play/watch the trained agent (common script name: `play.py` or `agent_play.py`):
```bash
python play.py
```

- If your project uses a package layout (example):
```bash
python -m src.train
```

### Where to Configure Settings
Typical configs you might expose in code:
- number of games/episodes
- learning rate
- batch size
- replay memory size
- epsilon start/end/decay

## RL Structure
The DQN code is split into clear RL parts:

- `DQN/src/envs/`: environment wrappers for each game
- `DQN/src/agents/`: agent logic and replay memory
- `DQN/src/models/`: Q-network and checkpoint helpers
- `DQN/src/training/`: config and training loop
- `DQN/src/visualization/`: text-based visualization runner

Generated outputs:

- `DQN/checkpoints/<game>/latest.pth`
- `DQN/logs/<game>/metrics.csv`

---

## Example Results (What You Should Expect)
Your exact results depend on rewards, state design, and hyperparameters, but a typical learning curve looks like:

- Early training: random movement, low scores (often 0–2)
- After some training: more consistent food collection and longer survival
- Later training: higher average score and fewer immediate deaths

Suggested metrics to report:
- **Score per episode**
- **Moving average score** (e.g. over last 50 games)
- **Best score** achieved

Add your training plot here:
- `assets/learning_curve.png` (placeholder)

---

## Screenshots / GIFs (Optional)
Add media to show progress visually:

- Gameplay GIF (placeholder): `assets/snake_agent.gif`
- Screenshot (placeholder): `assets/screenshot.png`

Markdown example:
```md
![Trained agent gameplay](assets/snake_agent.gif)
```

---

## Technologies Used
- Python
- PyTorch

Common supporting libraries (depending on your implementation):
- NumPy (math)
- Pygame (Snake rendering)
- Matplotlib (plots)

---

## Future Improvements
- Improve the state representation (more “game awareness”)
- Use a target network + soft updates (if not already)
- Tune hyperparameters (learning rate, batch size, epsilon decay)
- Prioritized experience replay
- Double DQN / Dueling DQN
- Save/load checkpoints and track experiments

---

## Conclusion
This project demonstrates how Reinforcement Learning can train an agent through trial and error. By using a DQN in PyTorch, the agents learn from rewards, replay memory, and neural-network updates over time.
