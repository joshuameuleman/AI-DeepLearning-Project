# DQN Workspace

Deze map bevat alle DQN-gerelateerde code en output.

## Structuur
- checkpoints/: opgeslagen modellen per game
- logs/: trainingslogs per game
- configs/: configuraties (optioneel YAML of JSON)
- src/: herbruikbare code (agent, model, training, utils)
- train.py: start trainingsrun
- simulate.py: start inferentie/simulatie

## Voorbeeld
python train.py --game snake --episodes 1000
python simulate.py --game snake --checkpoint checkpoints/snake/latest.pth --episodes 3
