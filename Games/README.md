# Games

Elke game staat in een eigen map.
Alle spel-specifieke logica hoort in de submap `logic` van die game.

## Structuur
- Snake/logic
- Flappy Bird/logic
- 2048/logic

## Richtlijn
- Houd rendering, training en game-logica gescheiden.
- In `logic` plaats je regels, state-transities, rewards en collision/terminal checks.
