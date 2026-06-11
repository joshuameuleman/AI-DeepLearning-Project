# AI Deep Learning Project

Dit project is een Python/PyTorch-workspace voor reinforcement learning met Deep Q-Networks (DQN). Het project bevat meerdere games, een gedeelde DQN-trainingspipeline, opgeslagen checkpoints, metrics en een webinterface om Snake/Flappy Bird live mee te volgen tijdens training of simulatie.

De huidige games zijn:

- `snake`: volledig ondersteund voor spelen, trainen, simuleren, visualiseren en live web-feed.
- `flappy`: ondersteund via dezelfde DQN-pipeline en live web-feed.
- `2048`: game-logica en DQN-adapter zijn aanwezig; rendering/live-feed is beperkter dan bij Snake en Flappy.

## Inhoud

- [Projectstructuur](#projectstructuur)
- [Belangrijkste bestanden](#belangrijkste-bestanden)
- [Technologie](#technologie)
- [Installatie](#installatie)
- [Snel starten](#snel-starten)
- [Interactieve launcher](#interactieve-launcher)
- [Webinterface](#webinterface)
- [Training](#training)
- [Simulatie en visualisatie](#simulatie-en-visualisatie)
- [Snake handmatig spelen](#snake-handmatig-spelen)
- [Checkpoints en logs](#checkpoints-en-logs)
- [Hoe de DQN-pipeline werkt](#hoe-de-dqn-pipeline-werkt)
- [DQN-variabelen uitgelegd](#dqn-variabelen-uitgelegd)
- [Game-logica](#game-logica)
- [Configuratie](#configuratie)
- [Veelvoorkomende problemen](#veelvoorkomende-problemen)

## Projectstructuur

```text
.
├── main.py                         # Interactieve terminal-launcher
├── serve_web.py                    # HTTP-server + SSE live-feed + API voor web UI
├── requirements.txt                # Pinned Python dependencies
├── web/
│   ├── index.html                  # Browserinterface voor live meekijken/trainen/simuleren
│   ├── app.js                      # Frontendlogica
│   ├── styles.css                  # Styling
│   └── live_state.json             # Live-state bestand/output
├── DQN/
│   ├── train.py                    # CLI voor training
│   ├── train_snake_grids.py        # Train meerdere Snake-gridgroottes na elkaar
│   ├── simulate.py                 # CLI voor inferentie/simulatie
│   ├── visualize.py                # Simulatie met pygame-rendering
│   ├── checkpoints/                # Opgeslagen modellen per run
│   ├── logs/                       # CSV-metrics per run
│   └── src/
│       ├── agents/                 # DQN-agent en replay memory
│       ├── envs/                   # GameEnvironment-adapter
│       ├── models/                 # QNetwork en checkpoint helpers
│       ├── training/               # TrainConfig en Trainer
│       └── utils/                  # Paths, Snake-config en live-feed payloads
└── Games/
    ├── Snake/
    │   ├── logic/                  # Snake-regels, rewards, state en terminal checks
    │   ├── play.py                 # Handmatig Snake spelen
    │   ├── renderer.py             # Pygame-renderer
    │   └── textures/               # Snake-assets
    ├── Flappy Bird/
    │   ├── logic/                  # Flappy Bird-regels en rewards
    │   ├── renderer.py             # Pygame-renderer
    │   └── textures/               # Flappy-assets
    └── 2048/
        └── logic/                  # 2048-regels en rewards
```

## Belangrijkste Bestanden

Hieronder staat kort wat de belangrijkste bestanden doen. De codecomments per bestand kunnen later apart toegevoegd worden; deze sectie is bedoeld als snelle navigatie in de README.

| Bestand/map | Functie |
| --- | --- |
| `main.py` | Interactieve launcher waarmee je kiest tussen spelen, trainen, simuleren en visualiseren. |
| `serve_web.py` | Start de lokale webserver, levert de webinterface en beheert API-endpoints/SSE-events voor live updates. |
| `requirements.txt` | Bevat de Python-dependencies met vaste versies voor reproduceerbare installatie. |
| `web/index.html` | HTML-structuur van de browserinterface. |
| `web/app.js` | Frontendlogica voor status ophalen, training/simulatie starten en live state tekenen. |
| `web/styles.css` | Styling van de webinterface. |
| `DQN/train.py` | Start een DQN-training voor Snake, Flappy of 2048 en past game-specifieke trainingsinstellingen toe. |
| `DQN/train_snake_grids.py` | Hulpscript om meerdere Snake-gridgroottes na elkaar te trainen. |
| `DQN/simulate.py` | Laadt een checkpoint en laat een getrainde agent spelen, met optionele renderer of web-feed. |
| `DQN/visualize.py` | Simpele wrapper rond simulatie met pygame-rendering. |
| `DQN/src/agents/dqn_agent.py` | Bevat de DQN-agent met epsilon-greedy actiekeuze. |
| `DQN/src/agents/replay_memory.py` | Slaat ervaringen op en ondersteunt prioritized replay voor efficiënter leren. |
| `DQN/src/envs/game_env.py` | Adapter die game-logica omzet naar een uniforme RL-interface. |
| `DQN/src/models/q_network.py` | Neural network dat Q-values voorspelt per mogelijke actie. |
| `DQN/src/models/checkpoint.py` | Hulpfuncties voor checkpoints opslaan en laden. |
| `DQN/src/training/config.py` | Centrale dataclass met trainingshyperparameters. |
| `DQN/src/training/trainer.py` | Hoofdtrainingsloop: episodes draaien, leren, evalueren, loggen en checkpoints schrijven. |
| `DQN/src/utils/live_feed.py` | Bouwt live payloads voor Snake/Flappy en publiceert die naar de webinterface. |
| `DQN/src/utils/snake_config.py` | Bepaalt Snake-gridgrootte via argumenten, environment variables of defaults. |
| `Games/Snake/logic/game_logic.py` | Snake-spelregels, state, rewards en collision checks. |
| `Games/Snake/play.py` | Start handmatige Snake met pygame. |
| `Games/Snake/renderer.py` | Tekent Snake in een pygame-window. |
| `Games/Flappy Bird/logic/game_logic.py` | Flappy Bird-spelregels, state en rewardlogica. |
| `Games/Flappy Bird/renderer.py` | Tekent Flappy Bird in een pygame-window. |
| `Games/2048/logic/game_logic.py` | 2048-spelregels en RL-state/rewards. |

## Technologie

Het project gebruikt:

- Python 3.10 of 3.11 aanbevolen
- PyTorch voor het neurale netwerk en training
- NumPy voor numerieke data
- Pygame voor lokale game-rendering
- Matplotlib voor eventuele grafieken/metrics
- Een eenvoudige standaardbibliotheek HTTP-server voor de webinterface
- Server-Sent Events (SSE) voor live updates naar de browser

De dependencies staan gepind in `requirements.txt`. Die file gebruikt momenteel CUDA 12.1 PyTorch wheels:

```text
torch==2.2.2+cu121
numpy==1.26.4
pygame==2.5.2
matplotlib==3.8.4
```

Gebruik je geen NVIDIA GPU of geen CUDA 12.1-compatible omgeving, pas dan de PyTorch-installatie aan via de officiële PyTorch wheel-keuze.

## Installatie

Maak eerst een virtual environment aan:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Installeer daarna de dependencies:

```bash
pip install -r requirements.txt
```

Controleer PyTorch:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Als `torch.cuda.is_available()` `False` print, draait training op CPU tenzij je `--device cuda` forceert. CPU werkt, maar is langzamer.

## Snel Starten

Start de interactieve launcher:

```bash
python main.py
```

Start de webinterface:

```bash
python serve_web.py --host 0.0.0.0 --port 8000
```

Open daarna:

```text
http://127.0.0.1:8000/web/
```

Train Snake op een 32x32-grid:

```bash
python DQN/train.py --game snake --grid-size 32 --episodes 10000 --profile balanced --device auto
```

Simuleer met het beste checkpoint:

```bash
python DQN/simulate.py --game snake --grid-size 32 --checkpoint auto --episodes 3
```

Visualiseer met een pygame-window:

```bash
python DQN/visualize.py --game snake --grid-size 32 --episodes 1
```

## Interactieve Launcher

`main.py` is de centrale terminalingang. De launcher laat je kiezen tussen:

- `play`: speel Snake handmatig.
- `train`: train een DQN-agent.
- `simulate`: draai een checkpoint zonder renderwindow, optioneel met web-feed.
- `visualize`: draai een checkpoint met pygame-rendering.

Voor Snake vraagt de launcher ook naar gridgrootte, solver, trainingsprofiel, device en live meekijken wanneer dat relevant is.

## Webinterface

De webinterface staat in `web/` en wordt geserveerd door `serve_web.py`.

Start de server:

```bash
python serve_web.py --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000/web/
```

De webinterface kan:

- beschikbare Snake-modellen tonen voor `16x16`, `32x32`, `64x64` en `128x128`;
- `best_eval.pth` simuleren voor een gekozen grid;
- training starten voor een gekozen Snake-grid;
- de simulatiesnelheid aanpassen;
- live Snake- en Flappy-state tekenen via SSE.

Belangrijke endpoints:

- `GET /api/status`: modelstatus en actieve job ophalen.
- `POST /api/simulate`: Snake-simulatie starten.
- `POST /api/train`: Snake-training starten.
- `POST /api/speed`: FPS voor simulatie aanpassen.
- `GET /events`: SSE-stream met live game-state.

Let op: via de webinterface draait er bewust maar een training of simulatie tegelijk. Als er al een job actief is, geeft de API een conflictmelding terug.

## Training

De standaard trainings-CLI is:

```bash
python DQN/train.py --game snake --episodes 10000
```

Belangrijke opties:

```bash
python DQN/train.py \
  --game snake \
  --grid-size 64 \
  --episodes 50000 \
  --profile balanced \
  --device auto \
  --cpu-threads 1
```

Opties:

- `--game`: `snake`, `flappy` of `2048`.
- `--episodes`: aantal trainingsepisodes.
- `--fresh`: negeert bestaande checkpoints en begint opnieuw.
- `--grid-size`: Snake-gridgrootte, standaard `32`.
- `--profile`: Snake-profiel `fast`, `balanced` of `quality`.
- `--cpu-threads`: beperkt PyTorch CPU-threads; `0` betekent PyTorch default.
- `--device`: `auto`, `cpu` of `cuda`.

Snake-profielen:

- `fast`: sneller testen, lagere belasting, minder agressieve training.
- `balanced`: standaardkeuze voor normale training.
- `quality`: zwaardere instellingen voor langere runs en mogelijk betere prestaties.

Meerdere Snake-grids na elkaar trainen:

```bash
python DQN/train_snake_grids.py --grids 32 64 128 --episodes 10000 --profile balanced --device auto
```

Training hervat standaard automatisch vanaf het beste beschikbare checkpoint. De voorkeur is:

1. `DQN/checkpoints/<run_name>/best_eval.pth`
2. `DQN/checkpoints/<run_name>/latest.pth`

Gebruik `--fresh` wanneer je echt opnieuw wilt beginnen.

## Simulatie en Visualisatie

Simuleren zonder renderwindow:

```bash
python DQN/simulate.py --game snake --grid-size 32 --checkpoint auto --episodes 5
```

Met pygame-rendering:

```bash
python DQN/simulate.py --game snake --grid-size 32 --checkpoint auto --episodes 1 --render --fps 12
```

Met live web-feed:

```bash
python DQN/simulate.py \
  --game snake \
  --grid-size 32 \
  --checkpoint auto \
  --episodes 3 \
  --live-feed \
  --serve-live \
  --open-browser
```

Snake heeft ook een algoritmische benchmark-solver:

```bash
python DQN/simulate.py --game snake --grid-size 32 --solver hamiltonian --episodes 1 --render
```

Deze Hamiltonian-solver gebruikt geen DQN-checkpoint. Hij is nuttig als vergelijking met de geleerde agent.

`DQN/visualize.py` is een korte wrapper rond `simulate.py --render`:

```bash
python DQN/visualize.py --game flappy --episodes 1
```

## Snake Handmatig Spelen

Start handmatige Snake:

```bash
python Games/Snake/play.py --grid-size 32 --fps 12
```

Besturing:

- Pijltjestoetsen of `WASD`: richting kiezen.
- `Space` of `P`: pauzeren.
- `R`: reset.
- `Esc` of `Q`: afsluiten.

Je kunt dezelfde modus ook via `python main.py` starten met `play`.

## Checkpoints en Logs

Checkpoints worden opgeslagen in:

```text
DQN/checkpoints/<run_name>/
```

Voorbeelden:

```text
DQN/checkpoints/snake_32x32/latest.pth
DQN/checkpoints/snake_32x32/best_eval.pth
DQN/checkpoints/flappy/latest.pth
```

Logs worden opgeslagen in:

```text
DQN/logs/<run_name>/
```

Belangrijke CSV-bestanden:

- `metrics.csv`: trainingsmetrics per episode.
- `eval_metrics.csv`: evaluatiemetrics wanneer evaluatie actief is.

Voor Snake is de runnaam afhankelijk van de gridgrootte:

```text
snake_16x16
snake_32x32
snake_64x64
snake_128x128
snake_256x256
```

## Hoe De DQN-Pipeline Werkt

De DQN-code is opgezet rond een gedeelde interface voor alle games.

`GameEnvironment` laadt dynamisch de juiste game-logica uit `Games/<game>/logic/game_logic.py`. Daarna biedt de omgeving een stabiele RL-interface:

- `reset()`: start een nieuwe episode en geeft de begintoestand terug.
- `step(action)`: voert een actie uit en geeft `state`, `reward`, `done` en `info` terug.
- `action_space()`: geeft geldige actie-indexen terug.

De agent gebruikt epsilon-greedy action selection:

- Met kans `epsilon` kiest hij een willekeurige actie.
- Anders kiest hij de actie met de hoogste Q-waarde volgens het netwerk.
- `epsilon` daalt tijdens training, zodat de agent steeds minder random speelt.

Het Q-netwerk is een eenvoudige multilayer perceptron:

```text
Linear(input_size, hidden_size)
ReLU
Linear(hidden_size, hidden_size)
ReLU
Linear(hidden_size, action_count)
```

De training gebruikt onder andere:

- Double DQN
- target network updates
- replay memory
- prioritized replay
- gradient clipping
- checkpointing
- optionele evaluatie en `best_eval.pth`
- action masking voor Snake om onveilige acties te vermijden

De kernconfiguratie staat in `DQN/src/training/config.py`. Game-specifieke tuning gebeurt vooral in `DQN/train.py`.

## DQN-Variabelen Uitgelegd

Deze termen kom je vaak tegen in de code en trainingsoutput.

| Variabele/begrip | Betekenis |
| --- | --- |
| `state` | De huidige speltoestand als numerieke vector. Dit is wat de agent "ziet". |
| `action` | De keuze die de agent maakt, bijvoorbeeld rechtdoor, links of rechts bij Snake. |
| `reward` | Beloning of straf na een actie. Positief gedrag krijgt meestal een hogere reward. |
| `next_state` | De speltoestand na het uitvoeren van de actie. |
| `done` | Geeft aan of de episode afgelopen is, bijvoorbeeld door botsing of game over. |
| `episode` | Een volledige poging/run van begin tot eind. |
| `step` | Een enkele actie binnen een episode. |
| `score` | Game-score, bijvoorbeeld aantal gegeten appels of gepasseerde pipes. |
| `Q-value` | De geschatte toekomstige waarde van een actie in een bepaalde state. |
| `policy_net` | Het neurale netwerk dat de huidige Q-values voorspelt. |
| `target_net` | Stabieler kopienetwerk dat gebruikt wordt om trainingsdoelen te berekenen. |
| `epsilon` | Kans dat de agent random verkent in plaats van de beste bekende actie kiest. |
| `epsilon_start` | Beginwaarde van epsilon, meestal hoog zodat de agent veel exploreert. |
| `epsilon_end` | Minimumwaarde van epsilon, zodat er altijd een klein beetje exploratie blijft. |
| `epsilon_decay` | Factor waarmee epsilon langzaam daalt tijdens training. |
| `gamma` | Discount factor: bepaalt hoeveel toekomstige beloningen meetellen. |
| `learning_rate` | Hoe groot de update-stappen van het neural network zijn. |
| `batch_size` | Aantal ervaringen dat tegelijk uit replay memory wordt geleerd. |
| `memory_size` | Maximale grootte van replay memory. |
| `learning_starts` | Aantal stappen voordat training begint, zodat replay memory eerst gevuld wordt. |
| `learn_every_n_steps` | Hoe vaak het model leert tijdens gameplay. |
| `target_update_every_episodes` | Hoe vaak het target network wordt ververst. |
| `hidden_size` | Aantal neuronen in de verborgen lagen van het Q-network. |
| `max_steps_per_episode` | Maximum aantal stappen voordat een episode wordt afgekapt. |
| `eval_episodes` | Aantal episodes dat gebruikt wordt om de agent zonder training te evalueren. |
| `eval_every_episodes` | Hoe vaak evaluatie draait tijdens training. |
| `best_eval.pth` | Checkpoint met de beste evaluatieprestatie. |
| `latest.pth` | Meest recent opgeslagen checkpoint. |
| `prioritized_replay` | Replay-strategie waarbij belangrijkere ervaringen vaker worden gesampled. |
| `double_dqn` | DQN-variant die overschatting van Q-values vermindert. |
| `action_mask` | Filter dat ongeldige of gevaarlijke acties kan uitsluiten. |
| `device` | Hardware waarop PyTorch draait: `cpu`, `cuda` of `auto`. |

Kort samengevat leert DQN door tuples op te slaan:

```text
(state, action, reward, next_state, done)
```

Daarna sampled de trainer batches uit replay memory en traint het Q-network om betere Q-values te voorspellen.

## Game-Logica

Alle spelregels zitten bewust buiten de DQN-code:

- `Games/Snake/logic/game_logic.py`
- `Games/Flappy Bird/logic/game_logic.py`
- `Games/2048/logic/game_logic.py`

Daar horen onder andere:

- state-representatie;
- reward-functie;
- collision/terminal checks;
- score;
- reset/step-logica.

Deze scheiding maakt het mogelijk om dezelfde DQN-agent en trainer te gebruiken voor meerdere games.

## Configuratie

Snake-gridgrootte wordt op twee manieren bepaald:

1. expliciet via `--grid-size`;
2. via environment variable `SNAKE_GRID_SIZE`;
3. anders via de default `32`.

Voorbeeld:

```bash
SNAKE_GRID_SIZE=64 python DQN/train.py --game snake --episodes 10000
```

Training-device:

```bash
python DQN/train.py --game snake --device auto
python DQN/train.py --game snake --device cpu
python DQN/train.py --game snake --device cuda
```

Gebruik `auto` als standaard. Dan gebruikt het project CUDA wanneer PyTorch een GPU ziet, anders CPU.

## Veelvoorkomende Problemen

### Checkpoint ontbreekt

Bij simulatie met DQN moet het checkpoint bestaan. `--checkpoint auto` zoekt eerst naar `best_eval.pth` en daarna naar `latest.pth`.

Voor Snake 64x64 zoekt het project bijvoorbeeld hier:

```text
DQN/checkpoints/snake_64x64/best_eval.pth
DQN/checkpoints/snake_64x64/latest.pth
```

Train de grid eerst als beide ontbreken.

### Webinterface toont niets

De webinterface toont alleen echte live-feed data. Start een simulatie of training met live-feed, of gebruik de knoppen in de webinterface om een Snake-job te starten.

### Pygame-window opent niet

Controleer of `pygame` geinstalleerd is en of je omgeving grafische vensters ondersteunt. In headless servers werkt pygame-rendering vaak niet zonder extra displayconfiguratie.

### CUDA werkt niet

Controleer:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Als dit `False` is, gebruik `--device cpu` of installeer een PyTorch-build die past bij je CUDA/NVIDIA-driver.

## Aanbevolen Workflow

1. Gebruik `python main.py` om snel projectfuncties te ontdekken.
2. Train korte runs met `--profile fast` om instellingen te testen.
3. Train langere runs met `--profile balanced` of `--profile quality`.
4. Bekijk `DQN/logs/<run_name>/metrics.csv` en `eval_metrics.csv`.
5. Simuleer `best_eval.pth` via CLI of webinterface.
6. Bewaar interessante checkpoints voordat je grote nieuwe experimenten start.
