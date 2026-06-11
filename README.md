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
- [DQN-Architectuur in Detail](#dqn-architectuur-in-detail)
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

## VM Setup (Stap-voor-Stap)

Dit is hoe je het project in één keer op een schone VM installeert en start.

### 1. Project Klonen

```bash
cd ~
git clone https://github.com/joshuameuleman/AI-DeepLearning-Project.git
cd AI-DeepLearning-Project
```

### 2. Python-Vereisten Controleren

Het project vereist **Python 3.10 of 3.11**. Controleer je versie:

```bash
python3 --version
```

Als Python niet beschikbaar is of je hebt een ander versienummer, installeer Python 3.11:

**Op Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

**Op CentOS/RHEL:**
```bash
sudo yum install python3.11 python3.11-devel
```

### 3. Virtual Environment Aanmaken en Activeren

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Na activatie zie je `(.venv)` in je terminal.

### 4. Dependencies Installeren

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Opmerking:** De `requirements.txt` bevat PyTorch met CUDA 12.1 support. 
- Heb je **geen NVIDIA GPU**, dan werkt dit ook; training is dan alleen langzamer.
- Heb je een **ander GPU-model of CUDA-versie**, pas dan de PyTorch-regel in `requirements.txt` aan via https://pytorch.org/get-started/locally/

### 5. Installatie Verifiëren

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pygame; print('Pygame: OK')"
python -c "import numpy; print('NumPy: OK')"
```

Als alles groen wordt, ben je klaar!

### 6. Project Starten

**Optie A: Interactieve Launcher (aanbevolen voor eerste keer)**
```bash
python main.py
```

Kies dan een optie:
- `1. play` → Speel Snake handmatig
- `2. train` → Train een nieuw DQN-model
- `3. simulate` → Draai een bestaand model
- `4. visualize` → Visualiseer met pygame

**Optie B: Web-Interface (voor live meekijken)**
```bash
python serve_web.py --host 0.0.0.0 --port 8000
```

Open daarna je browser op: `http://127.0.0.1:8000/web/`

**Optie C: Direct Trainen (als je in de voorkeur weet wat je wilt)**
```bash
python DQN/train.py --game snake --grid-size 32 --episodes 5000 --profile balanced --device auto
```

### 7. Output en Checkpoints

Na training/simulatie staan alle resultaten hier:

```text
DQN/checkpoints/snake_32x32/latest.pth      # Meest recente model
DQN/checkpoints/snake_32x32/best_eval.pth   # Beste evaluatieprestatie
DQN/logs/snake_32x32/metrics.csv             # Trainingsstatistieken
DQN/logs/snake_32x32/eval_metrics.csv        # Evaluatiestatistieken
```

### Veelvoorkomende Problemen

**PyTorch CUDA niet beschikbaar**
```bash
python DQN/train.py --game snake --device cpu
```

**Port 8000 al in gebruik**
```bash
python serve_web.py --port 8001
```

**Virtual environment vergeten activeren?**
```bash
source .venv/bin/activate  # Linux/Mac
# of op Windows:
.venv\Scripts\activate
```

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

## DQN-Architectuur in Detail

Dit gedeelte legt uit hoe de DQN-trainingsloop precies werkt. Zorg dat je dit begrijpt voordat je voor je docent staat!

### Overall Training Flow

De trainingsloop (`DQN/src/training/trainer.py`) werkt als volgt:

```
1. Laad checkpoint (best_eval of latest) of start opnieuw
2. Per episode:
   a. Reset omgeving → krijg begintoestand
   b. Per stap in episode:
      - Agent kiest actie (epsilon-greedy)
      - Voer actie uit → krijg reward, next_state, done
      - Sla (state, action, reward, next_state, done) op in replay memory
      - Learn: sample batch uit replay memory en update Q-netwerk
   c. Na episode: update target network (elke N episodes)
3. Elke M episodes: run evaluatie met epsilon=0.0 (geen exploratie)
4. Controleer of evaluatieprestatie beter is, zo ja: sla best_eval.pth op
5. Herhal tot max episodes bereikt
```

### Core Components

**1. GameEnvironment** (`DQN/src/envs/game_env.py`)
- Laadt game-logica dynamisch uit `Games/<game>/logic/game_logic.py`
- Voert een uniforme interface uit voor alle games:
  ```python
  state = env.reset()
  outcome = env.step(action)  # → state, reward, done, info
  actions = env.action_space()
  ```
- Abstraheert game-specifieke details weg (Snake vs Flappy Bird vs 2048)

**2. QNetwork** (`DQN/src/models/q_network.py`)
- Eenvoudig 3-layer neural network:
  ```
  Input (state_size)
      ↓
  Linear(state_size → hidden_size) + ReLU
      ↓
  Linear(hidden_size → hidden_size) + ReLU
      ↓
  Linear(hidden_size → action_count)
      ↓
  Output (Q-value per actie)
  ```
- **Policy Net**: Huidige netwerk dat acties voorstelt
- **Target Net**: Kopie die minder vaak update, voor stabiliteit
- Beide worden op GPU/CPU geplaatst via `config.device`

**3. DQNAgent** (`DQN/src/agents/dqn_agent.py`)
- Beheren epsilon-greedy exploration:
  ```python
  if random() < epsilon:
      action = random_choice(action_space)
  else:
      action = argmax(policy_net.forward(state))
  ```
- `epsilon` start hoog (veel verkenning) en daalt naar `epsilon_end`
- Decay formule: `epsilon = max(epsilon_end, epsilon_start * decay^(step))`

**4. ReplayMemory** (`DQN/src/agents/replay_memory.py`)
- Slaat ervaringen `(state, action, reward, next_state, done)` op
- Maximale grootte: `memory_size` (meestal 10k - 100k)
- Bij volheid: verwijder oudste ervaringen (FIFO)
- **Prioritized Replay**: Sample vaker ervaringen met hoge TD-error
  - TD-error = `|target_value - predicted_value|`
  - Belangrijke ervaringen = grote leerwaarde

**5. Trainer Loop** (`DQN/src/training/trainer.py`)

Per stap gebeurt:

```python
# 1. Selecteer actie
action = agent.select_action(state, policy_net, device)

# 2. Voer actie uit
outcome = env.step(action)
state, reward, done, info = outcome

# 3. Sla ervaringen op
memory.push(state, action, reward, next_state, done)

# 4. Learn (elke learn_every_n_steps)
if total_steps % learn_every_n_steps == 0:
    batch = memory.sample(batch_size)  # ← Prioritized replay hier
    
    # Double DQN update:
    # 1. Policy net kiest beste actie op next_state
    next_actions = policy_net(next_states).argmax(dim=1)
    
    # 2. Target net evalueert die acties
    target_values = target_net(next_states).gather(1, next_actions)
    
    # 3. Berekenen target: reward + gamma * target_value (als niet done)
    target = reward + gamma * target_values * (1 - done)
    
    # 4. Voorspellen huidge Q-waarden
    predicted = policy_net(states).gather(1, actions)
    
    # 5. MSE loss en backprop
    loss = (predicted - target).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net, 1.0)  # Clip grads
    optimizer.step()
    
    # Update priorities in replay memory
    memory.update_priorities(td_errors)

# 5. Update target network (elke target_update_every_episodes)
if episode % target_update_every_episodes == 0:
    target_net.load_state_dict(policy_net.state_dict())
```

### Double DQN Uitleg

Normaal DQN heeft een probleem: het **overschat** Q-values omdat:

```
Normale DQN:
  best_action = argmax(Q(s_next))  ← Policy net kiest
  target = r + gamma * max(Q(s_next))  ← Target net evalueert hetzelfde

Dit leidt tot optimistische schattingen!
```

**Double DQN** (wat wij gebruiken):

```
Double DQN:
  best_action = argmax(policy_net(s_next))  ← Policy net kiest
  target = r + gamma * target_net(s_next)[best_action]  ← Target net evalueert

Dit is realistischer omdat twee netwerken deelnemen.
```

Zie `_masked_next_policy_q()` in [DQN/src/training/trainer.py](DQN/src/training/trainer.py) voor implementatie.

### Prioritized Replay

Normaal sample je willekeurig uit replay memory. Maar sommige ervaringen zijn belangrijker dan anderen!

**TD-Error** (Temporal Difference Error):
```
TD-error = |target_Q - predicted_Q|
  - Hoog = veel geleerd
  - Laag = already fits well (minder nuttig)
```

**Prioritized Replay**:
- Compute TD-error na elke update
- Slaa TD-errors op samen met experiences
- Sample `high_TD_error` experiences vaker
- Dit versnelt leren omdat je focust op "moeilijke" cases

Implementatie in [DQN/src/agents/replay_memory.py](DQN/src/agents/replay_memory.py) met `PrioritizedReplayMemory` klasse.

### Action Masking voor Snake

Snake heeft vaak situaties waar bepaalde acties **onveilig** zijn:
- Je bent naar rechts aan het gaan → je kunt niet onmiddellijk naar links
- Je raakt jezelf!

**Hoe masking werkt**:

```python
# Extraheer eerste 3 state-waarden (collision detection)
is_wall_left = state[0] > 0.5
is_wall_straight = state[1] > 0.5
is_wall_right = state[2] > 0.5

safe_actions = [not is_wall_left, not is_wall_straight, not is_wall_right]

# Mask onveilige acties
Q_values = policy_net(state)
Q_values[~safe_actions] = -1e9  # Maak onveilige acties zeer negatief

action = argmax(Q_values)  # Kiest nu enkel veilige acties
```

Dit voorkomt dat de agent veel tijd verspilt met botsingen en verbetert leren.

Zie `_action_mask_from_state()` en `_masked_next_policy_q()` in [DQN/src/training/trainer.py](DQN/src/training/trainer.py).

### Checkpointing & Evaluation

De trainer slaat regelmatig modellen op:

```
DQN/checkpoints/<run_name>/
  ├── latest.pth          ← Meest recent (elke episode)
  └── best_eval.pth       ← Beste evaluatieprestatie
```

**Evaluatie** (elke `eval_every_episodes`):
- Draai `eval_episodes` met epsilon=0.0 (geen exploratie)
- Meet: gemiddelde reward, score, aantal stappen, type eindresultaat
- Als avg_score/avg_steps/avg_reward beter is dan vorige beste:
  - Sla beste model op als `best_eval.pth`
  - Log alles naar `eval_metrics.csv`

Dit zorgt dat je altijd het beste model hebt, niet alleen het meest recente.

### Training Configuration

Alle hyperparameters staan in [DQN/src/training/config.py](DQN/src/training/config.py) en kunnen via CLI gekozen worden:

```bash
python DQN/train.py \
  --game snake \
  --grid-size 32 \
  --episodes 10000 \
  --profile balanced \
  --device auto
```

Game-specifieke defaults staan in [DQN/train.py](DQN/train.py):
- Snake `fast`: lage hidden_size, korte episodes (snel testen)
- Snake `balanced`: gemiddelde instellingen (standaard)
- Snake `quality`: hogere hidden_size, langere training (betere resultaten)

### Debugging Tips

Voor je docent:

1. **Kijk in logs**:
   ```bash
   head -n 20 DQN/logs/snake_32x32/metrics.csv
   head -n 20 DQN/logs/snake_32x32/eval_metrics.csv
   ```
   Je ziet: episode rewards, epsilon decay, evaluatieprestatie

2. **Check checkpoint grootte**:
   ```bash
   ls -lh DQN/checkpoints/snake_32x32/
   ```
   ~1-2 MB is normaal (alleen network weights, geen optimizer state)

3. **Train een korte run**:
   ```bash
   python DQN/train.py --game snake --episodes 100 --profile fast
   ```
   Zie je rewards stijgen? Dan werkt het systeem!

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
