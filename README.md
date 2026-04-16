# MACRS — Multi-Agent Conversational Recommender System

Implementation of the MACRS architecture from [arxiv.org/abs/2402.01135](https://arxiv.org/abs/2402.01135). A team of specialized agents collaborate to manage goal-directed dialogue in conversational recommendation.

Uses [LiteLLM](https://github.com/BerriAI/litellm) — swap between OpenAI, Anthropic, Ollama, or any supported provider with a single flag.

## Architecture

```
User message
    │
    ▼
┌─────────────────── Reflection Phase ───────────────────┐
│  1. Detect feedback signal (rejection/disengagement/+)  │
│  2. Information-level reflection → update user profile   │
│  3. Strategy-level reflection → generate agent advice    │
│     (only on failure signals)                            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────── Generation Phase ───────────────────┐
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐         │
│  │  Asking   │  │ Recommending │  │ Chitchat │         │
│  │  Agent    │  │    Agent     │  │  Agent   │         │
│  │          │  │ (tool use:   │  │          │         │
│  │          │  │  search_     │  │          │         │
│  │          │  │  movies)     │  │          │         │
│  └────┬─────┘  └──────┬───────┘  └────┬─────┘         │
│       └───────────┬────┴───────────────┘               │
│                   ▼                                     │
│            ┌─────────────┐                              │
│            │   Planner   │  ← corrective experience    │
│            │    Agent    │                              │
│            └──────┬──────┘                              │
└───────────────────┼─────────────────────────────────────┘
                    ▼
              System output
```

### Agents

| Agent | Role | Tools |
|-------|------|-------|
| **Asking** | Elicit user preferences through targeted questions | — |
| **Recommending** | Suggest items from the catalog with engaging descriptions | `search_movies` (function calling) |
| **Chitchat** | Keep conversation natural, subtly steer toward preference discovery | — |
| **Planner** | Select the best candidate response based on dialogue strategy | — |

### Reflection Mechanism

The system adapts in real-time through two levels of reflection:

**Information-level** (every turn): Extracts preferences, liked/disliked items from the user's message and updates the shared user profile.

**Strategy-level** (on failure signals only): When the system detects rejection, disengagement, or repetition, it generates specific corrective advice for each agent:

```
Signal: rejection — Rejected recommendation: 'not really into romantic movies'
  [info] disliked=['Elemental'], prefs={'avoid_genre': 'romance'}
  [strategy] User rejected romance — pivot to action or comedy
    → asking: Ask about preferred mood instead of genre
    → recommending: Avoid romance, try animation+comedy
    → planner: Prefer asking over recommending for next turn
```

These strategy hints are injected into agent prompts on subsequent turns, forming a closed-loop adaptation system.

## Setup

```bash
git clone https://github.com/ZahrizhalAli/macrs.git
cd macrs
pip install -e .
```

### Download MovieLens dataset

```bash
curl -L -o /tmp/ml-latest-small.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip /tmp/ml-latest-small.zip -d data/
```

This gives you 9,742 movies with genres and years. The recommending agent searches this catalog via function calling — it decides what to query, not the system.

### API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

Or for other providers:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Default (gpt-4o-mini)
macrs

# Pick a model
macrs --model gpt-4o-mini
macrs --model claude-sonnet-4-6
macrs --model ollama/llama3

# Show all 3 agent candidates + reflection logs
macrs --verbose

# Full debug logging
macrs --debug

# Custom catalog path
macrs --catalog /path/to/movies.csv
```

### Example session (with `--verbose`)

```
You: I want to watch a relaxing animated film

--- Reflection (turn 1) ---
  Signal: neutral — No strong signal detected
  [info] prefs={'genre': 'animation', 'mood': 'relaxing'}
  [info] User wants relaxing animated content
  Profile: Preferences: genre=animation, mood=relaxing
-------------------------------

--- Candidates ---
  [asking]: Are you looking for classic animation or something more recent?
  [recommending]: How about Spirited Away? It's a beautifully calm fantasy...
  [chitchat]: There's something special about animated films for unwinding...
------------------

MACRS [asking]: Are you looking for classic animation or something more recent?

You: recent ones, and not romantic

--- Reflection (turn 2) ---
  Signal: rejection — Negative signal: 'not romantic'
  [info] prefs={'year_preference': 'recent', 'avoid_genre': 'romance'}
  [info] User prefers recent films and dislikes romance
  [strategy] User explicitly excluded romance — narrow to non-romantic animation
    → asking: Ask about preferred sub-genre (comedy, action, fantasy)
    → recommending: Search animation post-2015, exclude romance tags
    → planner: User gave clear preferences — favor recommending next
  Profile: Preferences: genre=animation, mood=relaxing, year_preference=recent, avoid_genre=romance
-------------------------------
```

## Project structure

```
macrs/
├── pyproject.toml
├── .env                     # API keys (git-ignored)
├── data/
│   ├── ml-latest-small/     # MovieLens dataset (git-ignored)
│   └── redial/              # ReDial CRS dataset (git-ignored, for future eval)
└── macrs/
    ├── models.py            # Message, UserProfile, DialogueHistory, CandidateResponse
    ├── catalog.py           # Item, ItemCatalog, MovieLens loader
    ├── config.py            # LLMConfig
    ├── tools.py             # search_movies tool schema + executor
    ├── reflection.py        # Feedback detection, info/strategy reflection, log
    ├── engine.py            # MACRSEngine — per-turn orchestration loop
    ├── main.py              # CLI entry point
    └── agents/
        ├── base.py          # ResponderAgent with tool-use loop + strategy hints
        ├── asking.py        # Preference elicitation
        ├── recommending.py  # Item recommendation (with search_movies tool)
        ├── chitchat.py      # Engagement and natural conversation
        └── planner.py       # Candidate selection with corrective experience
```

## Paper

Based on: [Multi-Agent Conversational Recommender System](https://arxiv.org/abs/2402.01135) (2024).
