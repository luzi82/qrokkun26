# Qrokkun26

A tiny 2D bullet-dodging survival game. Homage to the 1999 freeware **特訓99** (Tokkun 99), with **original** art, titles, and UI (nothing is copied from 特訓99).

**Made by Grok Bot.** Tribute to 特訓99.

You are a small ship on a cramped playfield. You cannot shoot. Survive as long as you can against nearly-random bullets that get denser and faster over time.

## How to open in Godot 4

1. Install [Godot 4.7+](https://godotengine.org/download) (4.3 or newer should open the project).
2. In Godot, **Import** / **Open** and select `project.godot` in this folder.
3. Press **F5** (Play).

Internal resolution is **320x240**, integer-scaled to a **960x720** window (stretch mode `viewport` + integer scale). Nearest-neighbor filtering. Physics ticks at 60 Hz.

## How to play

| Action | Keys |
|--------|------|
| Move | Arrow keys **or** WASD |
| Start / restart | **Enter**, **Space**, or **Z** |

- Hitbox is slightly smaller than the ship sprite. Contact with a bullet is death.
- In-play HUD shows elapsed time only. That time **is** the score.
- One endless wave. Some bullets come from the edges; some are aimed *near* you (not perfect homing, not memorization patterns).
- From the game-over screen, Enter / Space / Z starts a new run.

### Rank titles

30 seconds is a brag. Around 60 seconds is elite.

| Time | Title |
|------|-------|
| 0s | SPARK |
| 5s | GLINT |
| 10s | NEEDLE |
| 15s | RAZOR |
| 22s | ARC |
| **30s** | **GROKKER** |
| 40s | STORMCUT |
| 50s | IRONWAKE |
| **60s** | **APEX** |
| 75s | MYTHOS |
| 90s | VOIDPILOT |
| 120s | ETERNAL |

## Export (Windows / Linux)

`export_presets.cfg` already defines **Linux** and **Windows Desktop** presets.

1. Godot **Editor -> Manage Export Templates...** and download templates matching your Godot version.
2. **Project -> Export...**
3. Select **Linux** -> Export to `dist/Qrokkun.x86_64`.
4. Select **Windows Desktop** -> Export to `dist/Qrokkun.exe`.

If templates are missing, Godot cannot bake standalone binaries. The presets are still in the project so you can export after installing templates.

This packaged copy does **not** include baked `dist/` binaries unless templates were available on the machine that built it. See `dist/README.txt`.

Command-line example (templates required):

```bash
godot --headless --path . --export-release "Linux" dist/Qrokkun.x86_64
godot --headless --path . --export-release "Windows Desktop" dist/Qrokkun.exe
```

## License

MIT. See `LICENSE`.

Made by Grok Bot. Tribute to 特訓99.

## RL environment (Python)

`qrokkun_env/` is a Godot-free reimplementation of the playfield logic for training:

```bash
python3 -m pytest qrokkun_env/tests qrokkun_ai -q
python3 -c "from qrokkun_env import Qrokkun26Env; e=Qrokkun26Env(seed=0); print(e.reset()); print(e.step(0))"
```

See [docs/ROADMAP.md](docs/ROADMAP.md).

## Playable Gv20 and engine parity

Normal F5 / project launch now uses scripted Gv20. Title, score, hit animation,
and restart remain available. Player speed is 102 with immediate stopping.
All bullets have radius 2.6 and constant velocity. Spawns use a uniform ray
from (160,126) onto x=0..320, y=12..240, independent Gaussian target offsets
(sigma 13.2 per axis), and uniform speed 30..200. The shifted accumulator
starts at frame 0,27,53,79,104; no legacy double spawn or kind effects apply.
Every positive multiple of 60 adds a directly aimed field bullet using the
mean S speed from [frame-60,frame), excluding same-frame S.

Spawning precedes player movement. The reset bullet stays still on counted tick 1 and first moves on tick 2; bullets
added during a tick first move on the following tick. Collision checks use
current poses after movement, with player radius 4 plus bullet radius 2.6,
including newly spawned bullets. Godot Area2D signals lag this rule, so Gv20
resolves overlaps explicitly after movement.

Python string-seeded random.Random and Godot RNG **do not produce equivalent
streams**. Godot has separate S, field, and player RNG objects; QROKKUN_SEED
reproduces Godot runs only. The engine tests record explicit random variates
from real stochastic production spawns, replay them through Python Gv20, and
can inject them into the same production Godot spawner. No learned Sv22 is used.

Run with an existing Python environment containing pytest, numpy, and torch:

```bash
GODOT_BIN=/path/to/godot PYTHONPATH=. python -m pytest tools/test_gv20_godot.py -q
```

The tests use GODOT_BIN, then discover godot or godot4 on PATH; missing Godot
produces an explicit integration-test skip. They import assets before running and
check a fresh project import for parse errors. They run
the actual main scene headlessly, covers three seeds, moving and idle deaths,
actual input-driven title/restart with held movement, death and cleared restart
UI/bullet state, a 7200-tick stochastic schedule, field windows, independent
player draws, and explicit replay. Long schedule runs disable only the collision
resolver in the harness to survive 120 seconds; ordinary replay runs retain it.
The harness also supports GV20_CAPTURE for a real viewport PNG when a display
is accessible. Headless physics tests provide no graphical/manual coverage.

Python legacy Qrokkun26Env, v10 models, and v21 observation/checkpoint contracts
remain intact. The historical monkey fixture and monkey_verify.py describe the
legacy game and should not be used to validate new Gv20 recordings.

### Fixed-rule sanity check

```bash
PYTHONPATH=. python3 -m qrokkun_env.sanity --policy flee --seeds 20 --compare
```

`flee` (away from nearest bullet) vs scripted spawner; `--compare` also runs `idle`.

### Train PlayerV10 (120s cap, zero-inertia physics)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-train.txt
PYTHONPATH=. .venv/bin/python -m qrokkun_ai.v10.train.train_player_v10 --updates 300 --out-dir runs/player_v10
```

Legacy `v0`..`v5` (inertia-era physics) are archived at git tag `legacy-inertia-physics-v5`.

## Project journal

See the [Gv20 Godot integration and verification record](docs/journal/2026-11-10-1794266611-gv20-godot-integration-and-verification.txt) for delivered scope, fixes, verification, and limitations.

Store durable project records in [docs/journal](docs/journal/README.md) as `yyyy-mm-dd-unixtimestamp-topic.txt`, with hyphen-separated topics, actual record creation Unix seconds, and the matching `Asia/Hong_Kong` calendar date. `README.md` is exempt; see the journal README for migration provenance.

Gv20 diagnostic events are disabled by default; the parity harness opts in.
Production speed history retains only the required 60-frame window. The memory
regression runs 7200 ticks without observer clearing. Frame-zero transitions
move neither Player nor the reset bullet. Python Gv20 matches HEAD unchanged.
Fix-pass RED/GREEN evidence is in tmp/gv20/fixpass/; original evidence is retained.
No display was available for this pass, so renderer/manual evidence is outstanding.

Engine subprocesses, including imports, use temporary user-data, config, and
cache directories. Score persistence is verified inside the isolated profile;
input-driven restarts clear the flash timer and alpha at frame zero. Final
RED/GREEN evidence and the concise report are in `tmp/gv20/finalfix/`.
Run the full check with:

```bash
GODOT_BIN=/path/to/godot PYTHONPATH=. python -m pytest tools/test_gv20_godot.py qrokkun_env/tests qrokkun_ai/v10/tests qrokkun_ai/v21/tests -q
```

Sv22 owner-authorized learning experiment: [protocol and local runner](qrokkun_ai/v22/README.md).

Sv22 review operations: see [runner README](qrokkun_ai/v22/README.md) for finite
deadline enforcement, exact formal seed counts, live validation snapshots,
completed-update progress/status, and primary-result preservation before optional
diagnostics. CPU review evidence is in `tmp/sv22/review-fixes/`.

Sv22 owner-stop outcome: [historical training stop report](docs/journal/2026-11-10-1794311518-sv22-training-stop-report.txt).

Sv23 fixed-u1000 owner contract: [policy, tests, and launch handoff](qrokkun_ai/v23/README.md).

Sv23 owner-stop outcome: [training stop report](docs/journal/2026-11-11-1794351856-sv23-training-stop-report.txt).

Sv24 已完成 1000 updates：[訓練完成與正式最終結果報告](docs/journal/2026-11-11-1794395723-sv24-training-completion-report.txt)。固定驗證選出 **best u980**（66.6302 秒），u1000 驗證為 70.50625 秒；正式 heldout 使用 u980，平均玩家存活 69.7685 秒，低於三組對照（對 S 而言越低越好）。

![Sv24 完成 1000 updates 的 rollout 與固定驗證曲線，best 為 u980](docs/journal/assets/sv24-completed/progress.png)

訓練與驗證圖採非零起點放大 y 軸：rollout 55–105 秒、validation 60–100 秒。Best u980 與終點 u1000 為不同 checkpoint。

![Sv24 best u980 的正式 heldout 均值及對照配對差值 95% CI](docs/journal/assets/sv24-completed/final.png)

正式 heldout：每組 64 seeds；均值軸放大為 65–90 秒，差值軸為 −25–5 秒。CI 表示 Sv24 減對照的配對平均差，三組均低於零。這是單一 training seed、固定 Pv21 u400 的結果；該報告當時保存的全套測試快照為 18 failed、134 passed、26 skipped，歷史結果保留於報告。

Sv22 歷史期限測試維護（實作 `8604090`）：[維護回顧與重現證據](docs/journal/2026-11-11-1794397742-sv22-test-deadline-cleanup-report.txt)。修復後獨立重跑全套測試為 **164 passed、26 skipped（35.26s）**；26 skips 不算通過。此為測試維護結果，與上述 Sv24 訓練完成結果分開記錄，正式歷史 cutoff 未放寬。

Pv25 C warm-start 實驗（等待獨立 review／verified smoke／launch）：[owner 合約](docs/journal/2026-11-11-1794400966-pv25-warm-start-contract.txt)、[v25 runner、測試與 parent 啟動命令](qrokkun_ai/v25/README.md)。原 Pv21 u400 actor/critic 對固定 Sv24 best u980，fresh Adam、1000 completed updates；尚無 Pv25 正式結果。
