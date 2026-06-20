# TypeScript Robot UI

This folder contains the Vite + React + TypeScript frontend. The production build is served by the FastAPI backend in `../api_app.py`.

The robot draw path is not implemented in the browser. The TypeScript UI posts draw jobs to `/api/machine/draw`, and the Python backend uses the V3.0 `stream_gcode_to_bridge(...)` machine-control path with conservative send settings, GRBL acknowledgement tracking, transient ESP32 bridge recovery, automatic overlap resume, pen-up repositioning before replay, bridge cooldown, health scoring, optional ESP32 bridge restart, and replay for uncertain batches. Lost/recovered connection counts, ESP32 reboot count, latency, and bridge quality are reported through the normal job-progress poll and do not add extra ESP32 requests. The draw stream uses short HTTP command timeouts and pauses on repeated timeout-recovered commands, so a weak ESP32 connection cannot keep drawing one move per timeout in degraded slow mode. Home controls are locked out while a draw is active; a direct Home request during a draw first sends feed-hold and cancels the job instead of sending `$H`, and Home is only sent when GRBL reports Idle or Alarm.

## Run for Robot Operation

From the repository root:

```bash
source .venv/bin/activate
uvicorn api_app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

If port `8000` is already busy, choose another port:

```bash
uvicorn api_app:app --reload --port 8001
```

Then open `http://127.0.0.1:8001`.

Keep that terminal open while drawing. Press `CTRL+C` in the same terminal to close the WebUI.

## Edit the Frontend

Install Node.js first. Run the FastAPI backend from the repository root:

```bash
source .venv/bin/activate
uvicorn api_app:app --reload
```

Then run Vite from this folder:

```bash
npm install
npm run dev
```

Vite will print a local development URL, usually:

```text
http://localhost:5173
```

The Vite dev server proxies `/api` requests to `http://127.0.0.1:8000`.

If you run FastAPI on another port, pass it to Vite:

```bash
VITE_API_PROXY=http://127.0.0.1:8001 npm run dev
```

Before using the edited frontend through `api_app.py`, rebuild the static files:

```bash
npm run build
```
