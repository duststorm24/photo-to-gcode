import {
  Activity,
  Cable,
  CheckCircle2,
  Cpu,
  Download,
  FileImage,
  Gauge,
  Grid3X3,
  Home,
  ImagePlus,
  Layers3,
  Lock,
  Move,
  OctagonAlert,
  PenLine,
  Play,
  RefreshCcw,
  RotateCw,
  Route,
  Ruler,
  SlidersHorizontal,
  Sparkles,
  SquarePen,
  TerminalSquare,
  Trash2,
  Unlock,
  Upload,
  Wand2,
  Wifi
} from "lucide-react";
import {
  ChangeEvent,
  PointerEvent as ReactPointerEvent,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import type {
  DimensionUnit,
  DrawJob,
  GenerationMode,
  GenerationResult,
  MachineLogSnapshot,
  MachineSettings,
  MachineState,
  Placement,
  WorkflowSettings
} from "./types";

const APP_VERSION = "V3.0";
const PAGE_WIDTH_MM = 215.9;
const PAGE_HEIGHT_MM = 279.4;
const MM_PER_IN = 25.4;
const MM_PER_CM = 10;
const LAST_DRAW_JOB_STORAGE_KEY = "photoToGcode:lastDrawJobId";

const initialPlacement: Placement = {
  xMm: 28,
  yMm: 44,
  widthMm: 160,
  heightMm: 120,
  rotationDeg: 0
};

const initialWorkflowSettings: WorkflowSettings = {
  threshold: 165,
  invertInput: false,
  marginMm: 0,
  maskResolutionPxMm: 18,
  lineWidthMm: 0.05,
  wallLines: 1,
  infillDensityPercent: 100,
  drawSpeedMmSec: 120,
  travelSpeedMmSec: 120,
  fillStrategy: "continuous_zigzag",
  fillTurnSplitAngleDeg: 20,
  continuousFillChunkSegments: 0,
  pathSimplifyToleranceMm: 0.08,
  minSegmentLengthMm: 0.1,
  minToolpathLengthMm: 0.1,
  coordinateDecimals: 3
};

const initialMachineSettings: MachineSettings = {
  bridgeUrl: "http://10.0.0.90",
  timeoutSeconds: 8,
  penUpGapMm: 8,
  queueWindowSize: 24,
  batchAckTimeoutSeconds: 90,
  maxInFlight: 1,
  sendSpacingMs: 6,
  recoveryTimeoutSeconds: 180,
  bridgeRecoveryCooldownSeconds: 5,
  bridgeHealthMaxLatencyMs: 1200,
  bridgeRestartEnabled: true,
  bridgeRestartWaitSeconds: 12,
  autoResumeEnabled: true,
  autoResumeRewindCommands: 6,
  autoResumeMaxAttempts: 20,
  autoResumeRetryDelaySeconds: 5,
  usePenAxis: true,
  penAxis: "Z",
  penUpPositionMm: 20,
  penDownPositionMm: 28,
  penFeedRateMmMin: 7200,
  penUpDwellSeconds: 0,
  penDownDwellSeconds: 0
};

const modeMeta: Record<GenerationMode, { label: string; note: string }> = {
  vector_trace: {
    label: "Vector Trace",
    note: "V1.4 local planner, using the proven bridge stream path"
  },
  cura_slice: {
    label: "Cura Slice",
    note: "Uses the local CuraEngine pipeline when available"
  },
  triangle_mesh: {
    label: "Triangle Mesh",
    note: "Angular tonal drawing style"
  }
};

type DragState =
  | { kind: "move"; offsetX: number; offsetY: number }
  | { kind: "resize"; startAspect: number }
  | null;

export function App() {
  const [sourceImage, setSourceImage] = useState<string | null>(null);
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [sourceName, setSourceName] = useState("No image loaded");
  const [sourceAspect, setSourceAspect] = useState(initialPlacement.widthMm / initialPlacement.heightMm);
  const [aiComments, setAiComments] = useState("");
  const [aiResult, setAiResult] = useState<{ imageData: string; imageName: string } | null>(null);
  const [isSavingAiImage, setIsSavingAiImage] = useState(false);
  const [aiSaveMessage, setAiSaveMessage] = useState("");
  const [mode, setMode] = useState<GenerationMode>("vector_trace");
  const [settings, setSettings] = useState<WorkflowSettings>(initialWorkflowSettings);
  const [machine, setMachine] = useState<MachineSettings>(initialMachineSettings);
  const [machineState, setMachineState] = useState<MachineState>("Disconnected");
  const [placement, setPlacement] = useState<Placement>(initialPlacement);
  const [dimensionUnit, setDimensionUnit] = useState<DimensionUnit>("in");
  const [lockAspect, setLockAspect] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [generation, setGeneration] = useState<GenerationResult | null>(null);
  const [drawJob, setDrawJob] = useState<DrawJob | null>(null);
  const [resumeRewind, setResumeRewind] = useState(6);
  const [isPlanning, setIsPlanning] = useState(false);
  const [isAiConverting, setIsAiConverting] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [machineHomed, setMachineHomed] = useState(false);
  const [penMessage, setPenMessage] = useState("Home all, jog the pen to paper contact, then save contact.");
  const [trackedPenZMm, setTrackedPenZMm] = useState<number | null>(null);
  const [penConfirmed, setPenConfirmed] = useState(false);
  const [machineLog, setMachineLog] = useState<string[]>([]);
  const [logMessage, setLogMessage] = useState("Machine log has not been refreshed yet.");
  const [isLogLoading, setIsLogLoading] = useState(false);
  const [dragState, setDragState] = useState<DragState>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const stats = useMemo(() => {
    if (generation) {
      return {
        paths: generation.metrics.path_count,
        commands: generation.totalCommands,
        drawSeconds: generation.estimatedSeconds,
        commandsPerSecond: generation.commandsPerSecond,
        page: "Letter 216 x 279 mm"
      };
    }
    return {
      paths: 0,
      commands: 0,
      drawSeconds: 0,
      commandsPerSecond: 4,
      page: "Letter 216 x 279 mm"
    };
  }, [generation]);

  useEffect(() => {
    const storedJobId = window.localStorage.getItem(LAST_DRAW_JOB_STORAGE_KEY);
    if (!storedJobId) return;

    let canceled = false;
    void (async () => {
      try {
        const response = await fetch(`/api/jobs/${storedJobId}`);
        if (!response.ok) return;
        const storedJob = (await response.json()) as DrawJob;
        if (!canceled && storedJob.resume_available) {
          setDrawJob(storedJob);
          setMachineState("Alarm");
          setErrorMessage(storedJob.error || storedJob.message);
        }
      } catch {
        // Stored job recovery is best effort; normal drawing still works without it.
      }
    })();

    return () => {
      canceled = true;
    };
  }, []);

  useEffect(() => {
    if (drawJob?.id) {
      window.localStorage.setItem(LAST_DRAW_JOB_STORAGE_KEY, drawJob.id);
    }
  }, [drawJob?.id]);

  useEffect(() => {
    if (!drawJob || (drawJob.status !== "queued" && drawJob.status !== "running")) {
      return;
    }

    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/jobs/${drawJob.id}`);
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const nextJob = (await response.json()) as DrawJob;
        setDrawJob(nextJob);
        if (nextJob.active_bridge_url && nextJob.active_bridge_url !== machine.bridgeUrl) {
          setMachine((current) => ({ ...current, bridgeUrl: nextJob.active_bridge_url || current.bridgeUrl }));
        }
        if (nextJob.status === "complete") setMachineState("Idle");
        if (nextJob.status === "canceled") setMachineState("Hold");
        if (nextJob.status === "error") {
          setMachineState("Alarm");
          setErrorMessage(nextJob.error || nextJob.message);
          if (nextJob.recent_log?.length) {
            setMachineLog(nextJob.recent_log);
            setLogMessage("Captured from the bridge when the draw stopped.");
          }
        }
      } catch (error) {
        setMachineState("Disconnected");
        setErrorMessage(error instanceof Error ? error.message : "Could not poll draw job.");
      }
    }, 900);

    return () => window.clearInterval(timer);
  }, [drawJob, machine.bridgeUrl]);

  function handleImageUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    void loadSourceFile(file);
  }

  async function loadSourceFile(file: File, displayName = file.name, options: { preserveAiResult?: boolean } = {}) {
    const objectUrl = URL.createObjectURL(file);
    const aspect = await readImageAspect(objectUrl);
    setSourceName(displayName);
    setSourceImage(objectUrl);
    setSourceFile(file);
    setSourceAspect(aspect);
    setGeneration(null);
    setDrawJob(null);
    setErrorMessage("");
    window.localStorage.removeItem(LAST_DRAW_JOB_STORAGE_KEY);
    if (!options.preserveAiResult) {
      setAiResult(null);
      setAiSaveMessage("");
    }
    const widthMm = Math.min(170, PAGE_WIDTH_MM - 20);
    const heightMm = widthMm / aspect;
    setPlacement(centerPlacement({ ...initialPlacement, widthMm, heightMm }));
  }

  async function convertToAi() {
    if (!sourceFile) {
      setErrorMessage("Upload an image before using AI conversion.");
      return;
    }
    setIsAiConverting(true);
    setErrorMessage("");
    try {
      const formData = new FormData();
      formData.append("image", sourceFile);
      formData.append("additionalComments", aiComments);
      const response = await fetch("/api/ai/convert", { method: "POST", body: formData });
      if (!response.ok) throw new Error(await response.text());
      const payload = (await response.json()) as { imageData: string; imageName: string };
      const aiBlob = await fetch(payload.imageData).then((blobResponse) => blobResponse.blob());
      const aiFile = new File([aiBlob], payload.imageName, { type: "image/png" });
      setAiResult({ imageData: payload.imageData, imageName: payload.imageName });
      setAiSaveMessage("AI image ready to save.");
      void loadSourceFile(aiFile, payload.imageName, { preserveAiResult: true });
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Could not convert the image with AI.");
    } finally {
      setIsAiConverting(false);
    }
  }

  async function saveAiImageToDesktop() {
    if (!aiResult) {
      setAiSaveMessage("Convert an image with AI first.");
      return;
    }
    setIsSavingAiImage(true);
    setErrorMessage("");
    try {
      const response = await fetch("/api/ai/save-to-desktop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(aiResult)
      });
      const payload = (await response.json()) as { ok: boolean; fileName?: string; path?: string; detail?: string };
      if (!response.ok || !payload.ok) throw new Error(payload.detail || response.statusText);
      setAiSaveMessage(`Saved ${payload.fileName || aiResult.imageName} to Desktop.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not save the AI image.";
      setAiSaveMessage(message);
      setErrorMessage(message);
    } finally {
      setIsSavingAiImage(false);
    }
  }

  function updateSetting<Key extends keyof WorkflowSettings>(key: Key, value: WorkflowSettings[Key]) {
    setSettings((current) => ({ ...current, [key]: value }));
    setGeneration(null);
  }

  function updateMachine<Key extends keyof MachineSettings>(key: Key, value: MachineSettings[Key]) {
    setMachine((current) => ({ ...current, [key]: value }));
    if (key === "bridgeUrl") {
      setMachineHomed(false);
      setPenConfirmed(false);
      setTrackedPenZMm(null);
    }
    if (key.toString().startsWith("pen") || key === "usePenAxis") {
      setPenConfirmed(false);
    }
  }

  function updatePlacement(nextPlacement: Placement) {
    setPlacement(clampPlacement(nextPlacement));
    setGeneration(null);
  }

  function rotateArtworkClockwise() {
    updatePlacement(rotatePlacementClockwise(placement));
  }

  function pointerToPage(event: ReactPointerEvent<SVGSVGElement>) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return { xMm: 0, yMm: 0 };
    return {
      xMm: ((event.clientX - rect.left) / rect.width) * PAGE_WIDTH_MM,
      yMm: ((event.clientY - rect.top) / rect.height) * PAGE_HEIGHT_MM
    };
  }

  function startMove(event: ReactPointerEvent<SVGGElement>) {
    const point = pointerToPage(event as unknown as ReactPointerEvent<SVGSVGElement>);
    setDragState({ kind: "move", offsetX: point.xMm - placement.xMm, offsetY: point.yMm - placement.yMm });
  }

  function startResize(event: ReactPointerEvent<SVGCircleElement>) {
    event.stopPropagation();
    setDragState({ kind: "resize", startAspect: effectiveSourceAspect(sourceAspect, placement.rotationDeg) });
  }

  function handlePointerMove(event: ReactPointerEvent<SVGSVGElement>) {
    if (!dragState) return;
    const point = pointerToPage(event);
    if (dragState.kind === "move") {
      updatePlacement({ ...placement, xMm: point.xMm - dragState.offsetX, yMm: point.yMm - dragState.offsetY });
    } else {
      const widthMm = Math.max(10, point.xMm - placement.xMm);
      const heightMm = lockAspect ? widthMm / dragState.startAspect : Math.max(10, point.yMm - placement.yMm);
      updatePlacement({ ...placement, widthMm, heightMm });
    }
  }

  async function runMachineAction(action: string, distanceMm = 0) {
    setErrorMessage("");
    try {
      const response = await fetch("/api/machine/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          machine,
          action,
          distanceMm,
          pageWidthMm: PAGE_WIDTH_MM,
          pageHeightMm: PAGE_HEIGHT_MM
        })
      });
      const payload = (await response.json()) as {
        ok: boolean;
        completed?: boolean;
        label: string;
        message: string;
        state?: string | null;
        syncedZMm?: number;
        activeBridgeUrl?: string | null;
      };
      if (!response.ok || !payload.ok) throw new Error(payload.message || response.statusText);
      if (payload.activeBridgeUrl && payload.activeBridgeUrl !== machine.bridgeUrl) {
        setMachine((current) => ({ ...current, bridgeUrl: payload.activeBridgeUrl || current.bridgeUrl }));
      }
      setPenMessage(payload.message || `${payload.label} sent.`);
      setMachineState(normalizeMachineState(payload.state) ?? "Idle");
      return payload;
    } catch (error) {
      const rawMessage = error instanceof Error ? error.message : "Machine command failed.";
      const message = formatMachineError(rawMessage, machine.bridgeUrl);
      setMachineState(isBridgeConnectionError(rawMessage) ? "Disconnected" : "Alarm");
      setPenMessage(message);
      setErrorMessage(message);
      return null;
    }
  }

  async function homeAll() {
    if (drawJob?.status === "queued" || drawJob?.status === "running") {
      setPenMessage("Stop the active drawing before homing.");
      setErrorMessage("Home is locked out while a drawing job is active. Use Stop or E-STOP first.");
      return;
    }
    const result = await runMachineAction("home_all");
    if (result && result.completed !== false) {
      setMachineHomed(true);
      setTrackedPenZMm(result.syncedZMm ?? 3);
      setPenConfirmed(false);
      setPenMessage("Home all complete. Jog the pen down to contact, then save it.");
    }
  }

  async function homePen() {
    if (drawJob?.status === "queued" || drawJob?.status === "running") {
      setPenMessage("Stop the active drawing before homing the pen.");
      setErrorMessage("Home Pen is locked out while a drawing job is active. Use Stop or E-STOP first.");
      return;
    }
    if (!machineHomed) {
      setPenMessage("Home all first so X, Y, and Z are referenced.");
      return;
    }
    const result = await runMachineAction("home_pen");
    if (result && result.completed !== false) {
      setTrackedPenZMm(result.syncedZMm ?? 3);
      setPenConfirmed(false);
    }
  }

  async function jogPen(distanceMm: number) {
    if (!machineHomed) {
      setPenMessage("Home all first so the machine knows the drawing reference.");
      return;
    }
    if (await runMachineAction("jog_pen", distanceMm)) {
      setTrackedPenZMm((current) => (current === null ? distanceMm : current + distanceMm));
      setPenConfirmed(false);
    }
  }

  function savePenContactPoint() {
    if (!machineHomed || trackedPenZMm === null) {
      setPenMessage("Home all first, then jog the pen to paper contact.");
      setPenConfirmed(false);
      return;
    }
    const down = trackedPenZMm;
    const up = down - machine.penUpGapMm;
    setMachine((current) => ({ ...current, penDownPositionMm: down, penUpPositionMm: up }));
    setPenConfirmed(true);
    setPenMessage(`Saved pen-down at Z${down.toFixed(3)} and pen-up at Z${up.toFixed(3)}.`);
  }

  async function generatePreview() {
    if (!sourceFile) {
      setErrorMessage("Upload an image before generating G-code.");
      return;
    }
    if (!machineHomed) {
      setErrorMessage("Home all before generating G-code.");
      return;
    }
    if (!penConfirmed) {
      setErrorMessage("Save the pen contact point before generating G-code.");
      return;
    }
    setIsPlanning(true);
    setErrorMessage("");
    setGeneration(null);
    try {
      const formData = new FormData();
      formData.append("image", sourceFile);
      formData.append(
        "settings",
        JSON.stringify({
          ...settings,
          mode,
          pageWidthMm: PAGE_WIDTH_MM,
          pageHeightMm: PAGE_HEIGHT_MM,
          placement
        })
      );
      const response = await fetch("/api/generate", { method: "POST", body: formData });
      if (!response.ok) throw new Error(await response.text());
      setGeneration((await response.json()) as GenerationResult);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Could not generate G-code.");
    } finally {
      setIsPlanning(false);
    }
  }

  async function startDraw() {
    if (!generation) return;
    if (!machineHomed || !penConfirmed) {
      setErrorMessage("Home all and save pen contact before drawing.");
      return;
    }
    setErrorMessage("");
    try {
      const response = await fetch("/api/machine/draw", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gcode: generation.gcode, machine })
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = (await response.json()) as { job: DrawJob };
      setDrawJob(payload.job);
      setMachineState("Run");
    } catch (error) {
      setMachineState("Alarm");
      setErrorMessage(error instanceof Error ? error.message : "Could not start drawing.");
    }
  }

  async function stopDraw() {
    if (!drawJob) {
      setMachineState("Idle");
      return;
    }
    try {
      const response = await fetch(`/api/jobs/${drawJob.id}/cancel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(machine)
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = (await response.json()) as { job: DrawJob };
      setDrawJob(payload.job);
      setMachineState("Hold");
    } catch (error) {
      setMachineState("Alarm");
      setErrorMessage(error instanceof Error ? error.message : "Could not send feed hold.");
    }
  }

  async function resumeDraw() {
    if (!drawJob?.resume_available) return;
    setErrorMessage("");
    try {
      const response = await fetch(`/api/jobs/${drawJob.id}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ machine, rewindCommands: resumeRewind })
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = (await response.json()) as { job: DrawJob };
      setDrawJob(payload.job);
      setMachineState("Run");
    } catch (error) {
      setMachineState("Alarm");
      setErrorMessage(error instanceof Error ? error.message : "Could not resume drawing.");
    }
  }

  async function emergencyStop() {
    try {
      const response = await fetch("/api/machine/estop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(machine)
      });
      if (!response.ok) throw new Error(await response.text());
      setMachineState("Hold");
      setPenMessage("E-STOP sent. Feed-hold was sent to GRBL.");
    } catch (error) {
      setMachineState("Alarm");
      setErrorMessage(error instanceof Error ? error.message : "Could not send E-STOP.");
    }
  }

  async function refreshMachineLog() {
    setIsLogLoading(true);
    try {
      const response = await fetch("/api/machine/log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(machine)
      });
      const payload = (await response.json()) as MachineLogSnapshot;
      if (!response.ok || !payload.ok) throw new Error(payload.message || response.statusText);
      setMachineLog(payload.recentLog ?? []);
      setLogMessage(payload.state ? `Latest machine state: ${payload.state}` : "Machine log refreshed.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not refresh the machine log.";
      setLogMessage(message);
      setErrorMessage(message);
    } finally {
      setIsLogLoading(false);
    }
  }

  async function clearMachineLog() {
    setIsLogLoading(true);
    try {
      const response = await fetch("/api/machine/log/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(machine)
      });
      if (!response.ok) throw new Error(await response.text());
      setMachineLog([]);
      setLogMessage("Machine log cleared.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not clear the machine log.";
      setLogMessage(message);
      setErrorMessage(message);
    } finally {
      setIsLogLoading(false);
    }
  }

  function downloadGcode() {
    if (!generation) return;
    const blob = new Blob([generation.gcode], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = sourceName.replace(/\.[^.]+$/, "") + ".gcode";
    link.click();
    URL.revokeObjectURL(url);
  }

  const drawActive = drawJob?.status === "queued" || drawJob?.status === "running";
  const canGenerate = Boolean(sourceFile) && machineHomed && penConfirmed && !isPlanning;
  const canDraw = Boolean(generation) && machineHomed && penConfirmed && !drawActive;

  return (
    <main className="app-shell">
      <aside className="control-rail">
        <section className="brand-strip">
          <div className="brand-mark">
            <SquarePen size={22} />
          </div>
          <div>
            <h1>Photo to G-code</h1>
            <p>TypeScript plotter studio</p>
          </div>
        </section>

        <PanelTitle icon={<ImagePlus size={18} />} title="Source" />
        <label className="upload-drop">
          <Upload size={22} />
          <span>{sourceName}</span>
          <input type="file" accept="image/*" onChange={handleImageUpload} />
        </label>

        <PanelTitle icon={<Sparkles size={18} />} title="AI prep" />
        <textarea
          className="text-box"
          value={aiComments}
          onChange={(event) => setAiComments(event.target.value)}
          placeholder="Additional comments? e.g. simplify outlines"
        />
        <button className="primary-action" type="button" onClick={convertToAi} disabled={!sourceFile || isAiConverting}>
          <Wand2 size={18} />
          {isAiConverting ? "Converting" : "Convert to AI"}
        </button>
        <button className="secondary-action" type="button" onClick={saveAiImageToDesktop} disabled={!aiResult || isSavingAiImage}>
          <Download size={18} />
          {isSavingAiImage ? "Saving" : "Save AI image"}
        </button>
        {aiSaveMessage ? <p className="field-note">{aiSaveMessage}</p> : null}

        <PanelTitle icon={<Route size={18} />} title="Path mode" />
        <div className="segmented">
          {(Object.keys(modeMeta) as GenerationMode[]).map((option) => (
            <button
              key={option}
              type="button"
              className={mode === option ? "selected" : ""}
              onClick={() => {
                setMode(option);
                setGeneration(null);
              }}
            >
              {modeMeta[option].label}
            </button>
          ))}
        </div>
        <p className="field-note">{modeMeta[mode].note}</p>

        <PanelTitle icon={<SlidersHorizontal size={18} />} title="Processing" />
        <SliderField label="Threshold" value={settings.threshold} min={0} max={255} step={1} onChange={(value) => updateSetting("threshold", value)} />
        <SliderField label="Resolution" value={settings.maskResolutionPxMm} min={2} max={18} step={0.5} suffix="px/mm" onChange={(value) => updateSetting("maskResolutionPxMm", value)} />
        <SliderField label="Infill" value={settings.infillDensityPercent} min={0} max={100} step={5} suffix="%" onChange={(value) => updateSetting("infillDensityPercent", value)} />
      </aside>

      <section className="workbench">
        <header className="topbar">
          <div>
            <p className="eyebrow">Plotter control UI</p>
            <h2>Artwork to machine-ready paths</h2>
          </div>
          <div className="topbar-status">
            <span className="version-badge">{APP_VERSION}</span>
            <div className={`machine-pill ${machineState.toLowerCase()}`}>
              <Wifi size={17} />
              <span>{machine.bridgeUrl}</span>
              <strong>{machineState}</strong>
            </div>
          </div>
        </header>

        {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}

        <section className="studio-grid">
          <div className="preview-stage">
            <div className="stage-toolbar">
              <button type="button" className="selected">
                <Move size={17} />
                Place
              </button>
              <button type="button" className={showGrid ? "selected" : ""} onClick={() => setShowGrid((value) => !value)}>
                <Grid3X3 size={17} />
                Grid
              </button>
              <button type="button" onClick={() => updatePlacement(centerPlacement(placement))}>
                <RefreshCcw size={17} />
                Center
              </button>
              <button type="button" onClick={rotateArtworkClockwise} disabled={!sourceImage} title="Rotate artwork 90 degrees clockwise">
                <RotateCw size={17} />
                Rotate 90
              </button>
            </div>
            <div className="page-sheet">
              <svg
                ref={svgRef}
                className="cad-canvas"
                viewBox={`0 0 ${PAGE_WIDTH_MM} ${PAGE_HEIGHT_MM}`}
                onPointerMove={handlePointerMove}
                onPointerUp={() => setDragState(null)}
                onPointerLeave={() => setDragState(null)}
              >
                <rect width={PAGE_WIDTH_MM} height={PAGE_HEIGHT_MM} fill="#fbf7ec" />
                <GridOverlay visible={showGrid} />
                <RobotFrameMarkers />
                {sourceImage ? (
                  <g className="placed-art" onPointerDown={startMove}>
                    {(() => {
                      const artworkBox = rotatedArtworkBox(placement);
                      return (
                        <image
                          href={sourceImage}
                          x={artworkBox.xMm}
                          y={artworkBox.yMm}
                          width={artworkBox.widthMm}
                          height={artworkBox.heightMm}
                          preserveAspectRatio="none"
                          transform={`rotate(${placement.rotationDeg} ${artworkBox.centerX} ${artworkBox.centerY})`}
                        />
                      );
                    })()}
                    <rect
                      className="placement-border"
                      x={placement.xMm}
                      y={placement.yMm}
                      width={placement.widthMm}
                      height={placement.heightMm}
                    />
                    <DimensionLabels placement={placement} unit={dimensionUnit} />
                    <circle className="resize-handle" cx={placement.xMm} cy={placement.yMm} r={3.2} />
                    <circle className="resize-handle" cx={placement.xMm + placement.widthMm} cy={placement.yMm} r={3.2} />
                    <circle className="resize-handle" cx={placement.xMm} cy={placement.yMm + placement.heightMm} r={3.2} />
                    <circle
                      className="resize-handle"
                      cx={placement.xMm + placement.widthMm}
                      cy={placement.yMm + placement.heightMm}
                      r={3.2}
                      onPointerDown={startResize}
                    />
                  </g>
                ) : (
                  <g className="empty-state-svg">
                    <FileImage size={44} />
                    <text x={PAGE_WIDTH_MM / 2} y={PAGE_HEIGHT_MM / 2 + 10} textAnchor="middle">
                      Add an image to preview placement
                    </text>
                  </g>
                )}
              </svg>
              {isPlanning ? (
                <div className="processing-overlay">
                  <div className="plotter-loader">
                    <span />
                    <span />
                    <span />
                  </div>
                  <strong>Generating G-code</strong>
                </div>
              ) : null}
            </div>
          </div>

          <div className="right-stack">
            <PlacementPanel
              placement={placement}
              setPlacement={updatePlacement}
              unit={dimensionUnit}
              setUnit={setDimensionUnit}
              lockAspect={lockAspect}
              setLockAspect={setLockAspect}
              sourceAspect={sourceAspect}
              onRotate={rotateArtworkClockwise}
            />
            <CalibrationPanel
              machine={machine}
              updateMachine={updateMachine}
              machineHomed={machineHomed}
              penConfirmed={penConfirmed}
              penMessage={penMessage}
              trackedPenZMm={trackedPenZMm}
              drawActive={drawActive}
              onHomeAll={homeAll}
              onHomePen={homePen}
              onJogPen={jogPen}
              onSaveContact={savePenContactPoint}
            />
            <GeneratePanel canGenerate={canGenerate} isPlanning={isPlanning} generation={generation} onGenerate={generatePreview} onDownload={downloadGcode} />
            <StatusPanel stats={stats} generation={generation} drawJob={drawJob} />
            <EmergencyStopButton onEmergencyStop={emergencyStop} />
            <MachinePanel machine={machine} machineState={machineState} updateMachine={updateMachine} onDraw={startDraw} onStop={stopDraw} canDraw={canDraw} />
          </div>
        </section>

        <section className="workflow-panels">
          <TuningPanel settings={settings} updateSetting={updateSetting} />
          <CleanupPanel settings={settings} updateSetting={updateSetting} />
          <MachineLogPanel
            drawJob={drawJob}
            machineLog={machineLog}
            logMessage={logMessage}
            isLogLoading={isLogLoading}
            resumeRewind={resumeRewind}
            onResumeRewindChange={setResumeRewind}
            onResume={resumeDraw}
            onRefresh={refreshMachineLog}
            onClear={clearMachineLog}
          />
        </section>
      </section>
    </main>
  );
}

function GridOverlay({ visible }: { visible: boolean }) {
  if (!visible) return null;
  const verticals = [PAGE_WIDTH_MM / 4, PAGE_WIDTH_MM / 2, (PAGE_WIDTH_MM * 3) / 4];
  const horizontals = [1, 2, 3, 4, 5].map((index) => (PAGE_HEIGHT_MM * index) / 6);
  return (
    <g className="alignment-grid">
      {verticals.map((xPos) => (
        <line key={`v-${xPos}`} x1={xPos} y1={0} x2={xPos} y2={PAGE_HEIGHT_MM} />
      ))}
      {horizontals.map((yPos) => (
        <line key={`h-${yPos}`} x1={0} y1={yPos} x2={PAGE_WIDTH_MM} y2={yPos} />
      ))}
    </g>
  );
}

function RobotFrameMarkers() {
  return (
    <g className="robot-frame-markers">
      <path d={`M${PAGE_WIDTH_MM - 22} 8 H${PAGE_WIDTH_MM - 5} V25`} />
      <text x={PAGE_WIDTH_MM - 22} y={22}>HOME</text>
    </g>
  );
}

function DimensionLabels({ placement, unit }: { placement: Placement; unit: DimensionUnit }) {
  const width = unit === "in" ? placement.widthMm / MM_PER_IN : placement.widthMm / MM_PER_CM;
  const height = unit === "in" ? placement.heightMm / MM_PER_IN : placement.heightMm / MM_PER_CM;
  const suffix = unit;
  return (
    <g className="dimension-layer">
      <line x1={placement.xMm} y1={placement.yMm - 10} x2={placement.xMm + placement.widthMm} y2={placement.yMm - 10} />
      <text x={placement.xMm + placement.widthMm / 2} y={placement.yMm - 14} textAnchor="middle">
        {width.toFixed(2)} {suffix}
      </text>
      <line x1={placement.xMm - 10} y1={placement.yMm} x2={placement.xMm - 10} y2={placement.yMm + placement.heightMm} />
      <text x={placement.xMm - 13} y={placement.yMm + placement.heightMm / 2} textAnchor="middle" transform={`rotate(-90 ${placement.xMm - 13} ${placement.yMm + placement.heightMm / 2})`}>
        {height.toFixed(2)} {suffix}
      </text>
    </g>
  );
}

function PanelTitle({ icon, title }: { icon: JSX.Element; title: string }) {
  return (
    <div className="panel-title">
      {icon}
      <span>{title}</span>
    </div>
  );
}

function SliderField({ label, value, min, max, step, suffix, onChange }: { label: string; value: number; min: number; max: number; step: number; suffix?: string; onChange: (value: number) => void }) {
  return (
    <label className="slider-field">
      <span>
        {label}
        <strong>{formatNumber(value)}{suffix ? ` ${suffix}` : ""}</strong>
      </span>
      <input type="range" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function NumberField({ label, value, min, max, step, suffix, onChange }: { label: string; value: number; min: number; max: number; step: number; suffix?: string; onChange: (value: number) => void }) {
  return (
    <label className="number-field">
      <span>{label}</span>
      <div>
        <input type="number" value={Number.isFinite(value) ? value : 0} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} />
        {suffix ? <small>{suffix}</small> : null}
      </div>
    </label>
  );
}

function PlacementPanel({ placement, setPlacement, unit, setUnit, lockAspect, setLockAspect, sourceAspect, onRotate }: { placement: Placement; setPlacement: (placement: Placement) => void; unit: DimensionUnit; setUnit: (unit: DimensionUnit) => void; lockAspect: boolean; setLockAspect: (value: boolean) => void; sourceAspect: number; onRotate: () => void }) {
  const divisor = unit === "in" ? MM_PER_IN : MM_PER_CM;
  const suffix = unit;
  const placementAspect = effectiveSourceAspect(sourceAspect, placement.rotationDeg);
  function updateDimension(key: keyof Placement, displayValue: number) {
    const valueMm = displayValue * divisor;
    if (key === "widthMm" && lockAspect) {
      setPlacement({ ...placement, widthMm: valueMm, heightMm: valueMm / placementAspect });
      return;
    }
    if (key === "heightMm" && lockAspect) {
      setPlacement({ ...placement, heightMm: valueMm, widthMm: valueMm * placementAspect });
      return;
    }
    setPlacement({ ...placement, [key]: valueMm });
  }
  return (
    <section className="dimension-panel">
      <div className="panel-heading">
        <Ruler size={18} />
        <h3>Placement</h3>
      </div>
      <div className="unit-row">
        <button type="button" className={unit === "in" ? "selected" : ""} onClick={() => setUnit("in")}>inches</button>
        <button type="button" className={unit === "cm" ? "selected" : ""} onClick={() => setUnit("cm")}>cm</button>
        <button type="button" className={lockAspect ? "selected icon-only" : "icon-only"} onClick={() => setLockAspect(!lockAspect)} title="Lock aspect ratio">
          {lockAspect ? <Lock size={17} /> : <Unlock size={17} />}
        </button>
      </div>
      <div className="field-grid">
        <NumberField label="Width" value={placement.widthMm / divisor} min={0.1} max={20} step={0.01} suffix={suffix} onChange={(value) => updateDimension("widthMm", value)} />
        <NumberField label="Height" value={placement.heightMm / divisor} min={0.1} max={20} step={0.01} suffix={suffix} onChange={(value) => updateDimension("heightMm", value)} />
        <NumberField label="X" value={placement.xMm / divisor} min={0} max={20} step={0.01} suffix={suffix} onChange={(value) => updateDimension("xMm", value)} />
        <NumberField label="Y" value={placement.yMm / divisor} min={0} max={20} step={0.01} suffix={suffix} onChange={(value) => updateDimension("yMm", value)} />
      </div>
      <div className="placement-actions">
        <button type="button" onClick={() => setPlacement(centerPlacement(placement))}>
          <RefreshCcw size={17} />
          Center
        </button>
        <button type="button" onClick={onRotate}>
          <RotateCw size={17} />
          Rotate 90
        </button>
      </div>
      <p className="field-note">Rotation: {placement.rotationDeg} deg</p>
    </section>
  );
}

function CalibrationPanel({ machine, updateMachine, machineHomed, penConfirmed, penMessage, trackedPenZMm, drawActive, onHomeAll, onHomePen, onJogPen, onSaveContact }: { machine: MachineSettings; updateMachine: <Key extends keyof MachineSettings>(key: Key, value: MachineSettings[Key]) => void; machineHomed: boolean; penConfirmed: boolean; penMessage: string; trackedPenZMm: number | null; drawActive: boolean; onHomeAll: () => void; onHomePen: () => void; onJogPen: (distance: number) => void; onSaveContact: () => void }) {
  return (
    <section className={`calibration-panel ${penConfirmed ? "confirmed" : ""}`}>
      <div className="panel-heading">
        <PenLine size={18} />
        <h3>Calibration</h3>
      </div>
      <button className="primary-action home-all-button" type="button" onClick={onHomeAll} disabled={drawActive} title={drawActive ? "Stop the active drawing before homing" : undefined}>
        <Home size={17} />
        Home All
      </button>
      <div className="calibration-status">
        <span>{machineHomed ? "Machine homed" : "Machine not homed"}</span>
        <strong>{penConfirmed ? "Pen saved" : "Pen unsaved"}</strong>
      </div>
      <div className="calibration-status">
        <span>Pen position</span>
        <strong>{trackedPenZMm === null ? "-" : `Z${trackedPenZMm.toFixed(3)}`}</strong>
      </div>
      <button type="button" onClick={onHomePen} disabled={drawActive} title={drawActive ? "Stop the active drawing before homing" : undefined}>
        <Home size={17} />
        Home pen
      </button>
      <div className="jog-grid">
        <button type="button" onClick={() => onJogPen(10)}>Down 10</button>
        <button type="button" onClick={() => onJogPen(5)}>Down 5</button>
        <button type="button" onClick={() => onJogPen(1)}>Down 1</button>
        <button type="button" onClick={() => onJogPen(-1)}>Up 1</button>
      </div>
      <div className="field-grid compact">
        <NumberField label="Up gap" value={machine.penUpGapMm} min={0.5} max={12} step={0.1} suffix="mm" onChange={(value) => updateMachine("penUpGapMm", value)} />
        <NumberField label="Feed" value={machine.penFeedRateMmMin} min={100} max={12000} step={100} suffix="mm/min" onChange={(value) => updateMachine("penFeedRateMmMin", value)} />
      </div>
      <button className="secondary-action" type="button" onClick={onSaveContact}>
        <CheckCircle2 size={17} />
        Save pen contact
      </button>
      <p className="field-note">{penMessage}</p>
    </section>
  );
}

function GeneratePanel({ canGenerate, isPlanning, generation, onGenerate, onDownload }: { canGenerate: boolean; isPlanning: boolean; generation: GenerationResult | null; onGenerate: () => void; onDownload: () => void }) {
  return (
    <section className="generate-panel">
      <div className="panel-heading">
        <Cpu size={18} />
        <h3>Generate G-code</h3>
      </div>
      <button className="primary-action" type="button" onClick={onGenerate} disabled={!canGenerate}>
        <Cpu size={17} />
        {isPlanning ? "Generating" : "Generate G-code"}
      </button>
      <button className="secondary-action" type="button" disabled={!generation} onClick={onDownload}>
        <Download size={17} />
        Export G-code
      </button>
    </section>
  );
}

function StatusPanel({ stats, generation, drawJob }: { stats: { paths: number; commands: number; drawSeconds: number; commandsPerSecond: number; page: string }; generation: GenerationResult | null; drawJob: DrawJob | null }) {
  const activeProgress = drawJob ? Math.round(drawJob.progress * 100) : 0;
  const lossCount = drawJob?.connection_loss_count ?? 0;
  const recoveryCount = drawJob?.connection_recovery_count ?? 0;
  const autoResumeAttempts = drawJob?.auto_resume_attempts ?? 0;
  const qualityScore = drawJob?.connection_quality_score ?? 0;
  const qualityLabel = drawJob?.connection_quality_label ?? "unknown";
  const latency = drawJob?.connection_latency_ms;
  const rebootCount = drawJob?.bridge_reboot_count ?? 0;
  return (
    <section className="summary-panel">
      <div className="panel-heading">
        <Activity size={18} />
        <h3>Path summary</h3>
      </div>
      <div className="metric-grid">
        <Metric label="Paths" value={generation ? stats.paths.toLocaleString() : "-"} />
        <Metric label="G-code lines" value={generation ? stats.commands.toLocaleString() : "-"} />
        <Metric label="Estimate" value={generation ? formatDuration(stats.drawSeconds) : "-"} />
        <Metric label="Page" value={stats.page} />
      </div>
      {drawJob ? (
        <>
          <div className="progress-track">
            <div style={{ width: `${activeProgress}%` }} />
          </div>
          <p className="field-note">{drawJob.message}</p>
          <div className={`quality-meter ${qualityClass(qualityLabel, qualityScore)}`}>
            <div className="quality-row">
              <span>Bridge quality</span>
              <strong>{qualityLabel} · {qualityScore}/100</strong>
            </div>
            <div className="quality-track">
              <div style={{ width: `${Math.max(0, Math.min(100, qualityScore))}%` }} />
            </div>
            <p className="field-note">{drawJob.connection_health_message ?? "Bridge health has not been checked yet."}</p>
          </div>
          <div className="metric-grid recovery-metrics">
            <Metric label="Lost" value={lossCount.toLocaleString()} />
            <Metric label="Recovered" value={recoveryCount.toLocaleString()} />
            <Metric label="Auto resumes" value={autoResumeAttempts.toLocaleString()} />
            <Metric label="Latency" value={typeof latency === "number" ? `${Math.round(latency)} ms` : "-"} />
            <Metric label="ESP reboots" value={rebootCount.toLocaleString()} />
          </div>
        </>
      ) : null}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TuningPanel({ settings, updateSetting }: { settings: WorkflowSettings; updateSetting: <Key extends keyof WorkflowSettings>(key: Key, value: WorkflowSettings[Key]) => void }) {
  return (
    <section className="tool-panel">
      <div className="panel-heading">
        <Gauge size={18} />
        <h3>Stroke + speed</h3>
      </div>
      <div className="field-grid">
        <NumberField label="Line width" value={settings.lineWidthMm} min={0.02} max={0.5} step={0.001} suffix="mm" onChange={(value) => updateSetting("lineWidthMm", value)} />
        <NumberField label="Wall lines" value={settings.wallLines} min={0} max={6} step={1} onChange={(value) => updateSetting("wallLines", value)} />
        <NumberField label="Draw speed" value={settings.drawSpeedMmSec} min={5} max={180} step={5} suffix="mm/s" onChange={(value) => updateSetting("drawSpeedMmSec", value)} />
        <NumberField label="Travel speed" value={settings.travelSpeedMmSec} min={5} max={180} step={5} suffix="mm/s" onChange={(value) => updateSetting("travelSpeedMmSec", value)} />
      </div>
    </section>
  );
}

function CleanupPanel({ settings, updateSetting }: { settings: WorkflowSettings; updateSetting: <Key extends keyof WorkflowSettings>(key: Key, value: WorkflowSettings[Key]) => void }) {
  return (
    <section className="tool-panel">
      <div className="panel-heading">
        <Layers3 size={18} />
        <h3>Path cleanup</h3>
      </div>
      <div className="mini-toggle">
        <button type="button" className={settings.fillStrategy === "continuous_zigzag" ? "selected" : ""} onClick={() => updateSetting("fillStrategy", "continuous_zigzag")}>Continuous fill</button>
        <button type="button" className={settings.fillStrategy === "separate_paths" ? "selected" : ""} onClick={() => updateSetting("fillStrategy", "separate_paths")}>Separate paths</button>
      </div>
      <div className="field-grid">
        <NumberField label="Fill chunk size" value={settings.continuousFillChunkSegments} min={0} max={250} step={1} suffix="segments" onChange={(value) => updateSetting("continuousFillChunkSegments", value)} />
        <NumberField label="Path simplification" value={settings.pathSimplifyToleranceMm} min={0} max={5} step={0.01} suffix="mm" onChange={(value) => updateSetting("pathSimplifyToleranceMm", value)} />
        <NumberField label="Min segment length" value={settings.minSegmentLengthMm} min={0} max={5} step={0.01} suffix="mm" onChange={(value) => updateSetting("minSegmentLengthMm", value)} />
        <NumberField label="Coordinate decimals" value={settings.coordinateDecimals} min={1} max={4} step={1} onChange={(value) => updateSetting("coordinateDecimals", value)} />
      </div>
    </section>
  );
}

function MachinePanel({ machine, machineState, updateMachine, onDraw, onStop, canDraw }: { machine: MachineSettings; machineState: MachineState; updateMachine: <Key extends keyof MachineSettings>(key: Key, value: MachineSettings[Key]) => void; onDraw: () => void; onStop: () => void; canDraw: boolean }) {
  return (
    <section className="machine-panel">
      <div className="panel-heading">
        <Cable size={18} />
        <h3>Machine control</h3>
      </div>
      <label className="bridge-input">
        <span>ESP32 bridge</span>
        <input value={machine.bridgeUrl} onChange={(event) => updateMachine("bridgeUrl", event.target.value)} />
      </label>
      <div className="field-grid compact">
        <NumberField label="Timeout" value={machine.timeoutSeconds} min={0.5} max={20} step={0.5} suffix="s" onChange={(value) => updateMachine("timeoutSeconds", value)} />
        <NumberField label="Queue window" value={machine.queueWindowSize} min={1} max={128} step={1} onChange={(value) => updateMachine("queueWindowSize", value)} />
        <NumberField label="Batch ack" value={machine.batchAckTimeoutSeconds} min={2} max={300} step={1} suffix="s" onChange={(value) => updateMachine("batchAckTimeoutSeconds", value)} />
        <NumberField label="In flight" value={machine.maxInFlight} min={1} max={16} step={1} onChange={(value) => updateMachine("maxInFlight", value)} />
        <NumberField label="Send spacing" value={machine.sendSpacingMs} min={0} max={500} step={1} suffix="ms" onChange={(value) => updateMachine("sendSpacingMs", value)} />
        <NumberField label="Recovery" value={machine.recoveryTimeoutSeconds} min={0} max={600} step={5} suffix="s" onChange={(value) => updateMachine("recoveryTimeoutSeconds", value)} />
        <NumberField label="Cooldown" value={machine.bridgeRecoveryCooldownSeconds} min={0} max={60} step={1} suffix="s" onChange={(value) => updateMachine("bridgeRecoveryCooldownSeconds", value)} />
        <NumberField label="Healthy under" value={machine.bridgeHealthMaxLatencyMs} min={100} max={10000} step={100} suffix="ms" onChange={(value) => updateMachine("bridgeHealthMaxLatencyMs", value)} />
        <NumberField label="Reboot wait" value={machine.bridgeRestartWaitSeconds} min={1} max={60} step={1} suffix="s" onChange={(value) => updateMachine("bridgeRestartWaitSeconds", value)} />
        <label className="checkbox-field">
          <input type="checkbox" checked={machine.autoResumeEnabled} onChange={(event) => updateMachine("autoResumeEnabled", event.target.checked)} />
          <span>Auto recovery</span>
        </label>
        <label className="checkbox-field">
          <input type="checkbox" checked={machine.bridgeRestartEnabled} onChange={(event) => updateMachine("bridgeRestartEnabled", event.target.checked)} />
          <span>ESP restart</span>
        </label>
        <NumberField label="Overlap" value={machine.autoResumeRewindCommands} min={0} max={250} step={1} suffix="commands" onChange={(value) => updateMachine("autoResumeRewindCommands", value)} />
        <NumberField label="Retries" value={machine.autoResumeMaxAttempts} min={0} max={100} step={1} onChange={(value) => updateMachine("autoResumeMaxAttempts", value)} />
      </div>
      <div className="draw-row">
        <button className="draw-button" type="button" onClick={onDraw} disabled={!canDraw || machineState === "Run"}>
          <Play size={18} />
          Draw
        </button>
        <button type="button" onClick={onStop}>
          <CheckCircle2 size={18} />
          Feed hold
        </button>
      </div>
    </section>
  );
}

function EmergencyStopButton({ onEmergencyStop }: { onEmergencyStop: () => void }) {
  return (
    <button className="estop-button" type="button" onClick={onEmergencyStop}>
      <OctagonAlert size={24} />
      E-STOP
    </button>
  );
}

function MachineLogPanel({
  drawJob,
  machineLog,
  logMessage,
  isLogLoading,
  resumeRewind,
  onResumeRewindChange,
  onResume,
  onRefresh,
  onClear
}: {
  drawJob: DrawJob | null;
  machineLog: string[];
  logMessage: string;
  isLogLoading: boolean;
  resumeRewind: number;
  onResumeRewindChange: (value: number) => void;
  onResume: () => void;
  onRefresh: () => void;
  onClear: () => void;
}) {
  const failureContext = drawJob?.command_context ?? [];
  const canResume = drawJob?.status === "error" && Boolean(drawJob.resume_available);
  const resumeLine = drawJob?.resume_index == null ? null : drawJob.resume_index + 1;
  return (
    <section className="tool-panel machine-log-panel">
      <div className="panel-heading">
        <TerminalSquare size={18} />
        <h3>Machine log</h3>
      </div>
      <div className="log-actions">
        <button type="button" onClick={onRefresh} disabled={isLogLoading}><RefreshCcw size={17} /> Refresh</button>
        <button type="button" onClick={onClear} disabled={isLogLoading}><Trash2 size={17} /> Clear</button>
      </div>
      <p className="field-note">{logMessage}</p>
      {drawJob?.status === "error" && failureContext.length ? (
        <div className="failure-card">
          <strong>Failure context</strong>
          {resumeLine ? <span>Resume line {resumeLine}</span> : null}
          <code>{failureContext.slice(Math.max(0, (drawJob.failed_command_index ?? 0) - 4), (drawJob.failed_command_index ?? 0) + 6).join("\n")}</code>
          {canResume ? (
            <div className="resume-controls">
              <NumberField label="Overlap" value={resumeRewind} min={0} max={250} step={1} suffix="commands" onChange={onResumeRewindChange} />
              <button type="button" onClick={onResume}>
                <RefreshCcw size={17} />
                Resume
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
      <textarea className="log-textarea" readOnly value={machineLog.join("\n")} placeholder="Machine log will appear here." />
    </section>
  );
}

function readImageAspect(url: string): Promise<number> {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve(image.naturalWidth / Math.max(image.naturalHeight, 1));
    image.onerror = () => resolve(initialPlacement.widthMm / initialPlacement.heightMm);
    image.src = url;
  });
}

function centerPlacement(placement: Placement): Placement {
  return clampPlacement({
    ...placement,
    xMm: (PAGE_WIDTH_MM - placement.widthMm) / 2,
    yMm: (PAGE_HEIGHT_MM - placement.heightMm) / 2
  });
}

function clampPlacement(placement: Placement): Placement {
  const widthMm = Math.max(4, Math.min(placement.widthMm, PAGE_WIDTH_MM));
  const heightMm = Math.max(4, Math.min(placement.heightMm, PAGE_HEIGHT_MM));
  return {
    ...placement,
    widthMm,
    heightMm,
    rotationDeg: normalizeRightAngleRotation(placement.rotationDeg),
    xMm: Math.min(Math.max(placement.xMm, 0), PAGE_WIDTH_MM - widthMm),
    yMm: Math.min(Math.max(placement.yMm, 0), PAGE_HEIGHT_MM - heightMm)
  };
}

function rotatePlacementClockwise(placement: Placement): Placement {
  const centerX = placement.xMm + placement.widthMm / 2;
  const centerY = placement.yMm + placement.heightMm / 2;
  const widthMm = placement.heightMm;
  const heightMm = placement.widthMm;
  return clampPlacement({
    ...placement,
    widthMm,
    heightMm,
    rotationDeg: normalizeRightAngleRotation(placement.rotationDeg + 90),
    xMm: centerX - widthMm / 2,
    yMm: centerY - heightMm / 2
  });
}

function rotatedArtworkBox(placement: Placement) {
  const centerX = placement.xMm + placement.widthMm / 2;
  const centerY = placement.yMm + placement.heightMm / 2;
  const quarterTurns = Math.abs(normalizeRightAngleRotation(placement.rotationDeg) / 90) % 2;
  const widthMm = quarterTurns === 1 ? placement.heightMm : placement.widthMm;
  const heightMm = quarterTurns === 1 ? placement.widthMm : placement.heightMm;
  return {
    centerX,
    centerY,
    xMm: centerX - widthMm / 2,
    yMm: centerY - heightMm / 2,
    widthMm,
    heightMm
  };
}

function effectiveSourceAspect(sourceAspect: number, rotationDeg: number) {
  const safeAspect = Math.max(sourceAspect, 0.01);
  const quarterTurns = Math.abs(normalizeRightAngleRotation(rotationDeg) / 90) % 2;
  return quarterTurns === 1 ? 1 / safeAspect : safeAspect;
}

function normalizeRightAngleRotation(rotationDeg: number) {
  const snapped = Math.round(rotationDeg / 90) * 90;
  return ((snapped % 360) + 360) % 360;
}

function formatDuration(seconds: number) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  return `${Math.round(seconds / 60)} min`;
}

function formatNumber(value: number) {
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
}

function normalizeMachineState(state: string | null | undefined): MachineState | null {
  const normalized = (state ?? "").trim().toLowerCase();
  if (normalized === "idle") return "Idle";
  if (normalized === "run") return "Run";
  if (normalized === "hold") return "Hold";
  if (normalized === "alarm") return "Alarm";
  if (normalized === "disconnected") return "Disconnected";
  return null;
}

function qualityClass(label: string, score: number) {
  const normalized = label.trim().toLowerCase();
  if (normalized === "healthy" || score >= 80) return "healthy";
  if (normalized === "checking" || normalized === "cooldown" || normalized === "rebooting") return "working";
  if (normalized === "slow" || normalized === "degraded" || score >= 40) return "degraded";
  return "offline";
}

function isBridgeConnectionError(message: string) {
  const normalized = message.toLowerCase();
  return (
    normalized.includes("connecttimeout") ||
    normalized.includes("connection timed out") ||
    normalized.includes("failed to connect") ||
    normalized.includes("max retries exceeded") ||
    normalized.includes("connection refused") ||
    normalized.includes("timed out")
  );
}

function formatMachineError(message: string, bridgeUrl: string) {
  if (!isBridgeConnectionError(message)) return message;
  return `Could not reach the ESP32 bridge at ${bridgeUrl}. Check that the ESP32 is powered, connected to Wi-Fi, and still using this IP address.`;
}
