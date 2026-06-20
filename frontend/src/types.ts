export type GenerationMode = "vector_trace" | "cura_slice" | "triangle_mesh";
export type FillStrategy = "continuous_zigzag" | "separate_paths";
export type MachineState = "Idle" | "Run" | "Hold" | "Alarm" | "Disconnected";
export type DimensionUnit = "in" | "cm";

export interface Placement {
  xMm: number;
  yMm: number;
  widthMm: number;
  heightMm: number;
  rotationDeg: number;
}

export interface WorkflowSettings {
  threshold: number;
  invertInput: boolean;
  marginMm: number;
  maskResolutionPxMm: number;
  lineWidthMm: number;
  wallLines: number;
  infillDensityPercent: number;
  drawSpeedMmSec: number;
  travelSpeedMmSec: number;
  fillStrategy: FillStrategy;
  fillTurnSplitAngleDeg: number;
  continuousFillChunkSegments: number;
  pathSimplifyToleranceMm: number;
  minSegmentLengthMm: number;
  minToolpathLengthMm: number;
  coordinateDecimals: number;
}

export interface MachineSettings {
  bridgeUrl: string;
  timeoutSeconds: number;
  penUpGapMm: number;
  queueWindowSize: number;
  batchAckTimeoutSeconds: number;
  maxInFlight: number;
  sendSpacingMs: number;
  recoveryTimeoutSeconds: number;
  bridgeRecoveryCooldownSeconds: number;
  bridgeHealthMaxLatencyMs: number;
  bridgeRestartEnabled: boolean;
  bridgeRestartWaitSeconds: number;
  autoResumeEnabled: boolean;
  autoResumeRewindCommands: number;
  autoResumeMaxAttempts: number;
  autoResumeRetryDelaySeconds: number;
  usePenAxis: boolean;
  penAxis: "X" | "Y" | "Z";
  penUpPositionMm: number;
  penDownPositionMm: number;
  penFeedRateMmMin: number;
  penUpDwellSeconds: number;
  penDownDwellSeconds: number;
}

export interface GenerationResult {
  mode: GenerationMode;
  gcode: string;
  previewImage: string;
  maskImage: string;
  metrics: {
    path_count: number;
    perimeter_paths: number;
    fill_paths: number;
    centerline_paths: number;
    draw_distance_mm: number;
    travel_distance_mm: number;
  };
  diagnostics?: Record<string, number | string>;
  totalCommands: number;
  estimatedSeconds: number;
  secondsPerCommand: number;
  commandsPerSecond: number;
  planningSeconds: number;
  engine: string;
}

export interface DrawJob {
  id: string;
  status: "queued" | "running" | "complete" | "canceled" | "error";
  message: string;
  total_commands: number;
  completed_commands: number;
  sent_commands: number;
  progress: number;
  estimated_seconds: number;
  remaining_seconds: number;
  seconds_per_command: number;
  started_at: number;
  updated_at: number;
  finished_at?: number | null;
  error?: string | null;
  failed_command?: string | null;
  failed_command_index?: number | null;
  command_context?: string[] | null;
  recent_log?: string[] | null;
  resume_available?: boolean;
  resume_index?: number | null;
  resume_command_hash?: string;
  active_bridge_url?: string;
  parent_job_id?: string | null;
  auto_resume_attempts?: number;
  connection_loss_count?: number;
  connection_recovery_count?: number;
  connection_quality_score?: number;
  connection_quality_label?: string;
  connection_latency_ms?: number | null;
  connection_health_message?: string;
  bridge_reboot_count?: number;
}

export interface MachineLogSnapshot {
  ok: boolean;
  message: string;
  state?: string | null;
  lastCommand?: string;
  recentLog: string[];
  raw?: string | null;
}
