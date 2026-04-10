// Package api defines the HTTP request/response models for the Go engine.
package api

// ── Reset ────────────────────────────────────────────────────────────────────

// ResetRequest is the body for POST /reset.
// Supports both internal (episode_id + task_config) and OpenEnv (task) modes.
type ResetRequest struct {
	// OpenEnv-style fields
	Task string `json:"task,omitempty"` // "iot_8bit", "rv32im", "mseries_superscalar"
	Seed *int64 `json:"seed,omitempty"`

	// Internal fields (Python → Go communication)
	EpisodeID  string     `json:"episode_id,omitempty"`
	TaskID     int        `json:"task_id,omitempty"`
	TaskConfig TaskConfig `json:"task_config,omitempty"`
}

// TaskConfig holds task-specific constraints.
type TaskConfig struct {
	Name        string             `json:"name"`
	MaxNodes    int                `json:"max_nodes"`
	MaxSteps    int                `json:"max_steps"`
	Constraints map[string]float64 `json:"constraints"`
}

// ResetResponse returns the initial state of the graph.
type ResetResponse struct {
	Status           string            `json:"status"`
	Error            string            `json:"error,omitempty"`
	GraphID          string            `json:"graph_id"`
	InitialMetrics   MetricsPayload    `json:"initial_metrics"`
	Components       []string          `json:"components"`
	Connections      [][2]string       `json:"connections"`
	AvailableActions []string          `json:"available_actions"`
	Observation      *ObservationModel `json:"observation,omitempty"` // OpenEnv-style
}

// ── Step ─────────────────────────────────────────────────────────────────────

// StepRequest is sent for each agent action.
type StepRequest struct {
	EpisodeID string `json:"episode_id,omitempty"`
	Action    Action `json:"action"`
	// OpenEnv direct action fields (flattened)
	ActionType      string  `json:"action_type,omitempty"`
	TargetComponent string  `json:"target_component,omitempty"`
	ParameterName   string  `json:"parameter_name,omitempty"`
	ParameterValue  float64 `json:"parameter_value,omitempty"`
	SourceNode      string  `json:"source_node,omitempty"`
	TargetNode      string  `json:"target_node,omitempty"`
	Reasoning       string  `json:"reasoning,omitempty"`
}

// Action represents a single architectural decision.
type Action struct {
	Type       string  `json:"type"`
	Component  string  `json:"component"`
	ParamName  string  `json:"param_name"`
	Value      float64 `json:"value"`
	SourceNode string  `json:"source_node,omitempty"`
	TargetNode string  `json:"target_node,omitempty"`
}

// StepResponse returns the result of applying an action.
type StepResponse struct {
	Status           string            `json:"status"`
	Valid            bool              `json:"valid"`
	Error            string            `json:"error,omitempty"`
	Metrics          MetricsPayload    `json:"metrics"`
	Components       []string          `json:"components"`
	Connections      [][2]string       `json:"connections"`
	ValidationErrors []string          `json:"validation_errors"`
	EngineeringNotes string            `json:"engineering_notes"`
	Observation      *ObservationModel `json:"observation,omitempty"` // OpenEnv-style
	Reward           float64           `json:"reward"`
	Done             bool              `json:"done"`
}

// ── OpenEnv Observation ──────────────────────────────────────────────────────

// ObservationModel is the full OpenEnv observation returned to agents.
type ObservationModel struct {
	CurrentEstimatedIPC float64 `json:"current_estimated_ipc"`
	ThroughputGIPS      float64 `json:"throughput_gips"`
	EffectiveClockGHz   float64 `json:"effective_clock_ghz"`
	TotalPowerMW        float64 `json:"total_power_mw"`
	TotalAreaMM2        float64 `json:"total_area_mm2"`
	MaxPowerDensity     float64 `json:"max_power_density"`
	ThermalCelsius      float64 `json:"thermal_celsius"`
	HotspotCount        int     `json:"hotspot_count"`
	ThrottledFactor     float64 `json:"throttled_factor"`
	PerfPerWatt         float64 `json:"perf_per_watt"`
	ActiveComponents    string  `json:"active_components"`
	ComponentCount      int     `json:"component_count"`
	ConnectionCount     int     `json:"connection_count"`
	TaskID              int     `json:"task_id"`
	TaskName            string  `json:"task_name"`
	TaskConstraints     string  `json:"task_constraints"`
	StepNumber          int     `json:"step_number"`
	MaxSteps            int     `json:"max_steps"`
	CumulativeReward    float64 `json:"cumulative_reward"`
	FeedbackString      string  `json:"feedback_string"`
	FinalScore          float64 `json:"final_score"`
	Done                bool    `json:"done"`
	Reward              float64 `json:"reward"`
}

// ── Metrics ──────────────────────────────────────────────────────────────────

// MetricsPayload is the PPA metrics JSON payload.
type MetricsPayload struct {
	IPC               float64            `json:"ipc"`
	ThroughputGIPS    float64            `json:"throughput_gips"`
	EffectiveClockGHz float64            `json:"effective_clock_ghz"`
	TotalPowerMW      float64            `json:"total_power_mw"`
	TotalAreaMM2      float64            `json:"total_area_mm2"`
	MaxPowerDensity   float64            `json:"max_power_density"`
	ThermalCelsius    float64            `json:"thermal_celsius"`
	HotspotCount      int                `json:"hotspot_count"`
	ThrottledFactor   float64            `json:"throttled_factor"`
	PerfPerWatt       float64            `json:"perf_per_watt"`
	ComponentPower    map[string]float64 `json:"component_power,omitempty"`
	ComponentArea     map[string]float64 `json:"component_area,omitempty"`
}

// ── Health ────────────────────────────────────────────────────────────────────

// HealthResponse for the /health endpoint.
type HealthResponse struct {
	Status         string `json:"status"`
	Version        string `json:"version,omitempty"`
	ActiveEpisodes int    `json:"active_episodes"`
}

// ── Grade ────────────────────────────────────────────────────────────────────

// EpisodeGrade is returned from GET /grade.
type EpisodeGrade struct {
	TaskID          int                    `json:"task_id"`
	Score           float64                `json:"score"`
	SubScores       map[string]float64     `json:"sub_scores"`
	ExploitDetected bool                   `json:"exploit_detected"`
	PenaltyApplied  float64                `json:"penalty_applied"`
	Details         map[string]interface{} `json:"details"`
}

// ── Tasks ────────────────────────────────────────────────────────────────────

// TaskInfo describes a task for GET /tasks.
type TaskInfo struct {
	ID          string             `json:"id"`
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Difficulty  string             `json:"difficulty"`
	MaxSteps    int                `json:"max_steps"`
	Weights     map[string]float64 `json:"weights"`
}

// ── Replay ───────────────────────────────────────────────────────────────────

// ReplayEntry records a single timestep for episode replay.
type ReplayEntry struct {
	Step       int            `json:"step"`
	Action     Action         `json:"action"`
	Metrics    MetricsPayload `json:"metrics"`
	Reward     float64        `json:"reward"`
	Valid      bool           `json:"valid"`
	Done       bool           `json:"done"`
}

// ReplayResponse is returned from GET /replay.
type ReplayResponse struct {
	Replay []ReplayEntry `json:"replay"`
	Steps  int           `json:"steps"`
}

// ── State ────────────────────────────────────────────────────────────────────

// StateResponse is returned from GET /state.
type StateResponse struct {
	Episode    int            `json:"episode"`
	Step       int            `json:"step"`
	TaskID     int            `json:"task_id"`
	TaskName   string         `json:"task_name"`
	Done       bool           `json:"done"`
	Metrics    MetricsPayload `json:"metrics"`
	Components []string       `json:"components"`
	Connections [][2]string   `json:"connections"`
}
