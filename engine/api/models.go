// Package api defines the HTTP request/response models for the Go engine.
package api

// ResetRequest is sent by Python to initialize a new episode.
type ResetRequest struct {
	EpisodeID string     `json:"episode_id"`
	TaskID    int        `json:"task_id"`
	TaskConfig TaskConfig `json:"task_config"`
}

// TaskConfig holds task-specific constraints.
type TaskConfig struct {
	Name        string            `json:"name"`
	MaxNodes    int               `json:"max_nodes"`
	MaxSteps    int               `json:"max_steps"`
	Constraints map[string]float64 `json:"constraints"` // e.g. "max_power_mw": 50
}

// ResetResponse returns the initial state of the graph.
type ResetResponse struct {
	Status         string         `json:"status"` // "ok" or "error"
	Error          string         `json:"error,omitempty"`
	GraphID        string         `json:"graph_id"`
	InitialMetrics MetricsPayload `json:"initial_metrics"`
	Components     []string       `json:"components"`
	Connections    [][2]string    `json:"connections"`
	AvailableActions []string     `json:"available_actions"` // What the agent can do
}

// StepRequest is sent by Python for each agent action.
type StepRequest struct {
	EpisodeID string `json:"episode_id"`
	Action    Action `json:"action"`
}

// Action represents a single architectural decision.
type Action struct {
	Type      string  `json:"type"`       // "add_component", "remove_component", "resize", "connect"
	Component string  `json:"component"`  // Target component ID
	ParamName string  `json:"param_name"` // Parameter to modify
	Value     float64 `json:"value"`      // New value
	SourceNode string `json:"source_node,omitempty"` // For "connect" actions
	TargetNode string `json:"target_node,omitempty"` // For "connect" actions
}

// StepResponse returns the result of applying an action.
type StepResponse struct {
	Status           string         `json:"status"` // "ok" or "error"
	Valid            bool           `json:"valid"`   // Was the action valid?
	Error            string         `json:"error,omitempty"`
	Metrics          MetricsPayload `json:"metrics"`
	Components       []string       `json:"components"`
	Connections      [][2]string    `json:"connections"`
	ValidationErrors []string       `json:"validation_errors"`
	EngineeringNotes string         `json:"engineering_notes"` // LLM-readable feedback
}

// MetricsPayload is the PPA metrics JSON payload.
type MetricsPayload struct {
	IPC              float64            `json:"ipc"`
	ThroughputGIPS   float64            `json:"throughput_gips"`
	EffectiveClockGHz float64           `json:"effective_clock_ghz"`
	TotalPowerMW     float64            `json:"total_power_mw"`
	TotalAreaMM2     float64            `json:"total_area_mm2"`
	MaxPowerDensity  float64            `json:"max_power_density"`
	ThermalCelsius   float64            `json:"thermal_celsius"`
	HotspotCount     int                `json:"hotspot_count"`
	ThrottledFactor  float64            `json:"throttled_factor"`
	PerfPerWatt      float64            `json:"perf_per_watt"`
	ComponentPower   map[string]float64 `json:"component_power,omitempty"`
	ComponentArea    map[string]float64 `json:"component_area,omitempty"`
}

// HealthResponse for the /health endpoint.
type HealthResponse struct {
	Status         string `json:"status"`
	ActiveEpisodes int    `json:"active_episodes"`
}
