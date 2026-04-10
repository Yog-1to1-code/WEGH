// Package api implements HTTP handlers for the WEGH Go simulation engine.
// This is a self-contained OpenEnv-compliant server — no Python framework dependency.
package api

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/wegh/engine/internal/episodes"
	"github.com/wegh/engine/internal/graph"
	"github.com/wegh/engine/internal/simulator"
)

// ── Prometheus-style Metrics ─────────────────────────────────────────────────

type serverMetrics struct {
	mu             sync.Mutex
	stepCount      int64
	resetCount     int64
	stepLatencySum float64
	stepLatencyN   int64
	rewardSum      float64
	rewardN        int64
	rewardMin      float64
	rewardMax      float64
	errorCount     int64
}

var srvMetrics = &serverMetrics{
	rewardMin: math.MaxFloat64,
	rewardMax: -math.MaxFloat64,
}

func (m *serverMetrics) recordStep(latencyMs, reward float64) {
	atomic.AddInt64(&m.stepCount, 1)
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stepLatencySum += latencyMs
	m.stepLatencyN++
	m.rewardSum += reward
	m.rewardN++
	if reward < m.rewardMin {
		m.rewardMin = reward
	}
	if reward > m.rewardMax {
		m.rewardMax = reward
	}
}

func (m *serverMetrics) prometheus() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	avgLat := 0.0
	if m.stepLatencyN > 0 {
		avgLat = m.stepLatencySum / float64(m.stepLatencyN)
	}
	avgRew := 0.0
	if m.rewardN > 0 {
		avgRew = m.rewardSum / float64(m.rewardN)
	}
	rMin := m.rewardMin
	if rMin == math.MaxFloat64 {
		rMin = 0
	}
	rMax := m.rewardMax
	if rMax == -math.MaxFloat64 {
		rMax = 0
	}
	return fmt.Sprintf(`# HELP wegh_steps_total Total environment steps taken
# TYPE wegh_steps_total counter
wegh_steps_total %d

# HELP wegh_resets_total Total environment resets
# TYPE wegh_resets_total counter
wegh_resets_total %d

# HELP wegh_step_latency_ms_avg Average step latency (ms)
# TYPE wegh_step_latency_ms_avg gauge
wegh_step_latency_ms_avg %.4f

# HELP wegh_reward_avg Average reward per step
# TYPE wegh_reward_avg gauge
wegh_reward_avg %.4f

# HELP wegh_reward_min Minimum reward seen
# TYPE wegh_reward_min gauge
wegh_reward_min %.4f

# HELP wegh_reward_max Maximum reward seen
# TYPE wegh_reward_max gauge
wegh_reward_max %.4f

# HELP wegh_errors_total Total request errors
# TYPE wegh_errors_total counter
wegh_errors_total %d
`,
		atomic.LoadInt64(&m.stepCount),
		atomic.LoadInt64(&m.resetCount),
		avgLat, avgRew, rMin, rMax,
		atomic.LoadInt64(&m.errorCount),
	)
}

// ── Episode State ────────────────────────────────────────────────────────────

type episodeState struct {
	taskID         int
	taskName       string
	maxSteps       int
	step           int
	done           bool
	cumReward      float64
	prevMetrics    MetricsPayload
	replay         []ReplayEntry
	lastActions    []Action
	actionCounts   map[string]int // For exploit detection
	consecutiveSame int
}

// ── Handler ──────────────────────────────────────────────────────────────────

// Handler wraps the episode manager and exposes HTTP endpoints.
type Handler struct {
	manager  *episodes.Manager
	mu       sync.RWMutex
	states   map[string]*episodeState
	// Track current active episode for single-episode mode
	activeEpisodeID string
}

// NewHandler creates the API handler.
func NewHandler() *Handler {
	return &Handler{
		manager: episodes.NewManager(),
		states:  make(map[string]*episodeState),
	}
}

// RegisterRoutes sets up all HTTP routes.
func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/health", h.handleHealth)
	mux.HandleFunc("/ping", h.handlePing)
	mux.HandleFunc("/reset", h.handleReset)
	mux.HandleFunc("/step", h.handleStep)
	mux.HandleFunc("/state", h.handleState)
	mux.HandleFunc("/grade", h.handleGrade)
	mux.HandleFunc("/tasks", h.handleTasks)
	mux.HandleFunc("/metrics", h.handleMetrics)
	mux.HandleFunc("/replay", h.handleReplay)
}

// ── /health ──────────────────────────────────────────────────────────────────

func (h *Handler) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, HealthResponse{
		Status:         "ok",
		Version:        "1.0.0",
		ActiveEpisodes: h.manager.Count(),
	})
}

// ── /ping ────────────────────────────────────────────────────────────────────

func (h *Handler) handlePing(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// ── /reset ───────────────────────────────────────────────────────────────────

func (h *Handler) handleReset(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req ResetRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		// Allow empty body → defaults
		req = ResetRequest{}
	}

	// Map task name to ID
	taskID := req.TaskID
	if req.Task != "" {
		switch req.Task {
		case "iot_8bit":
			taskID = 0
		case "rv32im":
			taskID = 1
		case "mseries_superscalar":
			taskID = 2
		}
	}

	// Generate episode ID if not provided
	episodeID := req.EpisodeID
	if episodeID == "" {
		episodeID = fmt.Sprintf("ep_%d", time.Now().UnixNano())
	}

	// Clean up any previous episode with same ID
	h.manager.Delete(episodeID)

	// Create new episode graph
	dag, err := h.manager.Create(episodeID, taskID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ResetResponse{Status: "error", Error: err.Error()})
		return
	}

	// Run initial evaluation
	metrics := simulator.Evaluate(dag, taskID)
	metricsPayload := metricsToPayload(metrics)

	// Get task info
	taskInfo := getTaskInfo(taskID)

	// Create episode state
	h.mu.Lock()
	h.states[episodeID] = &episodeState{
		taskID:       taskID,
		taskName:     taskInfo.Name,
		maxSteps:     taskInfo.MaxSteps,
		step:         0,
		done:         false,
		cumReward:    0,
		prevMetrics:  metricsPayload,
		replay:       make([]ReplayEntry, 0, taskInfo.MaxSteps),
		lastActions:  nil,
		actionCounts: make(map[string]int),
	}
	h.activeEpisodeID = episodeID
	h.mu.Unlock()

	atomic.AddInt64(&srvMetrics.resetCount, 1)

	// Build observation
	components := dag.GetActiveNodes()
	connections := edgesToPairs(dag.GetEdges())
	obs := buildObservation(metricsPayload, components, connections, taskID, taskInfo, 0, 0, false, "", 0.001)

	resp := ResetResponse{
		Status:           "ok",
		GraphID:          episodeID,
		InitialMetrics:   metricsPayload,
		Components:       components,
		Connections:       connections,
		AvailableActions: availableActions(taskID),
		Observation:      &obs,
	}

	log.Printf("[RESET] episode=%s task=%d (%s) nodes=%d", episodeID, taskID, taskInfo.Name, dag.NodeCount())
	writeJSON(w, http.StatusOK, resp)
}

// ── /step ────────────────────────────────────────────────────────────────────

func (h *Handler) handleStep(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	start := time.Now()

	var req StepRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		atomic.AddInt64(&srvMetrics.errorCount, 1)
		writeJSON(w, http.StatusBadRequest, StepResponse{Status: "error", Error: err.Error()})
		return
	}

	// Resolve episode ID
	episodeID := req.EpisodeID
	if episodeID == "" {
		h.mu.RLock()
		episodeID = h.activeEpisodeID
		h.mu.RUnlock()
	}

	// Support OpenEnv-style flattened action fields
	action := req.Action
	if req.ActionType != "" {
		action = Action{
			Type:       req.ActionType,
			Component:  req.TargetComponent,
			ParamName:  req.ParameterName,
			Value:      req.ParameterValue,
			SourceNode: req.SourceNode,
			TargetNode: req.TargetNode,
		}
	}

	dag, err := h.manager.Get(episodeID)
	if err != nil {
		atomic.AddInt64(&srvMetrics.errorCount, 1)
		writeJSON(w, http.StatusNotFound, StepResponse{Status: "error", Error: err.Error()})
		return
	}

	// Get episode state
	h.mu.Lock()
	state, ok := h.states[episodeID]
	if !ok {
		h.mu.Unlock()
		writeJSON(w, http.StatusNotFound, StepResponse{Status: "error", Error: "episode state not found"})
		return
	}

	if state.done {
		h.mu.Unlock()
		writeJSON(w, http.StatusBadRequest, StepResponse{Status: "error", Error: "episode is done", Done: true})
		return
	}

	state.step++
	stepNum := state.step
	h.mu.Unlock()

	// Apply action to DAG
	valid, actionErr, notes := applyAction(dag, &action)

	// Validate DAG structure
	validationErrors := dag.Validate()

	// Re-evaluate metrics
	metrics := simulator.Evaluate(dag, state.taskID)
	metricsPayload := metricsToPayload(metrics)

	// Compute dense reward
	reward := computeStepReward(metricsPayload, state.prevMetrics, valid, validationErrors, state.taskID)

	// Generate engineering feedback
	engineeringNotes := notes
	if len(validationErrors) > 0 {
		engineeringNotes += " STRUCTURAL ISSUES: " + strings.Join(validationErrors, "; ")
	}
	if !valid {
		engineeringNotes = fmt.Sprintf("ACTION REJECTED: %s. %s", actionErr, engineeringNotes)
	}

	// Check if episode is done
	done := stepNum >= state.maxSteps

	// Compute final score if done
	finalScore := 0.001
	if done {
		finalScore = computeFinalScore(metricsPayload, state.taskID)
	}

	// Update state
	h.mu.Lock()
	state.cumReward += reward
	state.done = done
	state.prevMetrics = metricsPayload
	state.replay = append(state.replay, ReplayEntry{
		Step:    stepNum,
		Action:  action,
		Metrics: metricsPayload,
		Reward:  reward,
		Valid:   valid && len(validationErrors) == 0,
		Done:    done,
	})
	// Exploit detection: track action patterns
	actionKey := fmt.Sprintf("%s_%s_%s_%.0f", action.Type, action.Component, action.ParamName, action.Value)
	state.actionCounts[actionKey]++
	if len(state.lastActions) > 0 {
		lastKey := fmt.Sprintf("%s_%s_%s_%.0f", state.lastActions[len(state.lastActions)-1].Type,
			state.lastActions[len(state.lastActions)-1].Component,
			state.lastActions[len(state.lastActions)-1].ParamName,
			state.lastActions[len(state.lastActions)-1].Value)
		if lastKey == actionKey {
			state.consecutiveSame++
		} else {
			state.consecutiveSame = 0
		}
	}
	state.lastActions = append(state.lastActions, action)
	h.mu.Unlock()

	// Record metrics
	latency := float64(time.Since(start).Microseconds()) / 1000.0
	srvMetrics.recordStep(latency, reward)

	// Build observation
	components := dag.GetActiveNodes()
	connections := edgesToPairs(dag.GetEdges())
	taskInfo := getTaskInfo(state.taskID)

	feedbackStr := fmt.Sprintf("[Step %d/%d] %s %s (%s=%.1f) → %s | Reward: %+.4f | IPC=%.3f Power=%.1fmW Area=%.3fmm² Thermal=%.1f°C",
		stepNum, state.maxSteps, action.Type, action.Component, action.ParamName, action.Value,
		func() string {
			if valid {
				return "✓ VALID"
			}
			return "✗ INVALID"
		}(),
		reward, metricsPayload.IPC, metricsPayload.TotalPowerMW, metricsPayload.TotalAreaMM2, metricsPayload.ThermalCelsius)
	if engineeringNotes != "" {
		feedbackStr += "\n" + engineeringNotes
	}

	obs := buildObservation(metricsPayload, components, connections, state.taskID, taskInfo, stepNum, state.cumReward, done, feedbackStr, finalScore)

	resp := StepResponse{
		Status:           "ok",
		Valid:            valid && len(validationErrors) == 0,
		Metrics:          metricsPayload,
		Components:       components,
		Connections:       connections,
		ValidationErrors: validationErrors,
		EngineeringNotes: engineeringNotes,
		Observation:      &obs,
		Reward:           reward,
		Done:             done,
	}
	if actionErr != "" {
		resp.Error = actionErr
	}

	writeJSON(w, http.StatusOK, resp)
}

// ── /state ───────────────────────────────────────────────────────────────────

func (h *Handler) handleState(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "GET only", http.StatusMethodNotAllowed)
		return
	}

	h.mu.RLock()
	episodeID := h.activeEpisodeID
	state, ok := h.states[episodeID]
	h.mu.RUnlock()

	if !ok {
		writeJSON(w, http.StatusOK, StateResponse{Episode: 0, Step: 0, Done: true})
		return
	}

	dag, err := h.manager.Get(episodeID)
	if err != nil {
		writeJSON(w, http.StatusOK, StateResponse{Episode: 0, Step: 0, Done: true})
		return
	}

	metrics := simulator.Evaluate(dag, state.taskID)
	writeJSON(w, http.StatusOK, StateResponse{
		Episode:     1,
		Step:        state.step,
		TaskID:      state.taskID,
		TaskName:    state.taskName,
		Done:        state.done,
		Metrics:     metricsToPayload(metrics),
		Components:  dag.GetActiveNodes(),
		Connections: edgesToPairs(dag.GetEdges()),
	})
}

// ── /grade ───────────────────────────────────────────────────────────────────

func (h *Handler) handleGrade(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "GET only", http.StatusMethodNotAllowed)
		return
	}

	h.mu.RLock()
	episodeID := h.activeEpisodeID
	state, ok := h.states[episodeID]
	h.mu.RUnlock()

	if !ok {
		writeJSON(w, http.StatusOK, EpisodeGrade{Score: 0.001, SubScores: map[string]float64{}})
		return
	}

	dag, err := h.manager.Get(episodeID)
	if err != nil {
		writeJSON(w, http.StatusOK, EpisodeGrade{Score: 0.001, SubScores: map[string]float64{}})
		return
	}

	metrics := simulator.Evaluate(dag, state.taskID)
	mp := metricsToPayload(metrics)

	grade := gradeEpisode(mp, state)

	w.Header().Set("Access-Control-Allow-Origin", "*")
	writeJSON(w, http.StatusOK, grade)
}

// ── /tasks ───────────────────────────────────────────────────────────────────

func (h *Handler) handleTasks(w http.ResponseWriter, r *http.Request) {
	tasks := []TaskInfo{
		{
			ID: "iot_8bit", Name: "8-Bit IoT Microcontroller",
			Description: "Design a minimal 8-bit microcontroller for IoT sensors. Focus on ultra-low power.",
			Difficulty: "easy", MaxSteps: 20,
			Weights: map[string]float64{"power": 0.60, "throughput": 0.20, "area": 0.15, "thermal": 0.05},
		},
		{
			ID: "rv32im", Name: "RV32IM 5-Stage Pipelined Core",
			Description: "Design a RISC-V RV32IM core with a classic 5-stage pipeline. Maximize IPC within 10mm² area budget.",
			Difficulty: "medium", MaxSteps: 30,
			Weights: map[string]float64{"ipc": 0.50, "area": 0.30, "power": 0.15, "thermal": 0.05},
		},
		{
			ID: "mseries_superscalar", Name: "M-Series Heterogeneous Superscalar",
			Description: "Design an Apple M-series inspired heterogeneous CPU. Power density must stay below 1.5 W/mm².",
			Difficulty: "hard", MaxSteps: 40,
			Weights: map[string]float64{"throughput": 0.35, "efficiency": 0.25, "thermal": 0.25, "area": 0.15},
		},
	}
	writeJSON(w, http.StatusOK, tasks)
}

// ── /metrics ─────────────────────────────────────────────────────────────────

func (h *Handler) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprint(w, srvMetrics.prometheus())
}

// ── /replay ──────────────────────────────────────────────────────────────────

func (h *Handler) handleReplay(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "GET only", http.StatusMethodNotAllowed)
		return
	}

	h.mu.RLock()
	episodeID := h.activeEpisodeID
	state, ok := h.states[episodeID]
	h.mu.RUnlock()

	if !ok {
		writeJSON(w, http.StatusOK, ReplayResponse{Replay: []ReplayEntry{}, Steps: 0})
		return
	}

	h.mu.RLock()
	replay := make([]ReplayEntry, len(state.replay))
	copy(replay, state.replay)
	h.mu.RUnlock()

	w.Header().Set("Access-Control-Allow-Origin", "*")
	writeJSON(w, http.StatusOK, ReplayResponse{Replay: replay, Steps: len(replay)})
}

// ── Grading System ───────────────────────────────────────────────────────────

func gradeEpisode(metrics MetricsPayload, state *episodeState) EpisodeGrade {
	grade := EpisodeGrade{
		TaskID:    state.taskID,
		SubScores: make(map[string]float64),
		Details:   make(map[string]interface{}),
	}

	switch state.taskID {
	case 0: // IoT
		grade = gradeIoT(metrics, state, grade)
	case 1: // RV32IM
		grade = gradeRV32IM(metrics, state, grade)
	case 2: // M-Series
		grade = gradeMSeries(metrics, state, grade)
	}

	// Exploit detection
	exploitPenalty := detectExploit(state)
	if exploitPenalty > 0 {
		grade.ExploitDetected = true
		grade.PenaltyApplied = math.Min(exploitPenalty, 0.3)
		grade.Score = math.Max(0, grade.Score-grade.PenaltyApplied)
	}

	// Clamp to open interval (0, 1)
	grade.Score = clampOpen(roundTo(grade.Score, 4))
	for k, v := range grade.SubScores {
		grade.SubScores[k] = clampOpen(roundTo(v, 4))
	}

	return grade
}

func gradeIoT(metrics MetricsPayload, state *episodeState, grade EpisodeGrade) EpisodeGrade {
	maxPower := 50.0
	power := metrics.TotalPowerMW
	ipc := metrics.IPC

	var powerScore float64
	if power > maxPower {
		powerScore = math.Max(0, 1.0-(power-maxPower)/maxPower) * 0.3
	} else {
		powerScore = 1.0 - (power/maxPower)*0.3
	}

	throughputScore := math.Min(1.0, ipc/0.8)
	areaScore := math.Min(1.0, math.Max(0, 1.0-metrics.TotalAreaMM2/5.0*0.3))
	thermalScore := math.Min(1.0, math.Max(0, 1.0-math.Max(0, metrics.ThermalCelsius-80)/20.0))

	grade.SubScores["power"] = clampOpen(powerScore)
	grade.SubScores["throughput"] = clampOpen(throughputScore)
	grade.SubScores["area"] = clampOpen(areaScore)
	grade.SubScores["thermal"] = clampOpen(thermalScore)
	grade.Score = powerScore*0.60 + throughputScore*0.20 + areaScore*0.15 + thermalScore*0.05
	grade.Details["total_power_mw"] = power
	grade.Details["max_power_mw"] = maxPower
	grade.Details["ipc"] = ipc
	return grade
}

func gradeRV32IM(metrics MetricsPayload, state *episodeState, grade EpisodeGrade) EpisodeGrade {
	maxArea := 10.0
	area := metrics.TotalAreaMM2
	ipc := metrics.IPC

	var areaScore float64
	if area > maxArea {
		areaScore = math.Max(0, 1.0-(area-maxArea)/maxArea) * 0.3
	} else {
		areaScore = 1.0 - (area/maxArea)*0.2
	}

	ipcScore := math.Min(1.0, ipc/1.5)
	powerScore := math.Min(1.0, math.Max(0, 1.0-metrics.TotalPowerMW/5000.0*0.3))
	thermalScore := math.Min(1.0, math.Max(0, 1.0-math.Max(0, metrics.ThermalCelsius-80)/20.0))

	grade.SubScores["ipc"] = clampOpen(ipcScore)
	grade.SubScores["area"] = clampOpen(areaScore)
	grade.SubScores["power"] = clampOpen(powerScore)
	grade.SubScores["thermal"] = clampOpen(thermalScore)
	grade.Score = ipcScore*0.50 + areaScore*0.30 + powerScore*0.15 + thermalScore*0.05
	grade.Details["ipc"] = ipc
	grade.Details["total_area_mm2"] = area
	grade.Details["max_area_mm2"] = maxArea
	return grade
}

func gradeMSeries(metrics MetricsPayload, state *episodeState, grade EpisodeGrade) EpisodeGrade {
	maxPD := 1.5
	pd := metrics.MaxPowerDensity
	throughput := metrics.ThroughputGIPS
	ppw := metrics.PerfPerWatt
	throttle := metrics.ThrottledFactor

	throughputScore := math.Min(1.0, throughput/50.0)
	efficiencyScore := math.Min(1.0, ppw/2.0)

	var thermalScore float64
	if pd > maxPD {
		thermalScore = math.Max(0, 1.0-(pd-maxPD)/maxPD)
	} else {
		thermalScore = 1.0
	}
	thermalScore *= throttle

	areaScore := math.Min(1.0, math.Max(0, 1.0-math.Max(0, metrics.TotalAreaMM2-200)/200.0))

	grade.SubScores["throughput"] = clampOpen(throughputScore)
	grade.SubScores["efficiency"] = clampOpen(efficiencyScore)
	grade.SubScores["thermal"] = clampOpen(thermalScore)
	grade.SubScores["area"] = clampOpen(areaScore)
	grade.Score = throughputScore*0.35 + efficiencyScore*0.25 + thermalScore*0.25 + areaScore*0.15
	grade.Details["throughput_gips"] = throughput
	grade.Details["perf_per_watt"] = ppw
	grade.Details["max_power_density"] = pd
	grade.Details["throttled_factor"] = throttle
	return grade
}

func detectExploit(state *episodeState) float64 {
	if state.step < 5 {
		return 0
	}
	penalty := 0.0

	// Repeated action exploit: same action > 60% of the time
	maxCount := 0
	for _, count := range state.actionCounts {
		if count > maxCount {
			maxCount = count
		}
	}
	repeatRatio := float64(maxCount) / float64(state.step)
	if repeatRatio > 0.6 {
		penalty += (repeatRatio - 0.6) * 0.5
	}

	// Consecutive same action exploit
	if state.consecutiveSame > state.maxSteps/3 {
		penalty += 0.1
	}

	return penalty
}

// ── Step Reward Computation ──────────────────────────────────────────────────

func computeStepReward(metrics, prev MetricsPayload, valid bool, validationErrors []string, taskID int) float64 {
	reward := 0.0

	if !valid {
		reward -= 0.15
	} else {
		reward += 0.02
	}
	if len(validationErrors) > 0 {
		reward -= 0.05 * float64(len(validationErrors))
	}

	switch taskID {
	case 0: // IoT
		maxPower := 50.0
		power := metrics.TotalPowerMW
		if power <= maxPower {
			reward += 0.03 * (maxPower - power) / maxPower
		} else {
			reward -= 0.10 * math.Min((power-maxPower)/maxPower, 2.0)
		}
	case 1: // RV32IM
		maxArea := 10.0
		area := metrics.TotalAreaMM2
		if area <= maxArea {
			reward += 0.02 * (maxArea - area) / maxArea
		} else {
			reward -= 0.10 * math.Min((area-maxArea)/maxArea, 2.0)
		}
		if metrics.IPC > prev.IPC && prev.IPC > 0 {
			reward += 0.05 * math.Min((metrics.IPC-prev.IPC)/math.Max(prev.IPC, 0.1), 1.0)
		}
	case 2: // M-Series
		maxPD := 1.5
		pd := metrics.MaxPowerDensity
		if pd <= maxPD {
			reward += 0.04 * (maxPD - pd) / maxPD
		} else {
			reward -= 0.06 * math.Min(math.Log2(pd/maxPD+1), 3.0)
		}
		if metrics.ThrottledFactor < 1.0 {
			reward -= 0.03 * (1.0 - metrics.ThrottledFactor)
		}
		if metrics.ThroughputGIPS > prev.ThroughputGIPS && prev.ThroughputGIPS > 0 {
			reward += 0.05 * math.Min((metrics.ThroughputGIPS-prev.ThroughputGIPS)/prev.ThroughputGIPS, 1.0)
		}
		if pd < prev.MaxPowerDensity && prev.MaxPowerDensity > 0 {
			reward += 0.04 * math.Min((prev.MaxPowerDensity-pd)/math.Max(prev.MaxPowerDensity, 0.01), 1.0)
		}
	}

	// Pareto improvement bonus
	improvements := 0
	if metrics.IPC > prev.IPC { improvements++ }
	if metrics.PerfPerWatt > prev.PerfPerWatt { improvements++ }
	if metrics.TotalPowerMW < prev.TotalPowerMW { improvements++ }
	if metrics.TotalAreaMM2 < prev.TotalAreaMM2 { improvements++ }
	if metrics.MaxPowerDensity < prev.MaxPowerDensity { improvements++ }
	if improvements >= 3 {
		reward += 0.05
	}

	return roundTo(reward, 4)
}

// ── Final Score ──────────────────────────────────────────────────────────────

func computeFinalScore(metrics MetricsPayload, taskID int) float64 {
	score := 0.0
	switch taskID {
	case 0:
		power := metrics.TotalPowerMW
		maxPower := 50.0
		ipc := metrics.IPC
		var powerScore float64
		if power > maxPower {
			powerScore = math.Max(0, 1.0-(power-maxPower)/maxPower) * 0.3
		} else {
			powerScore = 1.0 - (power/maxPower)*0.3
		}
		throughputScore := math.Min(1.0, ipc/0.8)
		score = 0.6*powerScore + 0.4*throughputScore
	case 1:
		area := metrics.TotalAreaMM2
		maxArea := 10.0
		ipc := metrics.IPC
		var areaScore float64
		if area > maxArea {
			areaScore = math.Max(0, 1.0-(area-maxArea)/maxArea) * 0.3
		} else {
			areaScore = 1.0 - (area/maxArea)*0.2
		}
		ipcScore := math.Min(1.0, ipc/1.5)
		ppwScore := math.Min(1.0, metrics.PerfPerWatt/5.0)
		score = 0.5*ipcScore + 0.4*areaScore + 0.1*ppwScore
	case 2:
		throughput := metrics.ThroughputGIPS
		ppw := metrics.PerfPerWatt
		pd := metrics.MaxPowerDensity
		maxPD := 1.5
		throttle := metrics.ThrottledFactor
		throughputScore := math.Min(1.0, throughput/50.0)
		efficiencyScore := math.Min(1.0, ppw/2.0)
		var thermalScore float64
		if pd > maxPD {
			thermalScore = math.Max(0, 1.0-(pd-maxPD)/maxPD)
		} else {
			thermalScore = 1.0
		}
		thermalScore *= throttle
		score = 0.35*throughputScore + 0.25*efficiencyScore + 0.25*thermalScore + 0.15*math.Min(1.0, throttle)
	}
	return roundTo(math.Max(0, math.Min(1, score)), 4)
}

// ── Helpers ──────────────────────────────────────────────────────────────────

func buildObservation(m MetricsPayload, components []string, connections [][2]string,
	taskID int, taskInfo TaskInfo, step int, cumReward float64, done bool, feedback string, finalScore float64) ObservationModel {

	compJSON, _ := json.Marshal(components)

	return ObservationModel{
		CurrentEstimatedIPC: m.IPC,
		ThroughputGIPS:      m.ThroughputGIPS,
		EffectiveClockGHz:   m.EffectiveClockGHz,
		TotalPowerMW:        m.TotalPowerMW,
		TotalAreaMM2:        m.TotalAreaMM2,
		MaxPowerDensity:     m.MaxPowerDensity,
		ThermalCelsius:      m.ThermalCelsius,
		HotspotCount:        m.HotspotCount,
		ThrottledFactor:     m.ThrottledFactor,
		PerfPerWatt:         m.PerfPerWatt,
		ActiveComponents:    string(compJSON),
		ComponentCount:      len(components),
		ConnectionCount:     len(connections),
		TaskID:              taskID,
		TaskName:            taskInfo.Name,
		TaskConstraints:     getConstraintText(taskID),
		StepNumber:          step,
		MaxSteps:            taskInfo.MaxSteps,
		CumulativeReward:    roundTo(cumReward, 4),
		FeedbackString:      feedback,
		FinalScore:          clampOpen(finalScore),
		Done:                done,
		Reward:              0,
	}
}

func getTaskInfo(taskID int) TaskInfo {
	switch taskID {
	case 0:
		return TaskInfo{ID: "iot_8bit", Name: "8-Bit IoT Microcontroller", Difficulty: "easy", MaxSteps: 20,
			Weights: map[string]float64{"power": 0.60, "throughput": 0.20, "area": 0.15, "thermal": 0.05}}
	case 1:
		return TaskInfo{ID: "rv32im", Name: "RV32IM 5-Stage Pipelined Core", Difficulty: "medium", MaxSteps: 30,
			Weights: map[string]float64{"ipc": 0.50, "area": 0.30, "power": 0.15, "thermal": 0.05}}
	case 2:
		return TaskInfo{ID: "mseries_superscalar", Name: "M-Series Heterogeneous Superscalar", Difficulty: "hard", MaxSteps: 40,
			Weights: map[string]float64{"throughput": 0.35, "efficiency": 0.25, "thermal": 0.25, "area": 0.15}}
	default:
		return getTaskInfo(1)
	}
}

func getConstraintText(taskID int) string {
	switch taskID {
	case 0:
		return "POWER BUDGET: Total power MUST stay below 50mW. Battery-powered IoT sensor."
	case 1:
		return "AREA BUDGET: Total die area MUST stay below 10mm² (7nm). Maximize IPC."
	case 2:
		return "THERMAL: Power density MUST stay below 1.5 W/mm². Exceeding triggers throttling."
	default:
		return ""
	}
}

func applyAction(dag *graph.DAG, a *Action) (bool, string, string) {
	switch a.Type {
	case "add_component":
		return addComponent(dag, a)
	case "remove_component":
		return removeComponent(dag, a)
	case "resize":
		return resizeComponent(dag, a)
	case "connect":
		return connectComponents(dag, a)
	case "configure":
		return configureComponent(dag, a)
	default:
		return false, fmt.Sprintf("unknown action type '%s'", a.Type), ""
	}
}

func addComponent(dag *graph.DAG, a *Action) (bool, string, string) {
	compType, ok := graph.ComponentTypeFromName[a.Component]
	if !ok {
		return false, fmt.Sprintf("unknown component type '%s'", a.Component), ""
	}

	id := a.Component
	if _, err := dag.GetNode(id); err == nil {
		for i := 1; i < 10; i++ {
			id = fmt.Sprintf("%s_%d", a.Component, i)
			if _, err := dag.GetNode(id); err != nil {
				break
			}
		}
	}

	templates := graph.TaskComponents(inferTaskID(dag))
	var params []graph.Param
	for _, t := range templates {
		if t.Type == compType {
			params = make([]graph.Param, len(t.Params))
			copy(params, t.Params)
			break
		}
	}
	if params == nil {
		params = []graph.Param{{Name: "count", Value: 1, Min: 0, Max: 10}}
	}

	if err := dag.AddNode(id, compType, params); err != nil {
		return false, err.Error(), ""
	}

	notes := fmt.Sprintf("Added %s component '%s' with default parameters.", graph.ComponentTypeNames[compType], id)
	return true, "", notes
}

func removeComponent(dag *graph.DAG, a *Action) (bool, string, string) {
	node, err := dag.GetNode(a.Component)
	if err != nil {
		return false, err.Error(), ""
	}

	if err := dag.RemoveNode(a.Component); err != nil {
		return false, err.Error(), ""
	}

	notes := fmt.Sprintf("Removed %s '%s'. All connections severed.", node.TypeName, a.Component)
	return true, "", notes
}

func resizeComponent(dag *graph.DAG, a *Action) (bool, string, string) {
	_, err := dag.GetNode(a.Component)
	if err != nil {
		return false, err.Error(), ""
	}

	oldVal := 0.0
	node, _ := dag.GetNode(a.Component)
	for _, p := range node.Params {
		if p.Name == a.ParamName {
			oldVal = p.Value
			break
		}
	}

	if err := dag.SetParam(a.Component, a.ParamName, a.Value); err != nil {
		return false, err.Error(), ""
	}

	newVal, _ := dag.GetParamValue(a.Component, a.ParamName)
	delta := newVal - oldVal

	var notes string
	if delta > 0 {
		notes = fmt.Sprintf("Increased %s.%s from %.1f to %.1f (+%.1f).", a.Component, a.ParamName, oldVal, newVal, delta)
	} else if delta < 0 {
		notes = fmt.Sprintf("Decreased %s.%s from %.1f to %.1f (%.1f).", a.Component, a.ParamName, oldVal, newVal, delta)
	} else {
		notes = fmt.Sprintf("Parameter %s.%s unchanged at %.1f (clamped to bounds).", a.Component, a.ParamName, newVal)
	}

	if strings.Contains(a.ParamName, "size_kb") || strings.Contains(a.ParamName, "size_mb") {
		if delta > 0 {
			notes += " Larger cache improves hit rate but increases area and power."
		} else if delta < 0 {
			notes += " Smaller cache saves area but may increase miss rate."
		}
	} else if a.ParamName == "count" {
		notes += " More execution units reduce structural stalls but increase area and power."
	} else if a.ParamName == "pipeline_depth" {
		if delta > 0 {
			notes += " Deeper pipeline allows higher clock but worsens branch penalty."
		} else if delta < 0 {
			notes += " Shallower pipeline reduces branch penalty but may limit max frequency."
		}
	} else if a.ParamName == "entries" && strings.Contains(a.Component, "rob") {
		notes += " Larger ROB enables more ILP in OoO execution."
	}

	return true, "", notes
}

func connectComponents(dag *graph.DAG, a *Action) (bool, string, string) {
	from := a.SourceNode
	to := a.TargetNode
	if from == "" {
		from = a.Component
	}
	if to == "" {
		to = a.ParamName
	}

	if err := dag.AddEdge(from, to); err != nil {
		return false, err.Error(), ""
	}

	notes := fmt.Sprintf("Connected %s → %s. Data/control path established.", from, to)
	return true, "", notes
}

func configureComponent(dag *graph.DAG, a *Action) (bool, string, string) {
	return resizeComponent(dag, a)
}

func inferTaskID(dag *graph.DAG) int {
	if dag.NodeCount() <= 10 {
		return 0
	}
	pcores := dag.FindNodesByType(graph.CompPCore)
	if len(pcores) > 0 {
		return 2
	}
	return 1
}

func metricsToPayload(m simulator.Metrics) MetricsPayload {
	return MetricsPayload{
		IPC:               roundTo(m.IPC, 4),
		ThroughputGIPS:    roundTo(m.ThroughputGIPS, 4),
		EffectiveClockGHz: roundTo(m.EffectiveClockGHz, 3),
		TotalPowerMW:      roundTo(m.TotalPowerMW, 2),
		TotalAreaMM2:      roundTo(m.TotalAreaMM2, 3),
		MaxPowerDensity:   roundTo(m.MaxPowerDensity, 4),
		ThermalCelsius:    roundTo(m.ThermalCelsius, 1),
		HotspotCount:      m.HotspotCount,
		ThrottledFactor:   roundTo(m.ThrottledFactor, 3),
		PerfPerWatt:       roundTo(m.PerfPerWatt, 4),
		ComponentPower:    m.ComponentPower,
		ComponentArea:     m.ComponentArea,
	}
}

func edgesToPairs(edges []graph.Edge) [][2]string {
	pairs := make([][2]string, len(edges))
	for i, e := range edges {
		pairs[i] = [2]string{e.From, e.To}
	}
	return pairs
}

func availableActions(taskID int) []string {
	return []string{"add_component", "remove_component", "resize", "connect", "configure"}
}

func clampOpen(score float64) float64 {
	const eps = 1e-6
	if score <= 0.0 {
		return eps
	}
	if score >= 1.0 {
		return 1.0 - eps
	}
	return score
}

func roundTo(v float64, decimals int) float64 {
	pow := math.Pow(10, float64(decimals))
	return math.Round(v*pow) / pow
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}
