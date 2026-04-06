// Package api implements HTTP handlers for the Go simulation engine.
package api

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"strings"

	"github.com/wegh/engine/internal/episodes"
	"github.com/wegh/engine/internal/graph"
	"github.com/wegh/engine/internal/simulator"
)

// Handler wraps the episode manager and exposes HTTP endpoints.
type Handler struct {
	manager *episodes.Manager
}

// NewHandler creates the API handler.
func NewHandler() *Handler {
	return &Handler{
		manager: episodes.NewManager(),
	}
}

// RegisterRoutes sets up HTTP routes.
func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/health", h.handleHealth)
	mux.HandleFunc("/reset", h.handleReset)
	mux.HandleFunc("/step", h.handleStep)
}

func (h *Handler) handleHealth(w http.ResponseWriter, r *http.Request) {
	resp := HealthResponse{
		Status:         "ok",
		ActiveEpisodes: h.manager.Count(),
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *Handler) handleReset(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req ResetRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, ResetResponse{Status: "error", Error: err.Error()})
		return
	}

	// Clean up any previous episode with same ID
	h.manager.Delete(req.EpisodeID)

	// Create new episode graph
	dag, err := h.manager.Create(req.EpisodeID, req.TaskID)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ResetResponse{Status: "error", Error: err.Error()})
		return
	}

	// Run initial evaluation
	metrics := simulator.Evaluate(dag, req.TaskID)

	// Build response
	resp := ResetResponse{
		Status:         "ok",
		GraphID:        req.EpisodeID,
		InitialMetrics: metricsToPayload(metrics),
		Components:     dag.GetActiveNodes(),
		Connections:     edgesToPairs(dag.GetEdges()),
		AvailableActions: availableActions(req.TaskID),
	}

	log.Printf("[RESET] episode=%s task=%d nodes=%d", req.EpisodeID, req.TaskID, dag.NodeCount())
	writeJSON(w, http.StatusOK, resp)
}

func (h *Handler) handleStep(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req StepRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, StepResponse{Status: "error", Error: err.Error()})
		return
	}

	dag, err := h.manager.Get(req.EpisodeID)
	if err != nil {
		writeJSON(w, http.StatusNotFound, StepResponse{Status: "error", Error: err.Error()})
		return
	}

	// Apply action to DAG
	valid, actionErr, notes := applyAction(dag, &req.Action)

	// Validate DAG structure
	validationErrors := dag.Validate()

	// Re-evaluate metrics
	taskID := inferTaskID(dag) // Determine task from graph structure
	metrics := simulator.Evaluate(dag, taskID)

	// Generate engineering feedback
	engineeringNotes := notes
	if len(validationErrors) > 0 {
		engineeringNotes += " STRUCTURAL ISSUES: " + strings.Join(validationErrors, "; ")
	}
	if !valid {
		engineeringNotes = fmt.Sprintf("ACTION REJECTED: %s. %s", actionErr, engineeringNotes)
	}

	resp := StepResponse{
		Status:           "ok",
		Valid:            valid && len(validationErrors) == 0,
		Metrics:          metricsToPayload(metrics),
		Components:       dag.GetActiveNodes(),
		Connections:      edgesToPairs(dag.GetEdges()),
		ValidationErrors: validationErrors,
		EngineeringNotes: engineeringNotes,
	}
	if actionErr != "" {
		resp.Error = actionErr
	}

	writeJSON(w, http.StatusOK, resp)
}

// applyAction modifies the DAG based on the agent's action.
// Returns (valid, errorMsg, engineeringNotes).
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

	// Generate unique ID
	id := a.Component
	if _, err := dag.GetNode(id); err == nil {
		// Already exists, try numbered variant
		for i := 1; i < 10; i++ {
			id = fmt.Sprintf("%s_%d", a.Component, i)
			if _, err := dag.GetNode(id); err != nil {
				break
			}
		}
	}

	// Create with default params for this type
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

	notes := fmt.Sprintf("Removed %s '%s'. All connections to/from it have been severed.", node.TypeName, a.Component)
	return true, "", notes
}

func resizeComponent(dag *graph.DAG, a *Action) (bool, string, string) {
	node, err := dag.GetNode(a.Component)
	if err != nil {
		return false, err.Error(), ""
	}

	oldVal := 0.0
	for _, p := range node.Params {
		if p.Name == a.ParamName {
			oldVal = p.Value
			break
		}
	}

	if err := dag.SetParam(a.Component, a.ParamName, a.Value); err != nil {
		return false, err.Error(), ""
	}

	// Get the clamped value
	newVal, _ := dag.GetParamValue(a.Component, a.ParamName)

	var notes string
	delta := newVal - oldVal
	if delta > 0 {
		notes = fmt.Sprintf("Increased %s.%s from %.1f to %.1f (+%.1f). ", a.Component, a.ParamName, oldVal, newVal, delta)
	} else if delta < 0 {
		notes = fmt.Sprintf("Decreased %s.%s from %.1f to %.1f (%.1f). ", a.Component, a.ParamName, oldVal, newVal, delta)
	} else {
		notes = fmt.Sprintf("Parameter %s.%s unchanged at %.1f (value was clamped to bounds). ", a.Component, a.ParamName, newVal)
	}

	// Add domain-specific advice
	if strings.Contains(a.ParamName, "size_kb") || strings.Contains(a.ParamName, "size_mb") {
		if delta > 0 {
			notes += "Larger cache improves hit rate but increases area and power."
		} else {
			notes += "Smaller cache saves area but may increase miss rate."
		}
	} else if a.ParamName == "count" {
		notes += "More execution units reduce structural stalls but increase area and power."
	} else if a.ParamName == "pipeline_depth" {
		if delta > 0 {
			notes += "Deeper pipeline allows higher clock but worsens branch misprediction penalty."
		} else {
			notes += "Shallower pipeline reduces branch penalty but may limit max frequency."
		}
	} else if a.ParamName == "entries" && strings.Contains(a.Component, "rob") {
		notes += "Larger ROB enables more instruction-level parallelism in OoO execution."
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
		to = a.ParamName // Overloaded field for simplicity
	}

	if err := dag.AddEdge(from, to); err != nil {
		return false, err.Error(), ""
	}

	notes := fmt.Sprintf("Connected %s → %s. Data/control path established.", from, to)
	return true, "", notes
}

func configureComponent(dag *graph.DAG, a *Action) (bool, string, string) {
	// Same as resize, but semantically for non-size parameters
	return resizeComponent(dag, a)
}

// inferTaskID guesses the task from graph structure (heuristic).
func inferTaskID(dag *graph.DAG) int {
	if dag.NodeCount() <= 10 {
		return 0 // IoT
	}
	pcores := dag.FindNodesByType(graph.CompPCore)
	if len(pcores) > 0 {
		return 2 // M-Series
	}
	return 1 // RV32IM
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

func roundTo(v float64, decimals int) float64 {
	pow := math.Pow(10, float64(decimals))
	return math.Round(v*pow) / pow
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}
