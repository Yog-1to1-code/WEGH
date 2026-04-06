// Package graph implements a Directed Acyclic Graph (DAG) for CPU component modeling.
// Each node represents a CPU microarchitecture component (fetch unit, cache, ALU cluster, etc.)
// Edges represent data/control flow between components.
// The DAG is validated via topological sort — cycles = invalid architecture.
package graph

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

// ComponentType identifies the category of a CPU component.
type ComponentType int

const (
	CompFetchUnit ComponentType = iota
	CompDecodeUnit
	CompDispatchUnit
	CompIntALU
	CompIntMulDiv
	CompFPUnit
	CompSIMDUnit
	CompLoadUnit
	CompStoreUnit
	CompBranchPredictor
	CompReorderBuffer
	CompReservationStation
	CompL1ICache
	CompL1DCache
	CompL2Cache
	CompL3Cache
	CompMemoryController
	CompRegisterFile
	CompProgramCounter
	CompGPIO
	CompSRAM
	CompForwardingUnit
	CompHazardDetector
	CompTLB
	CompPrefetcher
	CompInterconnect
	CompPCore // Logical grouping for P-core cluster
	CompECore // Logical grouping for E-core cluster
	CompPowerManager
)

// ComponentTypeNames maps component types to human-readable names.
var ComponentTypeNames = map[ComponentType]string{
	CompFetchUnit:          "fetch_unit",
	CompDecodeUnit:         "decode_unit",
	CompDispatchUnit:       "dispatch_unit",
	CompIntALU:             "int_alu",
	CompIntMulDiv:          "int_mul_div",
	CompFPUnit:             "fp_unit",
	CompSIMDUnit:           "simd_unit",
	CompLoadUnit:           "load_unit",
	CompStoreUnit:          "store_unit",
	CompBranchPredictor:    "branch_predictor",
	CompReorderBuffer:      "reorder_buffer",
	CompReservationStation: "reservation_station",
	CompL1ICache:           "l1_icache",
	CompL1DCache:           "l1_dcache",
	CompL2Cache:            "l2_cache",
	CompL3Cache:            "l3_cache",
	CompMemoryController:   "memory_controller",
	CompRegisterFile:       "register_file",
	CompProgramCounter:     "program_counter",
	CompGPIO:               "gpio",
	CompSRAM:               "sram",
	CompForwardingUnit:     "forwarding_unit",
	CompHazardDetector:     "hazard_detector",
	CompTLB:                "tlb",
	CompPrefetcher:         "prefetcher",
	CompInterconnect:       "interconnect",
	CompPCore:              "p_core",
	CompECore:              "e_core",
	CompPowerManager:       "power_manager",
}

// ComponentTypeFromName reverses the name lookup.
var ComponentTypeFromName map[string]ComponentType

func init() {
	ComponentTypeFromName = make(map[string]ComponentType, len(ComponentTypeNames))
	for k, v := range ComponentTypeNames {
		ComponentTypeFromName[v] = k
	}
}

// Param holds a tunable parameter for a component.
type Param struct {
	Name  string  `json:"name"`
	Value float64 `json:"value"`
	Min   float64 `json:"min"`
	Max   float64 `json:"max"`
}

// Node represents a CPU component in the design DAG.
type Node struct {
	ID       string        `json:"id"`       // Unique identifier (e.g., "l1_icache_0")
	Type     ComponentType `json:"type"`     // Component category
	TypeName string        `json:"type_name"` // Human-readable type
	Params   []Param       `json:"params"`   // Tunable parameters
	Active   bool          `json:"active"`   // Whether component is enabled
}

// Edge represents a connection between two components.
type Edge struct {
	From string `json:"from"` // Source node ID
	To   string `json:"to"`   // Destination node ID
}

// DAG is the core Directed Acyclic Graph for CPU architecture modeling.
// Pre-allocated for performance. Thread-safe via mutex for concurrent episode access.
type DAG struct {
	mu    sync.RWMutex
	Nodes map[string]*Node `json:"nodes"`
	Edges []Edge           `json:"edges"`

	// Adjacency lists (rebuilt after mutations)
	adjOut map[string][]string // node -> outgoing neighbors
	adjIn  map[string][]string // node -> incoming neighbors

	// Pre-allocated buffers for topological sort (avoids GC pressure)
	topoOrder []string
	inDegree  map[string]int
	queue     []string
}

// NewDAG creates an empty DAG with pre-allocated capacity.
func NewDAG(nodeCapacity int) *DAG {
	return &DAG{
		Nodes:     make(map[string]*Node, nodeCapacity),
		Edges:     make([]Edge, 0, nodeCapacity*2),
		adjOut:    make(map[string][]string, nodeCapacity),
		adjIn:     make(map[string][]string, nodeCapacity),
		topoOrder: make([]string, 0, nodeCapacity),
		inDegree:  make(map[string]int, nodeCapacity),
		queue:     make([]string, 0, nodeCapacity),
	}
}

// AddNode adds a component to the DAG. Returns error if ID already exists.
func (d *DAG) AddNode(id string, compType ComponentType, params []Param) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if _, exists := d.Nodes[id]; exists {
		return fmt.Errorf("node '%s' already exists", id)
	}

	typeName, ok := ComponentTypeNames[compType]
	if !ok {
		return fmt.Errorf("unknown component type: %d", compType)
	}

	// Clamp params to bounds
	for i := range params {
		params[i].Value = clamp(params[i].Value, params[i].Min, params[i].Max)
	}

	d.Nodes[id] = &Node{
		ID:       id,
		Type:     compType,
		TypeName: typeName,
		Params:   params,
		Active:   true,
	}

	d.adjOut[id] = make([]string, 0, 4)
	d.adjIn[id] = make([]string, 0, 4)
	return nil
}

// RemoveNode removes a component and all its edges.
func (d *DAG) RemoveNode(id string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if _, exists := d.Nodes[id]; !exists {
		return fmt.Errorf("node '%s' not found", id)
	}

	// Remove edges involving this node
	newEdges := make([]Edge, 0, len(d.Edges))
	for _, e := range d.Edges {
		if e.From != id && e.To != id {
			newEdges = append(newEdges, e)
		}
	}
	d.Edges = newEdges

	// Clean adjacency
	delete(d.adjOut, id)
	delete(d.adjIn, id)
	for k, v := range d.adjOut {
		filtered := v[:0]
		for _, n := range v {
			if n != id {
				filtered = append(filtered, n)
			}
		}
		d.adjOut[k] = filtered
	}
	for k, v := range d.adjIn {
		filtered := v[:0]
		for _, n := range v {
			if n != id {
				filtered = append(filtered, n)
			}
		}
		d.adjIn[k] = filtered
	}

	delete(d.Nodes, id)
	return nil
}

// AddEdge connects two components. Returns error if it would create a cycle.
func (d *DAG) AddEdge(from, to string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if _, ok := d.Nodes[from]; !ok {
		return fmt.Errorf("source node '%s' not found", from)
	}
	if _, ok := d.Nodes[to]; !ok {
		return fmt.Errorf("dest node '%s' not found", to)
	}
	if from == to {
		return fmt.Errorf("self-loop not allowed: '%s'", from)
	}

	// Check for duplicate
	for _, e := range d.Edges {
		if e.From == from && e.To == to {
			return fmt.Errorf("edge '%s' → '%s' already exists", from, to)
		}
	}

	// Tentatively add and check for cycles
	d.adjOut[from] = append(d.adjOut[from], to)
	d.adjIn[to] = append(d.adjIn[to], from)

	if d.hasCycleLocked() {
		// Rollback
		d.adjOut[from] = d.adjOut[from][:len(d.adjOut[from])-1]
		d.adjIn[to] = d.adjIn[to][:len(d.adjIn[to])-1]
		return fmt.Errorf("edge '%s' → '%s' would create a cycle", from, to)
	}

	d.Edges = append(d.Edges, Edge{From: from, To: to})
	return nil
}

// SetParam updates a parameter on a component. Returns error if invalid.
func (d *DAG) SetParam(nodeID, paramName string, value float64) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	node, ok := d.Nodes[nodeID]
	if !ok {
		return fmt.Errorf("node '%s' not found", nodeID)
	}

	for i := range node.Params {
		if node.Params[i].Name == paramName {
			node.Params[i].Value = clamp(value, node.Params[i].Min, node.Params[i].Max)
			return nil
		}
	}
	return fmt.Errorf("param '%s' not found on node '%s'", paramName, nodeID)
}

// TopoSort returns a valid topological ordering using Kahn's algorithm.
// Returns nil, error if the graph contains a cycle.
func (d *DAG) TopoSort() ([]string, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.topoSortLocked()
}

func (d *DAG) topoSortLocked() ([]string, error) {
	// Reuse pre-allocated buffers
	d.topoOrder = d.topoOrder[:0]
	for k := range d.inDegree {
		delete(d.inDegree, k)
	}
	d.queue = d.queue[:0]

	// Compute in-degrees
	for id := range d.Nodes {
		d.inDegree[id] = 0
	}
	for _, e := range d.Edges {
		d.inDegree[e.To]++
	}

	// Seed queue with zero-in-degree nodes (sorted for determinism)
	for id, deg := range d.inDegree {
		if deg == 0 {
			d.queue = append(d.queue, id)
		}
	}
	sort.Strings(d.queue)

	head := 0
	for head < len(d.queue) {
		node := d.queue[head]
		head++
		d.topoOrder = append(d.topoOrder, node)

		for _, neighbor := range d.adjOut[node] {
			d.inDegree[neighbor]--
			if d.inDegree[neighbor] == 0 {
				d.queue = append(d.queue, neighbor)
			}
		}
	}

	if len(d.topoOrder) != len(d.Nodes) {
		return nil, fmt.Errorf("graph contains a cycle (processed %d of %d nodes)", len(d.topoOrder), len(d.Nodes))
	}

	result := make([]string, len(d.topoOrder))
	copy(result, d.topoOrder)
	return result, nil
}

// hasCycleLocked checks for cycles (caller must hold lock).
func (d *DAG) hasCycleLocked() bool {
	_, err := d.topoSortLocked()
	return err != nil
}

// Validate runs structural checks on the DAG.
func (d *DAG) Validate() []string {
	d.mu.RLock()
	defer d.mu.RUnlock()

	var errors []string

	// Check for cycles
	if _, err := d.topoSortLocked(); err != nil {
		errors = append(errors, err.Error())
	}

	// Check for orphan nodes (no edges at all)
	for id := range d.Nodes {
		if len(d.adjOut[id]) == 0 && len(d.adjIn[id]) == 0 && len(d.Nodes) > 1 {
			errors = append(errors, fmt.Sprintf("orphan component '%s' has no connections", id))
		}
	}

	return errors
}

// GetActiveNodes returns all active node IDs.
func (d *DAG) GetActiveNodes() []string {
	d.mu.RLock()
	defer d.mu.RUnlock()

	nodes := make([]string, 0, len(d.Nodes))
	for id, n := range d.Nodes {
		if n.Active {
			nodes = append(nodes, id)
		}
	}
	sort.Strings(nodes)
	return nodes
}

// GetEdges returns all edges as [from, to] pairs.
func (d *DAG) GetEdges() []Edge {
	d.mu.RLock()
	defer d.mu.RUnlock()
	result := make([]Edge, len(d.Edges))
	copy(result, d.Edges)
	return result
}

// GetNode returns a copy of a node.
func (d *DAG) GetNode(id string) (*Node, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	n, ok := d.Nodes[id]
	if !ok {
		return nil, fmt.Errorf("node '%s' not found", id)
	}
	// Return a copy to avoid data races
	cp := *n
	cp.Params = make([]Param, len(n.Params))
	copy(cp.Params, n.Params)
	return &cp, nil
}

// NodeCount returns the number of nodes.
func (d *DAG) NodeCount() int {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return len(d.Nodes)
}

// EdgeCount returns the number of edges.
func (d *DAG) EdgeCount() int {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return len(d.Edges)
}

// GetParamValue returns a specific parameter value from a node.
func (d *DAG) GetParamValue(nodeID, paramName string) (float64, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	node, ok := d.Nodes[nodeID]
	if !ok {
		return 0, fmt.Errorf("node '%s' not found", nodeID)
	}
	for _, p := range node.Params {
		if p.Name == paramName {
			return p.Value, nil
		}
	}
	return 0, fmt.Errorf("param '%s' not found on '%s'", paramName, nodeID)
}

// FindNodesByType returns all node IDs of a given type.
func (d *DAG) FindNodesByType(ct ComponentType) []string {
	d.mu.RLock()
	defer d.mu.RUnlock()

	var result []string
	for id, n := range d.Nodes {
		if n.Type == ct {
			result = append(result, id)
		}
	}
	sort.Strings(result)
	return result
}

func clamp(v, lo, hi float64) float64 {
	return math.Max(lo, math.Min(hi, v))
}
