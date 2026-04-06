// Package graph — component templates define the pre-built components
// available for each task difficulty level in WEGH.
package graph

// ComponentTemplate defines a reusable component with default parameters.
type ComponentTemplate struct {
	Type   ComponentType
	ID     string  // Default ID pattern
	Params []Param // Default parameters with bounds
}

// TaskComponents returns the available components for a given task.
// Task 0 = Easy (IoT), Task 1 = Medium (RV32IM), Task 2 = Hard (M-Series)
func TaskComponents(taskID int) []ComponentTemplate {
	switch taskID {
	case 0:
		return iotComponents()
	case 1:
		return rv32imComponents()
	case 2:
		return msSeriesComponents()
	default:
		return rv32imComponents()
	}
}

// iotComponents — 8-Bit IoT Microcontroller
// Simple single-cycle or 2-stage pipeline. ~12 params.
func iotComponents() []ComponentTemplate {
	return []ComponentTemplate{
		{Type: CompProgramCounter, ID: "pc", Params: []Param{
			{Name: "width_bits", Value: 16, Min: 8, Max: 32},
		}},
		{Type: CompFetchUnit, ID: "fetch", Params: []Param{
			{Name: "width", Value: 1, Min: 1, Max: 2},
		}},
		{Type: CompDecodeUnit, ID: "decode", Params: []Param{
			{Name: "width", Value: 1, Min: 1, Max: 2},
		}},
		{Type: CompRegisterFile, ID: "regfile", Params: []Param{
			{Name: "num_regs", Value: 16, Min: 8, Max: 32},
			{Name: "width_bits", Value: 8, Min: 8, Max: 16},
		}},
		{Type: CompIntALU, ID: "alu", Params: []Param{
			{Name: "count", Value: 1, Min: 1, Max: 2},
			{Name: "width_bits", Value: 8, Min: 8, Max: 16},
		}},
		{Type: CompSRAM, ID: "sram", Params: []Param{
			{Name: "size_kb", Value: 2, Min: 0.5, Max: 8},
		}},
		{Type: CompGPIO, ID: "gpio", Params: []Param{
			{Name: "pin_count", Value: 8, Min: 4, Max: 32},
		}},
		{Type: CompPowerManager, ID: "pmu", Params: []Param{
			{Name: "voltage", Value: 1.0, Min: 0.6, Max: 1.3},
			{Name: "clock_mhz", Value: 16, Min: 1, Max: 48},
		}},
	}
}

// rv32imComponents — RV32IM 5-Stage Pipelined Core
// Classic RISC-V with integer multiply/divide. ~25 params.
func rv32imComponents() []ComponentTemplate {
	return []ComponentTemplate{
		{Type: CompFetchUnit, ID: "fetch", Params: []Param{
			{Name: "width", Value: 1, Min: 1, Max: 2},
			{Name: "pipeline_reg_bits", Value: 64, Min: 32, Max: 128},
		}},
		{Type: CompDecodeUnit, ID: "decode", Params: []Param{
			{Name: "width", Value: 1, Min: 1, Max: 2},
		}},
		{Type: CompRegisterFile, ID: "regfile", Params: []Param{
			{Name: "num_regs", Value: 32, Min: 32, Max: 32},
			{Name: "read_ports", Value: 2, Min: 2, Max: 4},
			{Name: "write_ports", Value: 1, Min: 1, Max: 2},
		}},
		{Type: CompIntALU, ID: "alu", Params: []Param{
			{Name: "count", Value: 1, Min: 1, Max: 4},
		}},
		{Type: CompIntMulDiv, ID: "muldiv", Params: []Param{
			{Name: "count", Value: 1, Min: 0, Max: 2},
			{Name: "latency_cycles", Value: 4, Min: 2, Max: 8},
		}},
		{Type: CompLoadUnit, ID: "load_unit", Params: []Param{
			{Name: "count", Value: 1, Min: 1, Max: 2},
		}},
		{Type: CompStoreUnit, ID: "store_unit", Params: []Param{
			{Name: "count", Value: 1, Min: 1, Max: 2},
		}},
		{Type: CompBranchPredictor, ID: "bp", Params: []Param{
			{Name: "type", Value: 0, Min: 0, Max: 2},        // 0=static, 1=bimodal, 2=gshare
			{Name: "btb_entries", Value: 64, Min: 16, Max: 512},
		}},
		{Type: CompL1ICache, ID: "l1i", Params: []Param{
			{Name: "size_kb", Value: 16, Min: 4, Max: 64},
			{Name: "associativity", Value: 2, Min: 1, Max: 8},
			{Name: "line_bytes", Value: 32, Min: 16, Max: 64},
		}},
		{Type: CompL1DCache, ID: "l1d", Params: []Param{
			{Name: "size_kb", Value: 16, Min: 4, Max: 64},
			{Name: "associativity", Value: 2, Min: 1, Max: 8},
			{Name: "line_bytes", Value: 32, Min: 16, Max: 64},
		}},
		{Type: CompForwardingUnit, ID: "fwd", Params: []Param{
			{Name: "ex_to_ex", Value: 1, Min: 0, Max: 1},
			{Name: "mem_to_ex", Value: 1, Min: 0, Max: 1},
		}},
		{Type: CompHazardDetector, ID: "hazard", Params: []Param{
			{Name: "stall_on_load", Value: 1, Min: 0, Max: 1},
		}},
		{Type: CompMemoryController, ID: "memctrl", Params: []Param{
			{Name: "bus_width_bits", Value: 32, Min: 16, Max: 64},
			{Name: "latency_cycles", Value: 10, Min: 5, Max: 50},
		}},
		{Type: CompPowerManager, ID: "pmu", Params: []Param{
			{Name: "voltage", Value: 0.9, Min: 0.6, Max: 1.2},
			{Name: "clock_ghz", Value: 1.0, Min: 0.5, Max: 2.5},
		}},
	}
}

// msSeriesComponents — M-Series Heterogeneous Superscalar
// Inspired by Apple M-series. P-Cores + E-Cores, OoO execution. ~40 params.
func msSeriesComponents() []ComponentTemplate {
	return []ComponentTemplate{
		// === P-Core Cluster ===
		{Type: CompPCore, ID: "pcore", Params: []Param{
			{Name: "count", Value: 4, Min: 1, Max: 8},
			{Name: "pipeline_depth", Value: 14, Min: 8, Max: 20},
			{Name: "issue_width", Value: 6, Min: 4, Max: 10},
			{Name: "clock_ghz", Value: 3.5, Min: 2.0, Max: 5.0},
			{Name: "voltage", Value: 1.0, Min: 0.7, Max: 1.3},
		}},

		// === E-Core Cluster ===
		{Type: CompECore, ID: "ecore", Params: []Param{
			{Name: "count", Value: 4, Min: 0, Max: 8},
			{Name: "pipeline_depth", Value: 8, Min: 4, Max: 12},
			{Name: "issue_width", Value: 2, Min: 1, Max: 4},
			{Name: "clock_ghz", Value: 2.0, Min: 1.0, Max: 3.0},
			{Name: "voltage", Value: 0.7, Min: 0.5, Max: 1.0},
		}},

		// === Execution Units (P-Core) ===
		{Type: CompIntALU, ID: "p_alu", Params: []Param{
			{Name: "count", Value: 4, Min: 2, Max: 8},
		}},
		{Type: CompIntMulDiv, ID: "p_muldiv", Params: []Param{
			{Name: "count", Value: 2, Min: 1, Max: 4},
		}},
		{Type: CompFPUnit, ID: "p_fpu", Params: []Param{
			{Name: "count", Value: 2, Min: 1, Max: 6},
		}},
		{Type: CompSIMDUnit, ID: "p_simd", Params: []Param{
			{Name: "count", Value: 2, Min: 0, Max: 4},
			{Name: "width_bits", Value: 128, Min: 64, Max: 512},
		}},
		{Type: CompLoadUnit, ID: "p_load", Params: []Param{
			{Name: "count", Value: 2, Min: 1, Max: 4},
		}},
		{Type: CompStoreUnit, ID: "p_store", Params: []Param{
			{Name: "count", Value: 1, Min: 1, Max: 3},
		}},

		// === Out-of-Order Structures ===
		{Type: CompReorderBuffer, ID: "rob", Params: []Param{
			{Name: "entries", Value: 256, Min: 64, Max: 512},
		}},
		{Type: CompReservationStation, ID: "rs", Params: []Param{
			{Name: "entries", Value: 96, Min: 32, Max: 256},
		}},

		// === Branch Prediction ===
		{Type: CompBranchPredictor, ID: "bp", Params: []Param{
			{Name: "type", Value: 3, Min: 0, Max: 4},           // 0=bimodal..4=perceptron
			{Name: "btb_entries", Value: 4096, Min: 256, Max: 8192},
			{Name: "bht_size", Value: 8192, Min: 1024, Max: 16384},
			{Name: "ras_depth", Value: 32, Min: 8, Max: 64},
		}},

		// === Cache Hierarchy ===
		{Type: CompL1ICache, ID: "l1i", Params: []Param{
			{Name: "size_kb", Value: 64, Min: 16, Max: 128},
			{Name: "associativity", Value: 4, Min: 2, Max: 8},
		}},
		{Type: CompL1DCache, ID: "l1d", Params: []Param{
			{Name: "size_kb", Value: 64, Min: 16, Max: 128},
			{Name: "associativity", Value: 4, Min: 2, Max: 8},
		}},
		{Type: CompL2Cache, ID: "l2", Params: []Param{
			{Name: "size_kb", Value: 1024, Min: 256, Max: 4096},
			{Name: "associativity", Value: 8, Min: 4, Max: 16},
			{Name: "shared", Value: 0, Min: 0, Max: 1}, // 0=per-cluster, 1=shared
		}},
		{Type: CompL3Cache, ID: "l3", Params: []Param{
			{Name: "size_mb", Value: 16, Min: 0, Max: 64},
			{Name: "associativity", Value: 16, Min: 8, Max: 32},
		}},

		// === Memory ===
		{Type: CompTLB, ID: "tlb", Params: []Param{
			{Name: "entries", Value: 512, Min: 64, Max: 2048},
		}},
		{Type: CompMemoryController, ID: "memctrl", Params: []Param{
			{Name: "channels", Value: 4, Min: 1, Max: 8},
			{Name: "bus_width_bits", Value: 128, Min: 64, Max: 256},
			{Name: "bandwidth_gbps", Value: 100, Min: 32, Max: 256},
		}},
		{Type: CompPrefetcher, ID: "pf", Params: []Param{
			{Name: "type", Value: 2, Min: 0, Max: 3}, // 0=none, 1=stride, 2=stream, 3=multi
		}},

		// === Interconnect & Power ===
		{Type: CompInterconnect, ID: "noc", Params: []Param{
			{Name: "type", Value: 2, Min: 0, Max: 3},       // 0=bus,1=ring,2=mesh,3=crossbar
			{Name: "bandwidth_gbps", Value: 200, Min: 32, Max: 512},
		}},
		{Type: CompPowerManager, ID: "pmu", Params: []Param{
			{Name: "dvfs_states", Value: 4, Min: 2, Max: 8},
			{Name: "power_domains", Value: 3, Min: 1, Max: 6},
		}},
	}
}

// InitialEdges returns the default edges for a task.
func InitialEdges(taskID int) []Edge {
	switch taskID {
	case 0:
		return []Edge{
			{From: "pc", To: "fetch"},
			{From: "fetch", To: "decode"},
			{From: "decode", To: "regfile"},
			{From: "decode", To: "alu"},
			{From: "alu", To: "sram"},
			{From: "regfile", To: "alu"},
			{From: "sram", To: "gpio"},
			{From: "decode", To: "pmu"},
		}
	case 1:
		return []Edge{
			{From: "fetch", To: "decode"},
			{From: "decode", To: "regfile"},
			{From: "decode", To: "alu"},
			{From: "decode", To: "muldiv"},
			{From: "alu", To: "load_unit"},
			{From: "alu", To: "store_unit"},
			{From: "load_unit", To: "l1d"},
			{From: "store_unit", To: "l1d"},
			{From: "fetch", To: "l1i"},
			{From: "l1i", To: "l1d"},
			{From: "l1d", To: "memctrl"},
			{From: "decode", To: "hazard"},
			{From: "hazard", To: "fwd"},
			{From: "fetch", To: "bp"},
			{From: "decode", To: "pmu"},
		}
	case 2:
		return []Edge{
			{From: "pcore", To: "p_alu"},
			{From: "pcore", To: "p_muldiv"},
			{From: "pcore", To: "p_fpu"},
			{From: "pcore", To: "p_simd"},
			{From: "pcore", To: "p_load"},
			{From: "pcore", To: "p_store"},
			{From: "pcore", To: "rob"},
			{From: "pcore", To: "rs"},
			{From: "pcore", To: "bp"},
			{From: "pcore", To: "l1i"},
			{From: "pcore", To: "l1d"},
			{From: "ecore", To: "l1i"},
			{From: "ecore", To: "l1d"},
			{From: "l1i", To: "l2"},
			{From: "l1d", To: "l2"},
			{From: "l1d", To: "pf"},
			{From: "l2", To: "l3"},
			{From: "l3", To: "memctrl"},
			{From: "memctrl", To: "noc"},
			{From: "l1d", To: "tlb"},
			{From: "pcore", To: "pmu"},
			{From: "ecore", To: "pmu"},
		}
	default:
		return nil
	}
}

// BuildInitialGraph creates a fully wired DAG for a given task.
func BuildInitialGraph(taskID int) *DAG {
	templates := TaskComponents(taskID)
	edges := InitialEdges(taskID)

	dag := NewDAG(len(templates) + 10)
	for _, t := range templates {
		params := make([]Param, len(t.Params))
		copy(params, t.Params)
		_ = dag.AddNode(t.ID, t.Type, params)
	}
	for _, e := range edges {
		_ = dag.AddEdge(e.From, e.To)
	}
	return dag
}
