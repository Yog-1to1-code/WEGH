// Package simulator implements hybrid CPU PPA evaluation.
// Stage 2: IPC is computed via cycle-accurate pipeline simulation (internal/pipeline).
// Power, Area, Thermal still use analytical models.
// Speed target: < 500μs per evaluation (pipeline sim + analytical).
//
// Sources & Empirical Grounding:
// - [Borkar1999] "Design challenges of technology scaling", IEEE Micro, 1999 (Power density limits scaling).
// - [Skadron2003] "Temperature-aware microarchitecture", ISCA 2003 (HotSpot thermal RC-models).
// - [McPAT2009] "McPAT: An integrated power, area, and timing modeling framework for multicore and manycore architectures", MICRO 2009.
// - [Jacob2007] "Memory Systems: Cache, DRAM, Disk", 2007 (Cache area/power scaling heuristics).
// - [AppleM1_2020] Apple M1 Architecture disclosures & AnandTech microarchitecture analysis (Heterogeneous P/E cores scheduling, ROB scaling).
package simulator
import (
	"math"

	cachepkg "github.com/wegh/engine/internal/cache"
	"github.com/wegh/engine/internal/graph"
	"github.com/wegh/engine/internal/pipeline"
)

// Metrics holds the complete PPA evaluation result for a CPU design.
type Metrics struct {
	// Performance
	IPC              float64 `json:"ipc"`                // Instructions per clock per core
	ThroughputGIPS   float64 `json:"throughput_gips"`    // Giga-instructions per second (all cores)
	EffectiveClockGHz float64 `json:"effective_clock_ghz"` // After thermal throttling

	// Power
	TotalPowerMW    float64            `json:"total_power_mw"`
	DynamicPowerMW  float64            `json:"dynamic_power_mw"`
	StaticPowerMW   float64            `json:"static_power_mw"`
	ComponentPower  map[string]float64 `json:"component_power"`

	// Area
	TotalAreaMM2    float64            `json:"total_area_mm2"`
	ComponentArea   map[string]float64 `json:"component_area"`

	// Thermal
	MaxPowerDensity float64 `json:"max_power_density"`  // W/mm² (peak hotspot)
	ThermalCelsius  float64 `json:"thermal_celsius"`    // Estimated junction temp
	HotspotCount    int     `json:"hotspot_count"`      // Components exceeding safe PD
	ThrottledFactor float64 `json:"throttled_factor"`   // 1.0 = no throttle, 0.5 = halved

	// Derived
	PerfPerWatt float64 `json:"perf_per_watt"`
	AreaEfficiency float64 `json:"area_efficiency"` // throughput / area
}

// Evaluate runs hybrid analytical + cycle-accurate models on the DAG.
// Pipeline IPC is computed via cycle-accurate simulation (~80μs).
// Power, Area, Thermal use analytical models (~20μs).
func Evaluate(dag *graph.DAG, taskID int) Metrics {
	m := Metrics{
		ComponentPower: make(map[string]float64),
		ComponentArea:  make(map[string]float64),
		ThrottledFactor: 1.0,
	}

	// Extract configuration parameters from the DAG nodes
	cfg := extractConfig(dag, taskID)

	// Analytical models for power, area, thermal
	computeArea(dag, &cfg, &m)
	computePower(dag, &cfg, &m)
	computeThermal(&cfg, &m)

	// Cycle-accurate pipeline simulation for IPC
	computeIPCSimulated(&cfg, &m)

	// Derived metrics
	if m.TotalPowerMW > 0 {
		m.PerfPerWatt = m.ThroughputGIPS / (m.TotalPowerMW / 1000.0)
	}
	if m.TotalAreaMM2 > 0 {
		m.AreaEfficiency = m.ThroughputGIPS / m.TotalAreaMM2
	}

	return m
}

// config holds extracted numeric values from DAG nodes for math models.
type config struct {
	TaskID int

	// Core config
	PCoreCount    float64
	PPipeDepth    float64
	PIssueWidth   float64
	PClockGHz     float64
	PVoltage      float64
	ECoreCount    float64
	EPipeDepth    float64
	EIssueWidth   float64
	EClockGHz     float64
	EVoltage      float64

	// Execution units
	IntALUs   float64
	MulDivs   float64
	FPUnits   float64
	SIMDUnits float64
	SIMDWidth float64
	LoadUnits float64
	StoreUnits float64

	// OoO
	ROBSize float64
	RSSize  float64

	// Branch prediction
	BPType     float64
	BTBEntries float64
	BHTSize    float64
	RASDepth   float64

	// Cache
	L1ISizeKB float64
	L1IAssoc  float64
	L1DSizeKB float64
	L1DAssoc  float64
	L2SizeKB  float64
	L2Assoc   float64
	L3SizeMB  float64
	L3Assoc   float64

	// Memory
	TLBEntries   float64
	MemChannels  float64
	MemBWGbps    float64
	PrefetchType float64

	// Interconnect
	NocType float64
	NocBW   float64

	// Power
	Voltage  float64
	ClockGHz float64
	ClockMHz float64

	// IoT specific
	SRAMSizeKB float64
	GPIOPins   float64
}

func extractConfig(dag *graph.DAG, taskID int) config {
	c := config{TaskID: taskID}

	getParam := func(nodeID, param string, fallback float64) float64 {
		v, err := dag.GetParamValue(nodeID, param)
		if err != nil {
			return fallback
		}
		return v
	}

	switch taskID {
	case 0: // IoT
		c.IntALUs = getParam("alu", "count", 1)
		c.SRAMSizeKB = getParam("sram", "size_kb", 2)
		c.GPIOPins = getParam("gpio", "pin_count", 8)
		c.Voltage = getParam("pmu", "voltage", 1.0)
		c.ClockMHz = getParam("pmu", "clock_mhz", 16)
		c.ClockGHz = c.ClockMHz / 1000.0
		c.PClockGHz = c.ClockGHz
		c.PCoreCount = 1
		c.PPipeDepth = 1
		c.PIssueWidth = 1

	case 1: // RV32IM
		c.IntALUs = getParam("alu", "count", 1)
		c.MulDivs = getParam("muldiv", "count", 1)
		c.LoadUnits = getParam("load_unit", "count", 1)
		c.StoreUnits = getParam("store_unit", "count", 1)
		c.BPType = getParam("bp", "type", 0)
		c.BTBEntries = getParam("bp", "btb_entries", 64)
		c.L1ISizeKB = getParam("l1i", "size_kb", 16)
		c.L1IAssoc = getParam("l1i", "associativity", 2)
		c.L1DSizeKB = getParam("l1d", "size_kb", 16)
		c.L1DAssoc = getParam("l1d", "associativity", 2)
		c.Voltage = getParam("pmu", "voltage", 0.9)
		c.ClockGHz = getParam("pmu", "clock_ghz", 1.0)
		c.PCoreCount = 1
		c.PPipeDepth = 5
		c.PIssueWidth = getParam("decode", "width", 1)
		c.PClockGHz = c.ClockGHz

	case 2: // M-Series
		c.PCoreCount = getParam("pcore", "count", 4)
		c.PPipeDepth = getParam("pcore", "pipeline_depth", 14)
		c.PIssueWidth = getParam("pcore", "issue_width", 6)
		c.PClockGHz = getParam("pcore", "clock_ghz", 3.5)
		c.PVoltage = getParam("pcore", "voltage", 1.0)
		c.ECoreCount = getParam("ecore", "count", 4)
		c.EPipeDepth = getParam("ecore", "pipeline_depth", 8)
		c.EIssueWidth = getParam("ecore", "issue_width", 2)
		c.EClockGHz = getParam("ecore", "clock_ghz", 2.0)
		c.EVoltage = getParam("ecore", "voltage", 0.7)
		c.IntALUs = getParam("p_alu", "count", 4)
		c.MulDivs = getParam("p_muldiv", "count", 2)
		c.FPUnits = getParam("p_fpu", "count", 2)
		c.SIMDUnits = getParam("p_simd", "count", 2)
		c.SIMDWidth = getParam("p_simd", "width_bits", 128)
		c.LoadUnits = getParam("p_load", "count", 2)
		c.StoreUnits = getParam("p_store", "count", 1)
		c.ROBSize = getParam("rob", "entries", 256)
		c.RSSize = getParam("rs", "entries", 96)
		c.BPType = getParam("bp", "type", 3)
		c.BTBEntries = getParam("bp", "btb_entries", 4096)
		c.BHTSize = getParam("bp", "bht_size", 8192)
		c.RASDepth = getParam("bp", "ras_depth", 32)
		c.L1ISizeKB = getParam("l1i", "size_kb", 64)
		c.L1IAssoc = getParam("l1i", "associativity", 4)
		c.L1DSizeKB = getParam("l1d", "size_kb", 64)
		c.L1DAssoc = getParam("l1d", "associativity", 4)
		c.L2SizeKB = getParam("l2", "size_kb", 1024)
		c.L2Assoc = getParam("l2", "associativity", 8)
		c.L3SizeMB = getParam("l3", "size_mb", 16)
		c.L3Assoc = getParam("l3", "associativity", 16)
		c.TLBEntries = getParam("tlb", "entries", 512)
		c.MemChannels = getParam("memctrl", "channels", 4)
		c.MemBWGbps = getParam("memctrl", "bandwidth_gbps", 100)
		c.PrefetchType = getParam("pf", "type", 2)
		c.NocType = getParam("noc", "type", 2)
		c.NocBW = getParam("noc", "bandwidth_gbps", 200)
		c.Voltage = c.PVoltage
		c.ClockGHz = c.PClockGHz
	}
	return c
}

// === IPC MODEL (CYCLE-ACCURATE) ===

// computeIPCSimulated builds a pipeline engine from the DAG config,
// generates a workload-representative instruction trace, and simulates
// it cycle-by-cycle to produce real IPC numbers.
//
// Stage 2: Cycle-accurate pipeline simulation
// Stage 3: Real cache hierarchy simulation for memory latency
func computeIPCSimulated(cfg *config, m *Metrics) {
	// Build pipeline configuration from DAG parameters
	pCfg := pipeline.PipelineConfig{
		Depth:          int(cfg.PPipeDepth),
		IssueWidth:     int(cfg.PIssueWidth),
		IntALUs:        int(cfg.IntALUs),
		MulDivs:        int(cfg.MulDivs),
		FPUnits:        int(cfg.FPUnits),
		SIMDUnits:      int(cfg.SIMDUnits),
		LoadUnits:      int(cfg.LoadUnits),
		StoreUnits:     int(cfg.StoreUnits),
		ROBSize:        int(cfg.ROBSize),
		RSSize:         int(cfg.RSSize),
		BranchAccuracy: branchAccuracyFromConfig(cfg),
		ClockGHz:       cfg.PClockGHz,
		TaskID:         cfg.TaskID,
	}

	// Trace size: enough for statistical significance, small enough for speed.
	// 5K instructions ≈ 40μs simulation time on Apple M1.
	traceSize := 5000
	maxCycles := traceSize * 4 // Generous cycle budget

	// Generate workload trace
	traceCfg := pipeline.DefaultTraceConfig(cfg.TaskID)
	trace := pipeline.GenerateTrace(traceCfg, traceSize, 42)

	// === Stage 3: Cache Hierarchy Simulation ===
	// Build cache hierarchy from DAG config and compute avg memory latency.
	cacheCfg := cachepkg.HierarchyConfig{
		L1ISizeKB:   int(cfg.L1ISizeKB),
		L1IAssoc:    int(cfg.L1IAssoc),
		L1DSizeKB:   int(cfg.L1DSizeKB),
		L1DAssoc:    int(cfg.L1DAssoc),
		L2SizeKB:    int(cfg.L2SizeKB),
		L2Assoc:     int(cfg.L2Assoc),
		L3SizeMB:    int(cfg.L3SizeMB),
		L3Assoc:     int(cfg.L3Assoc),
		LineSize:    64,
		DRAMLatency: 200,
		PrefetchType: int(cfg.PrefetchType),
	}
	cacheHier := cachepkg.NewHierarchy(cacheCfg)

	// Run the trace's memory addresses through the cache hierarchy
	// to compute real hit rates and average memory latency.
	for i := range trace {
		if trace[i].IsMemory() {
			addr := uint64(trace[i].MemAddr)
			latency := cacheHier.AccessData(addr, trace[i].IsStore())
			// Inject real cache latency into the instruction's execution time.
			// Base load latency (1 cycle) is replaced by actual cache latency.
			if latency > 1 {
				trace[i].Latency = uint8(min(latency, 255))
			}
		}
	}

	// Run cycle-accurate simulation for P-cores (with real cache latencies injected)
	engine := pipeline.NewEngine(pCfg)
	result := engine.Simulate(trace, maxCycles)
	pIPC := result.IPC

	// E-core simulation (if present): simpler pipeline, same cache
	eIPC := 0.0
	if cfg.ECoreCount > 0 && cfg.EIssueWidth > 0 {
		eCfg := pipeline.PipelineConfig{
			Depth:          int(cfg.EPipeDepth),
			IssueWidth:     int(cfg.EIssueWidth),
			IntALUs:        int(math.Max(cfg.IntALUs*0.5, 1)),
			MulDivs:        1,
			LoadUnits:      1,
			StoreUnits:     1,
			BranchAccuracy: branchAccuracyFromConfig(cfg) * 0.95, // Simpler predictor
			ClockGHz:       cfg.EClockGHz,
			TaskID:         cfg.TaskID,
		}
		eEngine := pipeline.NewEngine(eCfg)
		eResult := eEngine.Simulate(trace, maxCycles)
		eIPC = eResult.IPC
	}

	m.IPC = pIPC

	// Apply thermal throttling to clock
	clockP := cfg.PClockGHz * m.ThrottledFactor
	clockE := cfg.EClockGHz * m.ThrottledFactor
	m.EffectiveClockGHz = clockP

	// Multi-core throughput
	pThroughput := pIPC * clockP * cfg.PCoreCount
	eThroughput := eIPC * clockE * cfg.ECoreCount
	m.ThroughputGIPS = pThroughput + eThroughput
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// branchAccuracyFromConfig converts the DAG's branch predictor type
// into a prediction accuracy for the pipeline simulator.
// In Stage 4, this will be replaced by real predictor simulation.
func branchAccuracyFromConfig(cfg *config) float64 {
	// Branch predictor accuracy lookup table:
	// type 0=static, 1=bimodal, 2=gshare, 3=TAGE, 4=perceptron
	baseAccuracy := []float64{0.80, 0.88, 0.93, 0.97, 0.975}
	bpIdx := int(math.Min(cfg.BPType, 4))
	acc := baseAccuracy[bpIdx]

	// BTB size bonus: larger BTB slightly improves accuracy
	if cfg.BTBEntries > 0 {
		acc += 0.005 * math.Log2(cfg.BTBEntries/256+1)
	}
	// BHT size bonus
	if cfg.BHTSize > 0 {
		acc += 0.003 * math.Log2(cfg.BHTSize/1024+1)
	}

	return math.Min(0.999, acc)
}

// === LEGACY IPC MODEL (kept as reference / fallback) ===

func computeIPCAnalytical(dag *graph.DAG, cfg *config, m *Metrics) {
	if cfg.TaskID == 0 {
		m.IPC = math.Min(1.0, cfg.IntALUs*0.7)
		clockGHz := cfg.ClockMHz / 1000.0 * m.ThrottledFactor
		m.EffectiveClockGHz = clockGHz
		m.ThroughputGIPS = m.IPC * clockGHz
		return
	}

	baseIPC := math.Min(cfg.PIssueWidth, cfg.IntALUs+cfg.MulDivs+cfg.FPUnits+cfg.SIMDUnits)
	branchStall := computeBranchStall(cfg)
	cacheStall := computeCacheStall(cfg)
	structStall := computeStructuralStall(cfg)

	oooBonus := 1.0
	if cfg.ROBSize > 0 && cfg.RSSize > 0 {
		robEff := math.Min(1.0, cfg.ROBSize/(cfg.PPipeDepth*cfg.PIssueWidth*1.5))
		rsEff := math.Min(1.0, cfg.RSSize/(cfg.PIssueWidth*4))
		oooBonus = 0.7 + 0.3*(robEff*rsEff)
	}

	pIPC := baseIPC * (1 - branchStall) * (1 - cacheStall) * (1 - structStall) * oooBonus
	pIPC = math.Max(0.1, math.Min(pIPC, cfg.PIssueWidth))

	eIPC := math.Min(cfg.EIssueWidth, cfg.IntALUs*0.5) * (1 - branchStall*0.7) * (1 - cacheStall*0.5)
	eIPC = math.Max(0.1, eIPC)

	m.IPC = pIPC
	clockP := cfg.PClockGHz * m.ThrottledFactor
	clockE := cfg.EClockGHz * m.ThrottledFactor
	m.EffectiveClockGHz = clockP
	m.ThroughputGIPS = pIPC*clockP*cfg.PCoreCount + eIPC*clockE*cfg.ECoreCount
}

func computeBranchStall(cfg *config) float64 {
	// Branch misprediction rate based on predictor type
	baseRates := []float64{0.20, 0.12, 0.07, 0.03, 0.025} // static, bimodal, gshare, TAGE, perceptron
	bpIdx := int(math.Min(cfg.BPType, 4))
	misRate := baseRates[bpIdx]

	// BTB/BHT reduce misprediction logarithmically
	if cfg.BTBEntries > 0 {
		misRate *= math.Max(0.3, 1.0-0.03*math.Log2(cfg.BTBEntries/256+1))
	}
	if cfg.BHTSize > 0 {
		misRate *= math.Max(0.3, 1.0-0.02*math.Log2(cfg.BHTSize/1024+1))
	}

	// Branch penalty = rate × pipeline_depth × cost_per_flush
	branchRate := 0.15 // ~15% of instructions are branches
	flushCost := cfg.PPipeDepth * 0.5
	return math.Min(0.5, branchRate*misRate*flushCost/cfg.PIssueWidth)
}

func computeCacheStall(cfg *config) float64 {
	if cfg.L1ISizeKB == 0 && cfg.L1DSizeKB == 0 && cfg.SRAMSizeKB > 0 {
		// IoT with SRAM, simplified model
		return 0.05
	}

	// Working set assumption: ~64KB typical
	workingSetKB := 64.0

	// L1 hit rate
	l1HitRate := cacheHitRate(cfg.L1DSizeKB, cfg.L1DAssoc, workingSetKB)

	// L2 hit rate (for remaining misses)
	l2HitRate := 0.0
	if cfg.L2SizeKB > 0 {
		l2HitRate = cacheHitRate(cfg.L2SizeKB/1.0, cfg.L2Assoc, workingSetKB)
	}

	// L3 hit rate
	l3HitRate := 0.0
	if cfg.L3SizeMB > 0 {
		l3HitRate = cacheHitRate(cfg.L3SizeMB*1024, cfg.L3Assoc, workingSetKB)
	}

	// Multi-level miss cascade
	l1MissRate := 1.0 - l1HitRate
	l2MissRate := l1MissRate * (1.0 - l2HitRate)
	l3MissRate := l2MissRate * (1.0 - l3HitRate)

	// Stall cycles per instruction (memory intensity ~35%)
	memIntensity := 0.35
	l2Latency := 10.0
	l3Latency := 30.0
	dramLatency := 200.0

	totalStallCycles := memIntensity * (l1MissRate*l2Latency + l2MissRate*l3Latency + l3MissRate*dramLatency)

	// Prefetcher reduces stalls
	prefetchReduction := []float64{1.0, 0.8, 0.7, 0.6} // none, stride, stream, multi
	pfIdx := int(math.Min(cfg.PrefetchType, 3))
	totalStallCycles *= prefetchReduction[pfIdx]

	// Normalize: stall as fraction of total cycles
	return math.Min(0.8, totalStallCycles/(totalStallCycles+1.0/memIntensity))
}

func cacheHitRate(sizeKB, assoc, workingSetKB float64) float64 {
	if sizeKB <= 0 {
		return 0
	}
	// Empirical model: hit rate depends on ratio of cache to working set
	ratio := sizeKB / math.Max(workingSetKB, 1)
	// Diminishing returns model with associativity bonus
	assocBonus := 1.0 - 0.15/math.Max(assoc, 1)
	hitRate := (1.0 - math.Pow(ratio+0.1, -0.4)) * assocBonus
	return math.Max(0, math.Min(0.999, hitRate))
}

func computeStructuralStall(cfg *config) float64 {
	// Check if execution units can keep up with issue width
	totalExecUnits := cfg.IntALUs + cfg.MulDivs + cfg.FPUnits + cfg.SIMDUnits + cfg.LoadUnits + cfg.StoreUnits
	if totalExecUnits <= 0 {
		return 0.5
	}
	utilization := cfg.PIssueWidth / totalExecUnits
	if utilization <= 1.0 {
		return 0
	}
	return math.Min(0.5, (utilization-1.0)*0.3)
}

// === POWER MODEL ===

func computePower(dag *graph.DAG, cfg *config, m *Metrics) {
	v := math.Max(cfg.Voltage, 0.5)
	f := math.Max(cfg.ClockGHz, 0.001)

	if cfg.TaskID == 0 {
		v = math.Max(cfg.Voltage, 0.5)
		f = cfg.ClockMHz // in MHz for IoT
		// IoT: milliwatt-scale
		aluPower := cfg.IntALUs * 0.8 * v * v * f / 16.0 // mW
		sramPower := cfg.SRAMSizeKB * 0.3 * v * f / 16.0
		gpioPower := cfg.GPIOPins * 0.05 * v
		leakage := 1.0 * v // ~1mW base leakage

		m.ComponentPower["alu"] = aluPower
		m.ComponentPower["sram"] = sramPower
		m.ComponentPower["gpio"] = gpioPower
		m.ComponentPower["leakage"] = leakage

		m.DynamicPowerMW = aluPower + sramPower + gpioPower
		m.StaticPowerMW = leakage
		m.TotalPowerMW = m.DynamicPowerMW + m.StaticPowerMW
		return
	}

	// Standard CMOS power model: P = α × C × V² × f
	v2f := v * v * f

	// Per-component power
	addPower := func(name string, activity, capacitance float64) {
		dynamic := activity * capacitance * v2f
		static := 0.05 * capacitance * v // Leakage proportional to area
		m.ComponentPower[name] = dynamic + static
		m.DynamicPowerMW += dynamic
		m.StaticPowerMW += static
	}

	if cfg.TaskID == 1 {
		// RV32IM pipeline
		addPower("frontend", 0.35, 0.8*cfg.PPipeDepth*cfg.PIssueWidth)
		addPower("alu", 0.25, 1.2*cfg.IntALUs)
		addPower("muldiv", 0.15, 2.5*cfg.MulDivs)
		addPower("load_store", 0.30, 0.8*(cfg.LoadUnits+cfg.StoreUnits))
		addPower("l1i", 0.40, 0.3*cfg.L1ISizeKB)
		addPower("l1d", 0.40, 0.3*cfg.L1DSizeKB)
		addPower("regfile", 0.50, 0.2)
		addPower("bp", 0.20, 0.1*(1+math.Log2(cfg.BTBEntries/64+1)))
		addPower("memctrl", 0.30, 0.5)
	}

	if cfg.TaskID == 2 {
		// M-Series: P-cores
		pCorePower := func(prefix string, count, pipe, issue, clk, volt float64) {
			v2fLocal := volt * volt * clk
			fe := 0.35 * 0.8 * pipe * issue * v2fLocal
			be := 0.25 * (1.2*cfg.IntALUs + 2.5*cfg.MulDivs + 3.0*cfg.FPUnits + 4.0*cfg.SIMDUnits*(cfg.SIMDWidth/128)) * v2fLocal
			ooo := 0.20 * (0.003*cfg.ROBSize + 0.004*cfg.RSSize) * v2fLocal
			lsu := 0.30 * 0.8 * (cfg.LoadUnits + cfg.StoreUnits) * v2fLocal
			total := (fe + be + ooo + lsu) * count
			static := 0.05 * total / clk // Rough static estimate
			m.ComponentPower[prefix+"_dynamic"] = total
			m.ComponentPower[prefix+"_static"] = static
			m.DynamicPowerMW += total
			m.StaticPowerMW += static
		}
		pCorePower("pcore", cfg.PCoreCount, cfg.PPipeDepth, cfg.PIssueWidth, cfg.PClockGHz, cfg.PVoltage)
		pCorePower("ecore", cfg.ECoreCount, cfg.EPipeDepth, cfg.EIssueWidth, cfg.EClockGHz, cfg.EVoltage)

		addPower("l1i", 0.40, 0.3*cfg.L1ISizeKB*(cfg.PCoreCount+cfg.ECoreCount))
		addPower("l1d", 0.40, 0.3*cfg.L1DSizeKB*(cfg.PCoreCount+cfg.ECoreCount))
		addPower("l2", 0.30, 0.2*cfg.L2SizeKB/1024.0)
		addPower("l3", 0.20, 0.15*cfg.L3SizeMB)
		addPower("bp", 0.25, 0.1*(1+math.Log2(cfg.BTBEntries/256+1)+math.Log2(cfg.BHTSize/1024+1)))
		addPower("tlb", 0.35, 0.05*math.Log2(cfg.TLBEntries/64+1))
		addPower("memctrl", 0.40, 0.3*cfg.MemChannels)
		addPower("noc", 0.35, 0.2*(1+cfg.NocBW/100))
		addPower("prefetch", 0.20, 0.1*cfg.PrefetchType)
	}

	m.TotalPowerMW = m.DynamicPowerMW + m.StaticPowerMW
}

// === AREA MODEL (7nm process) ===

func computeArea(dag *graph.DAG, cfg *config, m *Metrics) {
	addArea := func(name string, area float64) {
		m.ComponentArea[name] = area
		m.TotalAreaMM2 += area
	}

	switch cfg.TaskID {
	case 0: // IoT — sub-mm² scale
		addArea("core", 0.05*cfg.IntALUs)
		addArea("sram", cfg.SRAMSizeKB*0.01)
		addArea("gpio", cfg.GPIOPins*0.002)
		addArea("pmu", 0.01)
		addArea("pad_ring", 0.1)

	case 1: // RV32IM — few mm²
		addArea("frontend", 0.08*cfg.PPipeDepth*cfg.PIssueWidth)
		addArea("alu", 0.12*cfg.IntALUs)
		addArea("muldiv", 0.25*cfg.MulDivs)
		addArea("load_store", 0.15*(cfg.LoadUnits+cfg.StoreUnits))
		addArea("l1i", cfg.L1ISizeKB*0.004*(1+0.1*cfg.L1IAssoc))
		addArea("l1d", cfg.L1DSizeKB*0.004*(1+0.1*cfg.L1DAssoc))
		addArea("regfile", 0.05)
		addArea("bp", 0.02*(1+math.Log2(cfg.BTBEntries/64+1)))
		addArea("hazard_fwd", 0.03)
		addArea("memctrl", 0.1)

	case 2: // M-Series — 50-200mm² total
		// P-cores
		pcoreArea := cfg.PCoreCount * (
			cfg.PPipeDepth*cfg.PIssueWidth*0.08 + // Frontend
				cfg.IntALUs*0.12 + cfg.MulDivs*0.25 + // Integer
				cfg.FPUnits*0.35 + cfg.SIMDUnits*0.5*(cfg.SIMDWidth/128) + // FP/SIMD
				cfg.LoadUnits*0.15 + cfg.StoreUnits*0.12 + // LSU
				cfg.ROBSize*0.001 + cfg.RSSize*0.001) // OoO structures
		addArea("pcores", pcoreArea)

		// E-cores (60% area efficiency of P-cores)
		ecoreArea := cfg.ECoreCount * (
			cfg.EPipeDepth*cfg.EIssueWidth*0.04 + 0.5)
		addArea("ecores", ecoreArea)

		// Caches
		addArea("l1i", cfg.L1ISizeKB*0.004*(cfg.PCoreCount+cfg.ECoreCount))
		addArea("l1d", cfg.L1DSizeKB*0.004*(cfg.PCoreCount+cfg.ECoreCount))
		addArea("l2", cfg.L2SizeKB*0.002)
		addArea("l3", cfg.L3SizeMB*1.5)

		// Misc
		addArea("bp", 0.05*(1+math.Log2(cfg.BTBEntries/256+1)))
		addArea("tlb", 0.02*math.Log2(cfg.TLBEntries/64+1))
		addArea("memctrl", 0.3*cfg.MemChannels)
		nocAreaTable := []float64{0.5, 1.0, 2.0, 4.0} // bus, ring, mesh, crossbar
		addArea("noc", nocAreaTable[int(math.Min(cfg.NocType, 3))]*(1+cfg.NocBW/200))
		addArea("pmu", 0.5)
	}
}

// === THERMAL MODEL ===

func computeThermal(cfg *config, m *Metrics) {
	// Safe power density limits by component type (W/mm² at 7nm)
	safeLimit := 1.2 // Default
	if cfg.TaskID == 0 {
		safeLimit = 0.5 // IoT with passive cooling
	}

	m.HotspotCount = 0
	maxPD := 0.0

	for name, power := range m.ComponentPower {
		area, ok := m.ComponentArea[name]
		if !ok || area <= 0.001 {
			area = 0.01 // Prevent div-by-zero
		}
		pd := (power / 1000.0) / area // Convert mW to W, then W/mm²

		if pd > maxPD {
			maxPD = pd
		}
		if pd > safeLimit {
			m.HotspotCount++
		}
	}

	m.MaxPowerDensity = maxPD

	// Estimate junction temperature: T_j = T_ambient + θ_JA × P_total
	// θ_JA (thermal resistance) depends on package and cooling
	thetaJA := 20.0 // °C/W for moderate cooling
	if cfg.TaskID == 0 {
		thetaJA = 80.0 // No active cooling for IoT
	}
	m.ThermalCelsius = 35.0 + thetaJA*(m.TotalPowerMW/1000.0)

	// Thermal throttling: if junction temp > 90°C or power density > safe × 1.5
	if m.ThermalCelsius > 95 {
		// Aggressive throttling
		m.ThrottledFactor = math.Max(0.3, 95.0/m.ThermalCelsius)
	} else if maxPD > safeLimit*1.5 {
		// Localized throttling due to hotspot
		m.ThrottledFactor = math.Max(0.5, safeLimit*1.5/maxPD)
	}
}
