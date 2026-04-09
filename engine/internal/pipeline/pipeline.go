package pipeline

// PipelineConfig defines the architectural parameters for the simulated pipeline.
type PipelineConfig struct {
	// Pipeline geometry
	Depth      int // Number of pipeline stages (2-20)
	IssueWidth int // Instructions issued per cycle (1-8)

	// Functional units
	IntALUs   int
	MulDivs   int
	FPUnits   int
	SIMDUnits int
	LoadUnits int
	StoreUnits int

	// OoO parameters (0 = in-order)
	ROBSize int // Reorder buffer entries
	RSSize  int // Reservation station entries

	// Branch prediction accuracy (0.0 - 1.0)
	// In Stage 4 this will be replaced by real predictor simulation.
	BranchAccuracy float64

	// Clock and task context
	ClockGHz float64
	TaskID   int
}

// SimResult holds the complete output of a pipeline simulation run.
type SimResult struct {
	IPC              float64      // Instructions committed / cycles elapsed
	Cycles           uint64       // Total cycles simulated
	Committed        uint64       // Instructions successfully retired
	Stalls           StallCounters // Detailed stall breakdown
	FlushedInsts     uint64       // Instructions flushed due to mispredicts
	BranchMispredicts uint64      // Number of mispredictions
	TotalBranches    uint64       // Total branch instructions
}

// Engine is the cycle-accurate pipeline simulator.
//
// Architecture: Models a simplified pipeline with these logical stages:
//   Fetch → Decode → Issue → Execute (variable latency) → Writeback → Commit
//
// DSA Design:
//   - Scoreboard array for O(1) register dependency checking
//   - Functional unit array for O(1) structural hazard checking
//   - Ring buffer style processing (advance all stages each tick)
//   - No heap allocations in the simulation hot loop
type Engine struct {
	cfg PipelineConfig

	// Hazard tracking
	scoreboard Scoreboard
	fu         FunctionalUnits

	// Pipeline state
	cycle     uint32
	fetchIdx  int    // Index into trace
	committed uint64

	// Stall tracking
	stalls StallCounters

	// Branch prediction state (simplified for Stage 2)
	// Stage 4 will replace this with real predictors.
	branchMispredicts uint64
	totalBranches     uint64
	flushed           uint64

	// In-flight instruction window (models ROB for OoO)
	// DSA: Fixed-capacity circular buffer avoids allocations.
	inflight    [256]inflightEntry // Max 256 in-flight instructions
	inflightLen int
	inflightHead int
}

// inflightEntry tracks an instruction currently in the pipeline.
type inflightEntry struct {
	inst       Instruction
	completeCycle uint32
	committed  bool
}

// NewEngine creates a pipeline engine from architectural config.
func NewEngine(cfg PipelineConfig) *Engine {
	// Clamp to sane bounds
	if cfg.Depth < 2 {
		cfg.Depth = 2
	}
	if cfg.Depth > 20 {
		cfg.Depth = 20
	}
	if cfg.IssueWidth < 1 {
		cfg.IssueWidth = 1
	}
	if cfg.IssueWidth > 8 {
		cfg.IssueWidth = 8
	}
	if cfg.IntALUs < 1 {
		cfg.IntALUs = 1
	}
	if cfg.BranchAccuracy <= 0 {
		cfg.BranchAccuracy = 0.80
	}
	if cfg.BranchAccuracy > 1.0 {
		cfg.BranchAccuracy = 1.0
	}

	e := &Engine{cfg: cfg}
	// Functional units: if a type has 0 dedicated units, instructions
	// of that type are executed on the ALU (with extra latency).
	e.fu = NewFunctionalUnits(
		cfg.IntALUs,
		max(cfg.MulDivs, 1),   // At least 1 mul/div unit (shared with ALU)
		max(cfg.MulDivs, 1),   // Div shares with mul
		max(cfg.LoadUnits, 1), // At least 1 load port
		max(cfg.StoreUnits, 1),// At least 1 store port
		max(cfg.FPUnits, 1),   // Fall back to 1 slow FP unit
		max(cfg.SIMDUnits, 1), // Fall back to 1 slow SIMD unit
	)
	return e
}

// Simulate runs the trace through the pipeline and returns real IPC.
//
// Algorithm: Each cycle performs these steps in reverse pipeline order
// (writeback → execute → issue → fetch) to correctly model forwarding.
//
// Time complexity: O(maxCycles × issueWidth) — linear in simulation length.
// Space complexity: O(1) — all buffers are pre-allocated fixed-size.
func (e *Engine) Simulate(trace []Instruction, maxCycles int) SimResult {
	if len(trace) == 0 {
		return SimResult{IPC: 0}
	}

	traceLen := len(trace)
	e.cycle = 0
	e.fetchIdx = 0
	e.committed = 0
	e.stalls = StallCounters{}
	e.branchMispredicts = 0
	e.totalBranches = 0
	e.flushed = 0
	e.inflightLen = 0
	e.inflightHead = 0
	e.scoreboard.Flush()

	windowSize := e.cfg.ROBSize
	if windowSize <= 0 {
		// In-order pipeline: window must be large enough to hold
		// instructions spanning the full pipeline depth + execution latency.
		// Otherwise long-latency ops (MUL=3, DIV=12) block the whole pipeline.
		windowSize = e.cfg.Depth * e.cfg.IssueWidth * 2
		if windowSize < 32 {
			windowSize = 32
		}
	}
	if windowSize > 256 {
		windowSize = 256
	}

	for e.cycle < uint32(maxCycles) && e.fetchIdx < traceLen {
		e.cycle++

		// === Phase 1: Retire completed instructions (oldest first) ===
		e.retire()

		// === Phase 2: Reset functional unit availability ===
		e.fu.ResetCycle()

		// === Phase 3: Issue new instructions (up to issue width) ===
		issued := 0
		for issued < e.cfg.IssueWidth && e.fetchIdx < traceLen && e.inflightLen < windowSize {
			inst := &trace[e.fetchIdx]

			// Check data hazards: are source registers ready?
			readyCycle := e.scoreboard.StallUntil(inst.SrcReg1, inst.SrcReg2)
			if readyCycle > e.cycle {
				// RAW hazard — pipeline bubble
				stallCycles := readyCycle - e.cycle
				e.stalls.RAW += uint64(stallCycles)
				e.stalls.TotalBubbles += uint64(stallCycles) * uint64(e.cfg.IssueWidth-issued)
				e.cycle = readyCycle
				// Don't break — try issuing at the new cycle
			}

			// Check structural hazards: is a functional unit available?
			if !e.fu.TryAllocate(inst.OpType) {
				e.stalls.Structural++
				e.stalls.TotalBubbles += uint64(e.cfg.IssueWidth - issued)
				break // Can't issue more this cycle
			}

			// Issue the instruction
			completeCycle := e.cycle + uint32(inst.Latency)

			// For pipelined stages, add pipeline traversal latency
			// (fetch → decode → issue takes `depth/2` cycles approximately)
			frontendLatency := uint32(e.cfg.Depth / 3)
			if frontendLatency < 1 {
				frontendLatency = 1
			}
			completeCycle += frontendLatency

			// Reserve destination register in scoreboard
			if inst.DstReg > 0 {
				e.scoreboard.Reserve(inst.DstReg, completeCycle)
			}

			// Handle branches
			if inst.OpType == OpBranch {
				e.totalBranches++
				// Simulate prediction accuracy
				// Misprediction causes a pipeline flush penalty
				predicted := e.predictBranch(inst)
				if !predicted {
					e.branchMispredicts++
					// Flush penalty = pipeline depth cycles
					flushPenalty := uint32(e.cfg.Depth)
					e.cycle += flushPenalty
					e.stalls.Control += uint64(flushPenalty)
					e.stalls.TotalBubbles += uint64(flushPenalty) * uint64(e.cfg.IssueWidth)

					// Flush younger instructions from window
					flushedCount := e.flushYounger(e.cycle - flushPenalty)
					e.flushed += uint64(flushedCount)

					// Reset scoreboard (conservative: in reality ROB handles this)
					e.scoreboard.Flush()
					issued++
					e.fetchIdx++
					break
				}
			}

			// Add to in-flight window
			idx := (e.inflightHead + e.inflightLen) % 256
			e.inflight[idx] = inflightEntry{
				inst:          *inst,
				completeCycle: completeCycle,
				committed:     false,
			}
			e.inflightLen++

			e.fetchIdx++
			issued++
		}

		// If we couldn't issue anything and nothing retired, we're stalled
		if issued == 0 {
			e.stalls.TotalBubbles += uint64(e.cfg.IssueWidth)
		}
	}

	// Drain remaining in-flight instructions
	e.retire()

	totalCycles := uint64(e.cycle)
	if totalCycles == 0 {
		totalCycles = 1
	}

	return SimResult{
		IPC:               float64(e.committed) / float64(totalCycles),
		Cycles:            totalCycles,
		Committed:         e.committed,
		Stalls:            e.stalls,
		FlushedInsts:      e.flushed,
		BranchMispredicts: e.branchMispredicts,
		TotalBranches:     e.totalBranches,
	}
}

// retire commits completed instructions from the in-flight window.
// In-order retirement: instructions commit in program order.
func (e *Engine) retire() {
	for e.inflightLen > 0 {
		entry := &e.inflight[e.inflightHead]
		if entry.completeCycle > e.cycle {
			break // Oldest instruction not yet complete
		}
		entry.committed = true
		e.committed++
		e.inflightHead = (e.inflightHead + 1) % 256
		e.inflightLen--
	}
}

// flushYounger removes instructions issued after the given cycle.
// Returns the number of flushed instructions.
func (e *Engine) flushYounger(issuedAfter uint32) int {
	count := 0
	// Scan from tail backward
	for e.inflightLen > 0 {
		tailIdx := (e.inflightHead + e.inflightLen - 1) % 256
		entry := &e.inflight[tailIdx]
		if entry.inst.IssueAt <= issuedAfter && entry.completeCycle > 0 {
			break
		}
		e.inflightLen--
		count++
	}
	return count
}

// predictBranch uses the configured accuracy to simulate prediction.
// Deterministic: uses instruction PC as the random seed so results
// are reproducible across runs.
// In Stage 4, this will be replaced by real predictor models.
func (e *Engine) predictBranch(inst *Instruction) bool {
	// Deterministic pseudo-random based on PC and cycle
	// Uses a fast integer hash (splitmix32) — no floating point
	hash := inst.PC ^ (e.cycle * 2654435761) // Knuth multiplicative hash
	hash = (hash ^ (hash >> 16)) * 0x45d9f3b
	hash = (hash ^ (hash >> 16))

	// Map to [0, 1000) range and compare against accuracy threshold
	threshold := uint32(e.cfg.BranchAccuracy * 1000)
	return (hash % 1000) < threshold
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
