package pipeline

// StallCounters provides a granular breakdown of pipeline stalls.
// Used for engineering feedback to the RL agent.
type StallCounters struct {
	RAW          uint64 // Read-After-Write data hazards
	WAW          uint64 // Write-After-Write (OoO only)
	Structural   uint64 // Not enough functional units
	Control      uint64 // Branch misprediction flushes
	Memory       uint64 // Cache miss stalls (placeholder for Stage 3)
	TotalBubbles uint64 // Total wasted pipeline slots
}

// Scoreboard tracks when each architectural register becomes available.
// DSA: Fixed-size array — O(1) read/write, zero allocations, cache-friendly.
// Index 0 is unused (register 0 = "no dependency").
const MaxRegs = 65 // Support up to 64 architectural registers (1-indexed)

type Scoreboard [MaxRegs]uint32 // scoreboard[reg] = cycle when value is ready

// Available returns true if register r is ready by cycle c.
// Branchless: the comparison itself produces the boolean.
func (s *Scoreboard) Available(r uint8, c uint32) bool {
	return s[r] <= c
}

// Reserve marks register r as unavailable until cycle c.
func (s *Scoreboard) Reserve(r uint8, c uint32) {
	if r > 0 && c > s[r] {
		s[r] = c
	}
}

// StallUntil returns the cycle at which both source registers are ready.
// O(1) — two array lookups and a max.
func (s *Scoreboard) StallUntil(src1, src2 uint8) uint32 {
	a := s[src1]
	b := s[src2]
	if a > b {
		return a
	}
	return b
}

// Flush resets all registers to available (cycle 0) after a branch mispredict.
// Inline loop — compiler will auto-vectorize on amd64.
func (s *Scoreboard) Flush() {
	for i := range s {
		s[i] = 0
	}
}

// FunctionalUnits tracks available execution slots per functional unit type.
// DSA: Array indexed by OpCode for O(1) availability check.
type FunctionalUnits struct {
	Capacity  [9]int    // Total units per OpCode
	Available [9]int    // Free units this cycle
}

// NewFunctionalUnits creates unit pools from the CPU config.
func NewFunctionalUnits(intALU, mul, div, load, store, fp, simd int) FunctionalUnits {
	fu := FunctionalUnits{}
	fu.Capacity[OpALU] = intALU
	fu.Capacity[OpMul] = mul
	fu.Capacity[OpDiv] = div
	fu.Capacity[OpLoad] = load
	fu.Capacity[OpStore] = store
	fu.Capacity[OpBranch] = intALU // Branches use ALU
	fu.Capacity[OpFP] = fp
	fu.Capacity[OpSIMD] = simd
	fu.Capacity[OpNop] = 1000 // NOP always available
	return fu
}

// ResetCycle resets all units to full capacity at the start of each cycle.
func (fu *FunctionalUnits) ResetCycle() {
	fu.Available = fu.Capacity
}

// TryAllocate attempts to claim a unit for the given opcode.
// Returns true if a unit was available, false if structural stall.
func (fu *FunctionalUnits) TryAllocate(op OpCode) bool {
	if fu.Available[op] > 0 {
		fu.Available[op]--
		return true
	}
	return false
}
