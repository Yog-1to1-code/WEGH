package cache

// Hierarchy orchestrates multi-level cache lookups (L1 → L2 → L3 → DRAM).
//
// Lookup algorithm:
//   1. Check L1D (or L1I for instruction fetches)
//   2. On miss → check L2 (if present)
//   3. On miss → check L3 (if present)
//   4. On miss → DRAM penalty
//
// DSA: Each level is checked in sequence. The total latency is the sum of
// hit latencies up to the level where the data is found, plus a miss penalty
// at each level that misses. This is O(levels) = O(3-4), constant time.
type Hierarchy struct {
	L1I *Level // Instruction cache (optional)
	L1D *Level // Data cache
	L2  *Level // Unified L2 (optional)
	L3  *Level // Unified L3 (optional)

	// DRAM parameters
	DRAMLatency int // Cycles for DRAM access (100-300 typical)

	// Prefetcher (placeholder for future expansion)
	PrefetchEnabled bool
	PrefetchStride  int // Sequential stride in bytes

	// Running stats
	totalLatency uint64
	totalAccesses uint64
}

// HierarchyConfig holds initialization parameters.
type HierarchyConfig struct {
	// L1I
	L1ISizeKB int
	L1IAssoc  int

	// L1D
	L1DSizeKB int
	L1DAssoc  int

	// L2
	L2SizeKB int
	L2Assoc  int

	// L3
	L3SizeMB int
	L3Assoc  int

	// Line size (shared)
	LineSize int

	// DRAM
	DRAMLatency int

	// Prefetcher type: 0=none, 1=stride, 2=stream
	PrefetchType int
}

// Latencies for each cache level (cycles).
const (
	L1Latency = 1   // L1 hit: 1 cycle (pipelined)
	L2Latency = 10  // L2 hit: 10 cycles
	L3Latency = 35  // L3 hit: 35 cycles
	DefaultDRAMLatency = 200 // DRAM: 200 cycles
)

// NewHierarchy creates a multi-level cache hierarchy from config.
func NewHierarchy(cfg HierarchyConfig) *Hierarchy {
	h := &Hierarchy{
		DRAMLatency: cfg.DRAMLatency,
	}
	if h.DRAMLatency <= 0 {
		h.DRAMLatency = DefaultDRAMLatency
	}

	lineSize := cfg.LineSize
	if lineSize <= 0 {
		lineSize = 64
	}

	// L1I (instruction cache)
	if cfg.L1ISizeKB > 0 {
		h.L1I = NewLevel("L1I", cfg.L1ISizeKB, maxInt(cfg.L1IAssoc, 2), lineSize, L1Latency, PolicyLRU)
	}

	// L1D (data cache) — always present
	l1dSize := cfg.L1DSizeKB
	if l1dSize <= 0 {
		l1dSize = 16 // Default 16KB
	}
	l1dAssoc := cfg.L1DAssoc
	if l1dAssoc <= 0 {
		l1dAssoc = 4
	}
	h.L1D = NewLevel("L1D", l1dSize, l1dAssoc, lineSize, L1Latency, PolicyLRU)

	// L2 (unified)
	if cfg.L2SizeKB > 0 {
		h.L2 = NewLevel("L2", cfg.L2SizeKB, maxInt(cfg.L2Assoc, 4), lineSize, L2Latency, PolicyLRU)
	}

	// L3 (unified)
	if cfg.L3SizeMB > 0 {
		h.L3 = NewLevel("L3", cfg.L3SizeMB*1024, maxInt(cfg.L3Assoc, 8), lineSize, L3Latency, PolicyPseudoLRU)
	}

	// Prefetcher
	if cfg.PrefetchType > 0 {
		h.PrefetchEnabled = true
		h.PrefetchStride = lineSize // Stride = one cache line ahead
	}

	return h
}

// AccessData performs a data memory access through the cache hierarchy.
// Returns the total latency in cycles.
//
// Algorithm: Sequential level probing with early exit on hit.
// Miss at each level adds that level's latency + triggers lookup at next level.
// Total latency = sum of latencies at all probed levels.
//
// Time complexity: O(levels × assoc) = O(4 × 16) = O(64) = constant.
// Space complexity: O(1) — no allocations.
func (h *Hierarchy) AccessData(addr uint64, isWrite bool) int {
	h.totalAccesses++
	latency := 0

	// L1D lookup
	hit, _ := h.L1D.Access(addr, isWrite)
	latency += h.L1D.Latency
	if hit {
		h.totalLatency += uint64(latency)
		return latency
	}

	// L2 lookup (if present)
	if h.L2 != nil {
		hit, _ = h.L2.Access(addr, isWrite)
		latency += h.L2.Latency
		if hit {
			// Fill L1D on L2 hit (inclusive)
			h.L1D.Access(addr, false) // Install in L1
			h.totalLatency += uint64(latency)
			return latency
		}
	}

	// L3 lookup (if present)
	if h.L3 != nil {
		hit, _ = h.L3.Access(addr, isWrite)
		latency += h.L3.Latency
		if hit {
			// Fill L1D and L2 on L3 hit
			if h.L2 != nil {
				h.L2.Access(addr, false)
			}
			h.L1D.Access(addr, false)
			h.totalLatency += uint64(latency)
			return latency
		}
	}

	// DRAM access
	latency += h.DRAMLatency

	// Fill all levels on DRAM return
	if h.L3 != nil {
		h.L3.Access(addr, false)
	}
	if h.L2 != nil {
		h.L2.Access(addr, false)
	}
	h.L1D.Access(addr, false)

	// Simple prefetch: also bring in next line
	if h.PrefetchEnabled {
		nextAddr := addr + uint64(h.PrefetchStride)
		if h.L3 != nil {
			h.L3.Access(nextAddr, false)
		}
		if h.L2 != nil {
			h.L2.Access(nextAddr, false)
		}
		h.L1D.Access(nextAddr, false)
	}

	h.totalLatency += uint64(latency)
	return latency
}

// AccessInstruction performs an instruction fetch through L1I → L2 → L3 → DRAM.
func (h *Hierarchy) AccessInstruction(addr uint64) int {
	if h.L1I == nil {
		return 1 // No I-cache modeled
	}

	h.totalAccesses++
	latency := 0

	hit, _ := h.L1I.Access(addr, false)
	latency += h.L1I.Latency
	if hit {
		h.totalLatency += uint64(latency)
		return latency
	}

	// I-cache miss falls through to L2/L3 (unified)
	if h.L2 != nil {
		hit, _ = h.L2.Access(addr, false)
		latency += h.L2.Latency
		if hit {
			h.L1I.Access(addr, false)
			h.totalLatency += uint64(latency)
			return latency
		}
	}

	if h.L3 != nil {
		hit, _ = h.L3.Access(addr, false)
		latency += h.L3.Latency
		if hit {
			if h.L2 != nil {
				h.L2.Access(addr, false)
			}
			h.L1I.Access(addr, false)
			h.totalLatency += uint64(latency)
			return latency
		}
	}

	latency += h.DRAMLatency
	if h.L3 != nil {
		h.L3.Access(addr, false)
	}
	if h.L2 != nil {
		h.L2.Access(addr, false)
	}
	h.L1I.Access(addr, false)

	h.totalLatency += uint64(latency)
	return latency
}

// GetStats returns aggregate statistics for the hierarchy.
func (h *Hierarchy) GetStats() HierarchyStats {
	s := HierarchyStats{
		L1D:                      h.L1D.Stats,
		TotalMemoryLatencyCycles: h.totalLatency,
		TotalAccesses:            h.totalAccesses,
	}
	if h.totalAccesses > 0 {
		s.AverageLatency = float64(h.totalLatency) / float64(h.totalAccesses)
	}
	if h.L1I != nil {
		s.L1I = h.L1I.Stats
	}
	if h.L2 != nil {
		s.L2 = h.L2.Stats
	}
	if h.L3 != nil {
		s.L3 = h.L3.Stats
	}
	return s
}

// Flush resets all cache levels.
func (h *Hierarchy) Flush() {
	h.L1D.Flush()
	if h.L1I != nil {
		h.L1I.Flush()
	}
	if h.L2 != nil {
		h.L2.Flush()
	}
	if h.L3 != nil {
		h.L3.Flush()
	}
	h.totalLatency = 0
	h.totalAccesses = 0
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
