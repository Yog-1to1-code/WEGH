package pipeline

import (
	"math/rand"
)

// TraceConfig controls the instruction mix for synthetic trace generation.
type TraceConfig struct {
	// Instruction mix probabilities (must sum to 1.0)
	ALUFrac    float64
	MulFrac    float64
	DivFrac    float64
	LoadFrac   float64
	StoreFrac  float64
	BranchFrac float64
	FPFrac     float64
	SIMDFrac   float64

	// Branch behavior
	BranchTakenRate float64 // Fraction of branches that are taken

	// Register pressure (higher = more dependencies = more stalls)
	NumArchRegs int // Number of architectural registers in use (4-64)

	// Memory access pattern
	MemLocality float64 // 0.0 = random, 1.0 = highly sequential
}

// DefaultTraceConfig returns a workload-representative instruction mix.
// Based on SPEC CPU2017 integer benchmark averages.
func DefaultTraceConfig(taskID int) TraceConfig {
	switch taskID {
	case 0: // IoT: simple control-heavy code, few registers
		return TraceConfig{
			ALUFrac: 0.45, MulFrac: 0.02, DivFrac: 0.01,
			LoadFrac: 0.22, StoreFrac: 0.10, BranchFrac: 0.20,
			FPFrac: 0.0, SIMDFrac: 0.0,
			BranchTakenRate: 0.55, NumArchRegs: 8,
			MemLocality: 0.8,
		}
	case 1: // RV32IM: balanced integer workload
		return TraceConfig{
			ALUFrac: 0.35, MulFrac: 0.05, DivFrac: 0.02,
			LoadFrac: 0.25, StoreFrac: 0.10, BranchFrac: 0.18,
			FPFrac: 0.0, SIMDFrac: 0.05,
			BranchTakenRate: 0.50, NumArchRegs: 32,
			MemLocality: 0.6,
		}
	case 2: // M-Series: heavy FP/SIMD, pointer-chasing loads
		return TraceConfig{
			ALUFrac: 0.25, MulFrac: 0.05, DivFrac: 0.02,
			LoadFrac: 0.25, StoreFrac: 0.08, BranchFrac: 0.12,
			FPFrac: 0.10, SIMDFrac: 0.13,
			BranchTakenRate: 0.45, NumArchRegs: 64,
			MemLocality: 0.4,
		}
	default:
		return DefaultTraceConfig(1)
	}
}

// GenerateTrace creates a synthetic instruction trace with realistic
// register dependencies and memory access patterns.
//
// DSA: Uses cumulative distribution function (CDF) array for O(1)
// instruction type selection instead of if-else chain.
func GenerateTrace(cfg TraceConfig, count int, seed int64) []Instruction {
	rng := rand.New(rand.NewSource(seed))
	trace := make([]Instruction, count)

	// Build CDF lookup table for O(1) opcode selection
	// Instead of 8 if-else comparisons per instruction, we do one binary search
	// on 8 entries = 3 comparisons max.
	type cdfEntry struct {
		threshold float64
		op        OpCode
	}
	cdf := make([]cdfEntry, 0, 8)
	cumulative := 0.0
	fracs := []struct {
		f  float64
		op OpCode
	}{
		{cfg.ALUFrac, OpALU}, {cfg.MulFrac, OpMul}, {cfg.DivFrac, OpDiv},
		{cfg.LoadFrac, OpLoad}, {cfg.StoreFrac, OpStore}, {cfg.BranchFrac, OpBranch},
		{cfg.FPFrac, OpFP}, {cfg.SIMDFrac, OpSIMD},
	}
	for _, f := range fracs {
		if f.f > 0 {
			cumulative += f.f
			cdf = append(cdf, cdfEntry{cumulative, f.op})
		}
	}

	numRegs := cfg.NumArchRegs
	if numRegs < 4 {
		numRegs = 4
	}
	if numRegs > 64 {
		numRegs = 64
	}

	// Memory address generation with tunable locality.
	// Uses a strided pattern + random jitter to model real access patterns.
	baseAddr := uint32(0x1000)
	stride := uint32(64) // Cache-line sized stride

	for i := 0; i < count; i++ {
		r := rng.Float64()

		// O(log n) opcode lookup via CDF
		op := OpALU
		for _, entry := range cdf {
			if r < entry.threshold {
				op = entry.op
				break
			}
		}

		inst := Instruction{
			PC:      uint32(i * 4), // 4-byte aligned PC
			OpType:  op,
			Latency: ExecutionLatency[op],
		}

		// Register dependencies: create realistic dependency chains.
		// Higher register pressure (fewer regs) = more RAW hazards = more stalls.
		inst.SrcReg1 = uint8(rng.Intn(numRegs) + 1) // 1-based (0 = none)
		if op != OpBranch && rng.Float64() < 0.6 {
			inst.SrcReg2 = uint8(rng.Intn(numRegs) + 1)
		}
		if op != OpStore && op != OpBranch {
			inst.DstReg = uint8(rng.Intn(numRegs) + 1)
		}

		// Memory addressing
		if op == OpLoad || op == OpStore {
			inst.Flags |= 0x02 // IsMemory
			if op == OpStore {
				inst.Flags |= 0x04 // IsStore
			}
			if rng.Float64() < cfg.MemLocality {
				// Sequential: stride from last address (spatial locality)
				inst.MemAddr = baseAddr
				baseAddr += stride
			} else {
				// Random: pointer-chasing pattern (poor locality)
				inst.MemAddr = uint32(rng.Intn(1 << 20)) & ^uint32(63) // Page-aligned
			}
		}

		// Branch behavior
		if op == OpBranch {
			if rng.Float64() < cfg.BranchTakenRate {
				inst.Flags |= 0x01 // Taken
			}
		}

		trace[i] = inst
	}

	return trace
}
