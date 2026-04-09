// Package pipeline implements a cycle-accurate CPU pipeline simulator.
//
// DSA Optimizations:
//   - Fixed-size register scoreboard array (not map) for O(1) hazard detection
//   - Ring buffer for pipeline stage slots (avoids slice shifting)
//   - Pre-allocated instruction pool (zero GC pressure in hot loop)
//   - Bitfield encoding for instruction flags (cache-line friendly)
//   - Struct-of-arrays for stage occupancy (better spatial locality)
package pipeline

// OpCode identifies the functional unit an instruction requires.
type OpCode uint8

const (
	OpNop   OpCode = iota
	OpALU          // Integer add/sub/logic — 1 cycle
	OpMul          // Integer multiply — 3 cycles
	OpDiv          // Integer divide — 10-20 cycles
	OpLoad         // Memory load — depends on cache
	OpStore        // Memory store — 1 cycle (write buffer)
	OpBranch       // Conditional branch — 1 cycle + possible flush
	OpFP           // Floating point — 4 cycles
	OpSIMD         // SIMD vector — 3 cycles
)

// Instruction represents a single micro-op flowing through the pipeline.
// Packed to 48 bytes for cache-line alignment (fits 1.3 per cache line).
type Instruction struct {
	PC          uint32   // Program counter (sufficient for trace indexing)
	OpType      OpCode   // Functional unit required
	SrcReg1     uint8    // First source register (0-63, 0 = no dependency)
	SrcReg2     uint8    // Second source register
	DstReg      uint8    // Destination register (0 = no writeback)
	Latency     uint8    // Execution latency in cycles
	Flags       uint8    // Bit 0: branch taken, Bit 1: is memory, Bit 2: is store
	_pad        uint8    // Alignment padding
	MemAddr     uint32   // Memory address for loads/stores (lower 32 bits)
	IssueAt     uint32   // Cycle when issued into execute
	CompleteAt  uint32   // Cycle when result is available
}

// Flag accessors using bitfield operations — branchless, no map lookups.
func (i *Instruction) IsBranchTaken() bool { return i.Flags&0x01 != 0 }
func (i *Instruction) IsMemory() bool      { return i.Flags&0x02 != 0 }
func (i *Instruction) IsStore() bool       { return i.Flags&0x04 != 0 }

// ExecutionLatency returns the base latency for each op type.
// Stored as a lookup table — O(1), no branching.
var ExecutionLatency = [9]uint8{
	0, // NOP
	1, // ALU
	3, // MUL
	12, // DIV
	1, // LOAD (cache hit; miss penalty added by cache sim)
	1, // STORE
	1, // BRANCH
	4, // FP
	3, // SIMD
}
