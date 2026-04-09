package pipeline

import (
	"testing"
)

func TestSingleCycleIoT(t *testing.T) {
	// A single-cycle IoT processor should achieve IPC ≈ 1.0 on simple code
	cfg := PipelineConfig{
		Depth:      2,
		IssueWidth: 1,
		IntALUs:    1,
		MulDivs:    0,
		LoadUnits:  1,
		StoreUnits: 1,
		BranchAccuracy: 0.80,
		TaskID:     0,
	}
	engine := NewEngine(cfg)
	traceCfg := DefaultTraceConfig(0)
	trace := GenerateTrace(traceCfg, 1000, 42)
	result := engine.Simulate(trace, 50000)

	if result.IPC <= 0 || result.IPC > 1.0 {
		t.Errorf("IoT single-cycle IPC should be (0, 1.0], got %.4f", result.IPC)
	}
	if result.Committed == 0 {
		t.Error("No instructions committed")
	}
	t.Logf("IoT: IPC=%.4f committed=%d cycles=%d stalls=%+v",
		result.IPC, result.Committed, result.Cycles, result.Stalls)
}

func TestRV32IM5Stage(t *testing.T) {
	cfg := PipelineConfig{
		Depth:          5,
		IssueWidth:     1,
		IntALUs:        1,
		MulDivs:        1,
		LoadUnits:      1,
		StoreUnits:     1,
		BranchAccuracy: 0.88,
		TaskID:         1,
	}
	engine := NewEngine(cfg)
	trace := GenerateTrace(DefaultTraceConfig(1), 5000, 42)
	result := engine.Simulate(trace, 100000)

	if result.IPC <= 0 {
		t.Errorf("RV32IM IPC should be > 0, got %.4f", result.IPC)
	}
	if result.Stalls.RAW == 0 {
		t.Error("Expected some RAW hazards in a 5-stage pipeline")
	}
	t.Logf("RV32IM: IPC=%.4f committed=%d cycles=%d RAW=%d structural=%d control=%d",
		result.IPC, result.Committed, result.Cycles,
		result.Stalls.RAW, result.Stalls.Structural, result.Stalls.Control)
}

func TestSuperscalarOoO(t *testing.T) {
	cfg := PipelineConfig{
		Depth:          14,
		IssueWidth:     6,
		IntALUs:        4,
		MulDivs:        2,
		FPUnits:        2,
		SIMDUnits:      2,
		LoadUnits:      2,
		StoreUnits:     1,
		ROBSize:        128,
		RSSize:         64,
		BranchAccuracy: 0.95,
		TaskID:         2,
	}
	engine := NewEngine(cfg)
	trace := GenerateTrace(DefaultTraceConfig(2), 10000, 42)
	result := engine.Simulate(trace, 200000)

	// Superscalar should achieve higher IPC than single-issue
	if result.IPC <= 0.5 {
		t.Errorf("Superscalar IPC should be > 0.5, got %.4f", result.IPC)
	}
	t.Logf("M-Series: IPC=%.4f committed=%d cycles=%d mispredicts=%d/%d",
		result.IPC, result.Committed, result.Cycles,
		result.BranchMispredicts, result.TotalBranches)
}

func TestDeeperPipelineWorseBranchPenalty(t *testing.T) {
	// Prove: deeper pipeline with same branch accuracy = lower IPC
	// (because flush penalty = pipeline depth)
	trace := GenerateTrace(DefaultTraceConfig(1), 3000, 42)

	shallow := NewEngine(PipelineConfig{
		Depth: 5, IssueWidth: 1, IntALUs: 1, MulDivs: 1,
		LoadUnits: 1, StoreUnits: 1, BranchAccuracy: 0.85, TaskID: 1,
	})
	deep := NewEngine(PipelineConfig{
		Depth: 15, IssueWidth: 1, IntALUs: 1, MulDivs: 1,
		LoadUnits: 1, StoreUnits: 1, BranchAccuracy: 0.85, TaskID: 1,
	})

	rShallow := shallow.Simulate(trace, 100000)
	rDeep := deep.Simulate(trace, 100000)

	t.Logf("Shallow (5-stage): IPC=%.4f, control_stalls=%d", rShallow.IPC, rShallow.Stalls.Control)
	t.Logf("Deep   (15-stage): IPC=%.4f, control_stalls=%d", rDeep.IPC, rDeep.Stalls.Control)

	if rDeep.IPC >= rShallow.IPC {
		t.Error("Deeper pipeline should have LOWER IPC due to higher branch flush penalty")
	}
	if rDeep.Stalls.Control <= rShallow.Stalls.Control {
		t.Error("Deeper pipeline should have MORE control stalls")
	}
}

func TestWidthVsExecUnits(t *testing.T) {
	// Prove: issue width > available exec units = structural stalls
	trace := GenerateTrace(DefaultTraceConfig(1), 3000, 42)

	matched := NewEngine(PipelineConfig{
		Depth: 5, IssueWidth: 4, IntALUs: 4, MulDivs: 2,
		FPUnits: 2, LoadUnits: 2, StoreUnits: 1,
		BranchAccuracy: 0.90, TaskID: 1,
	})
	bottleneck := NewEngine(PipelineConfig{
		Depth: 5, IssueWidth: 4, IntALUs: 1, MulDivs: 1,
		FPUnits: 0, LoadUnits: 1, StoreUnits: 1,
		BranchAccuracy: 0.90, TaskID: 1,
	})

	rMatched := matched.Simulate(trace, 100000)
	rBottom := bottleneck.Simulate(trace, 100000)

	t.Logf("Matched (4 ALU): IPC=%.4f, structural=%d", rMatched.IPC, rMatched.Stalls.Structural)
	t.Logf("Bottleneck (1 ALU): IPC=%.4f, structural=%d", rBottom.IPC, rBottom.Stalls.Structural)

	if rBottom.Stalls.Structural <= rMatched.Stalls.Structural {
		t.Error("Bottlenecked config should have MORE structural stalls")
	}
}

// BenchmarkPipelineSimulation measures simulation throughput.
// Target: >50 MIPS (million instructions per second).
func BenchmarkPipelineSimulation(b *testing.B) {
	cfg := PipelineConfig{
		Depth: 14, IssueWidth: 6,
		IntALUs: 4, MulDivs: 2, FPUnits: 2, SIMDUnits: 2,
		LoadUnits: 2, StoreUnits: 1,
		ROBSize: 128, RSSize: 64,
		BranchAccuracy: 0.95, TaskID: 2,
	}
	engine := NewEngine(cfg)
	trace := GenerateTrace(DefaultTraceConfig(2), 10000, 42)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		engine.Simulate(trace, 200000)
	}

	// Report MIPS
	result := engine.Simulate(trace, 200000)
	b.ReportMetric(float64(result.Committed)/1e6, "Minst")
}

// BenchmarkTraceGeneration measures trace generation speed.
func BenchmarkTraceGeneration(b *testing.B) {
	cfg := DefaultTraceConfig(2)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		GenerateTrace(cfg, 10000, int64(i))
	}
}
