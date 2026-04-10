package cache

import (
	"testing"
)

func TestBasicHitMiss(t *testing.T) {
	l := NewLevel("test", 4, 2, 64, 1, PolicyLRU) // 4KB, 2-way, 64B lines = 32 sets

	// First access: cold miss
	hit, _ := l.Access(0x1000, false)
	if hit {
		t.Error("First access should be a cold miss")
	}

	// Second access to same address: hit
	hit, _ = l.Access(0x1000, false)
	if !hit {
		t.Error("Second access to same address should hit")
	}

	// Access to same cache line (different byte offset): hit
	hit, _ = l.Access(0x1020, false) // Same 64B line as 0x1000
	if !hit {
		t.Error("Access within same cache line should hit")
	}

	if l.Stats.Hits != 2 || l.Stats.Misses != 1 {
		t.Errorf("Expected 2 hits, 1 miss, got %d hits, %d misses", l.Stats.Hits, l.Stats.Misses)
	}
}

func TestLRUEviction(t *testing.T) {
	// 1KB, 2-way, 64B lines = 8 sets
	l := NewLevel("test", 1, 2, 64, 1, PolicyLRU)

	// Fill both ways of set 0
	addr1 := uint64(0x0000)                                  // set 0, way 0
	addr2 := uint64(0x0000 + uint64(l.NumSets)*64)           // set 0, way 1 (different tag, same set)
	addr3 := uint64(0x0000 + uint64(l.NumSets)*64*2)         // set 0, evicts LRU

	l.Access(addr1, false) // Miss, fills way 0
	l.Access(addr2, false) // Miss, fills way 1
	l.Access(addr1, false) // Hit, makes addr1 MRU

	// Now addr2 is LRU. Access addr3 should evict addr2.
	l.Access(addr3, false) // Miss, evicts addr2

	// addr1 should still be present (was MRU)
	hit, _ := l.Access(addr1, false)
	if !hit {
		t.Error("addr1 should still be in cache (was MRU)")
	}

	// addr2 should be evicted
	hit, _ = l.Access(addr2, false)
	if hit {
		t.Error("addr2 should have been evicted (was LRU)")
	}
}

func TestDirtyWriteback(t *testing.T) {
	l := NewLevel("test", 1, 1, 64, 1, PolicyLRU) // Direct-mapped, 1KB

	// Write to addr1 — marks it dirty
	l.Access(0x1000, true) // miss, install dirty

	// Write to conflicting address (same set) — evicts dirty line
	conflicting := uint64(0x1000 + uint64(l.NumSets)*64)
	_, evictedDirty := l.Access(conflicting, false)

	if !evictedDirty {
		t.Error("Evicting a dirty line should report evictedDirty=true")
	}
	if l.Stats.Writebacks != 1 {
		t.Errorf("Expected 1 writeback, got %d", l.Stats.Writebacks)
	}
}

func TestHierarchyL1L2(t *testing.T) {
	h := NewHierarchy(HierarchyConfig{
		L1DSizeKB:   4,
		L1DAssoc:    2,
		L2SizeKB:    64,
		L2Assoc:     4,
		LineSize:    64,
		DRAMLatency: 200,
	})

	// Cold miss: goes all the way to DRAM
	lat := h.AccessData(0x1000, false)
	if lat <= L2Latency {
		t.Errorf("Cold miss should have high latency, got %d", lat)
	}

	// Second access: L1 hit
	lat = h.AccessData(0x1000, false)
	if lat != L1Latency {
		t.Errorf("Second access should be L1 hit (lat=%d), got %d", L1Latency, lat)
	}

	t.Logf("L1D: hits=%d misses=%d (%.1f%%)",
		h.L1D.Stats.Hits, h.L1D.Stats.Misses, h.L1D.Stats.HitRate()*100)
}

func TestHierarchyFullStack(t *testing.T) {
	h := NewHierarchy(HierarchyConfig{
		L1ISizeKB:   32,
		L1IAssoc:    4,
		L1DSizeKB:   32,
		L1DAssoc:    4,
		L2SizeKB:    256,
		L2Assoc:     8,
		L3SizeMB:    4,
		L3Assoc:     16,
		LineSize:    64,
		DRAMLatency: 200,
	})

	// Sequential access pattern: should have great locality
	for i := 0; i < 10000; i++ {
		addr := uint64(0x10000 + i*4) // 4-byte stride, sequential
		h.AccessData(addr, false)
	}

	stats := h.GetStats()
	l1HitRate := stats.L1D.HitRate()

	t.Logf("Sequential 10K accesses:")
	t.Logf("  L1D: hits=%d misses=%d rate=%.2f%%",
		stats.L1D.Hits, stats.L1D.Misses, l1HitRate*100)
	if stats.L2.Hits+stats.L2.Misses > 0 {
		t.Logf("  L2:  hits=%d misses=%d rate=%.2f%%",
			stats.L2.Hits, stats.L2.Misses, stats.L2.HitRate()*100)
	}
	t.Logf("  Avg latency: %.2f cycles", stats.AverageLatency)

	// Sequential access should have >90% L1 hit rate
	if l1HitRate < 0.90 {
		t.Errorf("Sequential access should have >90%% L1 hit rate, got %.2f%%", l1HitRate*100)
	}
}

func TestLargerCacheBetterHitRate(t *testing.T) {
	// Property: doubling cache size should improve hit rate
	// Working set = 32KB (512 lines × 64B) — fits in 64KB but not in 16KB.
	workingSetLines := 512
	addrs := make([]uint64, workingSetLines)
	for i := range addrs {
		addrs[i] = uint64(i * 64)
	}

	runWith := func(sizeKB int) float64 {
		l := NewLevel("test", sizeKB, 4, 64, 1, PolicyLRU)
		// Multiple passes to warm up and measure steady-state
		for pass := 0; pass < 5; pass++ {
			for _, a := range addrs {
				l.Access(a, false)
			}
		}
		return l.Stats.HitRate()
	}

	rate16 := runWith(16)   // 16KB < 32KB working set — high miss rate
	rate64 := runWith(64)   // 64KB > 32KB working set — fits
	rate256 := runWith(256) // 256KB >> working set — fits easily

	t.Logf("16KB hit rate:  %.2f%%", rate16*100)
	t.Logf("64KB hit rate:  %.2f%%", rate64*100)
	t.Logf("256KB hit rate: %.2f%%", rate256*100)

	if rate64 <= rate16 {
		t.Error("64KB cache should have better hit rate than 16KB")
	}
	if rate256 < rate64 {
		t.Error("256KB cache should have >= hit rate than 64KB")
	}
}

func TestHigherAssocReducesConflicts(t *testing.T) {
	// Property: higher associativity reduces conflict misses.
	// Access 4 different addresses mapping to the same set repeatedly.
	// 2-way can hold 2, thrashes on 4. 4-way holds all 4. 8-way holds all.
	numAddrsPerSet := 4
	numPasses := 1000

	runWith := func(assoc int) float64 {
		l := NewLevel("test", 16, assoc, 64, 1, PolicyLRU)
		stride := uint64(l.NumSets * 64) // All map to same set
		for pass := 0; pass < numPasses; pass++ {
			for way := 0; way < numAddrsPerSet; way++ {
				l.Access(uint64(way)*stride, false)
			}
		}
		return l.Stats.HitRate()
	}

	rate2way := runWith(2)
	rate4way := runWith(4)
	rate8way := runWith(8)

	t.Logf("2-way hit rate: %.2f%%", rate2way*100)
	t.Logf("4-way hit rate: %.2f%%", rate4way*100)
	t.Logf("8-way hit rate: %.2f%%", rate8way*100)

	if rate4way <= rate2way {
		t.Error("4-way should have better rate than 2-way for 4-address conflict pattern")
	}
	if rate8way < rate4way {
		t.Error("8-way should have >= rate than 4-way")
	}
}

// Benchmark: measure cache access throughput
func BenchmarkCacheAccess(b *testing.B) {
	l := NewLevel("L1D", 32, 4, 64, 1, PolicyLRU)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		addr := uint64(i * 8) // Sequential 8-byte stride
		l.Access(addr, false)
	}
}

func BenchmarkHierarchyAccess(b *testing.B) {
	h := NewHierarchy(HierarchyConfig{
		L1DSizeKB: 32, L1DAssoc: 4,
		L2SizeKB: 256, L2Assoc: 8,
		L3SizeMB: 1, L3Assoc: 8,
		LineSize: 64, DRAMLatency: 200,
	})
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		addr := uint64(i * 8)
		h.AccessData(addr, false)
	}
}
