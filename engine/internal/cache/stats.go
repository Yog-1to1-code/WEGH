// Package cache implements a cycle-accurate cache hierarchy simulator.
//
// DSA Optimizations:
//   - Direct-mapped set indexing via bitmask (O(1) set lookup, no modulo)
//   - LRU tracking via move-to-front in fixed-size arrays (no linked list overhead)
//   - Tag comparison via uint64 (single compare instruction)
//   - Pre-allocated set arrays (zero GC pressure)
//   - Power-of-2 enforcement on set count for bitmask addressing
package cache

// ReplacementPolicy determines how eviction victims are chosen.
type ReplacementPolicy uint8

const (
	PolicyLRU       ReplacementPolicy = iota // Least Recently Used
	PolicyPseudoLRU                          // Tree-based pseudo-LRU (faster, slightly less accurate)
	PolicyRandom                             // Random replacement
)

// CacheStats tracks hit/miss statistics for a single cache level.
type CacheStats struct {
	Hits       uint64
	Misses     uint64
	Evictions  uint64
	Writebacks uint64 // Dirty evictions that must propagate
}

// HitRate returns the cache hit rate as a float64.
func (s *CacheStats) HitRate() float64 {
	total := s.Hits + s.Misses
	if total == 0 {
		return 0
	}
	return float64(s.Hits) / float64(total)
}

// HierarchyStats holds stats for each level plus overall latency.
type HierarchyStats struct {
	L1I     CacheStats
	L1D     CacheStats
	L2      CacheStats
	L3      CacheStats
	TLB     CacheStats
	TotalMemoryLatencyCycles uint64 // Sum of all access latencies
	TotalAccesses            uint64
	AverageLatency           float64
}
