package cache

// CacheLine represents one line in a cache set.
// Packed to 16 bytes: tag(8) + meta(8).
type CacheLine struct {
	Tag   uint64 // Upper address bits
	// Bit 0: Valid, Bit 1: Dirty
	// Bits 2-33: LRU counter (30 bits, sufficient for assoc ≤ 32)
	Meta  uint64
}

// Valid returns whether this line contains valid data.
func (cl *CacheLine) Valid() bool { return cl.Meta&1 != 0 }

// Dirty returns whether this line has been modified.
func (cl *CacheLine) Dirty() bool { return cl.Meta&2 != 0 }

// LRUAge returns the LRU age counter (lower = more recently used).
func (cl *CacheLine) LRUAge() uint32 { return uint32(cl.Meta >> 2) }

// setValid marks the line as valid.
func (cl *CacheLine) setValid()  { cl.Meta |= 1 }

// setDirty marks the line as dirty.
func (cl *CacheLine) setDirty()  { cl.Meta |= 2 }

// clearDirty clears the dirty bit.
func (cl *CacheLine) clearDirty() { cl.Meta &^= 2 }

// setLRUAge sets the LRU counter.
func (cl *CacheLine) setLRUAge(age uint32) {
	cl.Meta = (cl.Meta & 3) | (uint64(age) << 2)
}

// invalidate clears the line completely.
func (cl *CacheLine) invalidate() { cl.Meta = 0; cl.Tag = 0 }

// Level models a single level of cache (L1, L2, or L3).
//
// DSA Design:
//   - Sets are stored as a flat 2D array: sets[setIndex][way]
//   - Set index is computed via bitmask (requires power-of-2 set count)
//   - Tag comparison is a single uint64 ==
//   - LRU is maintained by incrementing all ages and resetting the accessed way
//     This is O(assoc) per access but assoc is tiny (2-16) so it fits in a
//     single cache line and is vectorizable.
type Level struct {
	Name     string
	SizeKB   int
	Assoc    int    // Set associativity (ways per set)
	LineSize int    // Bytes per line (typically 64)
	NumSets  int    // Computed: SizeKB*1024 / (Assoc * LineSize)
	Latency  int    // Access latency in cycles (hit)

	// DSA: Flat array of sets, each set is a slice of CacheLine.
	// sets[i] has exactly `Assoc` entries.
	sets [][]CacheLine

	// Bitmask for O(1) set index extraction (requires power-of-2 NumSets)
	setMask  uint64
	tagShift uint // Number of bits to shift right to get the tag

	// Statistics
	Stats CacheStats

	// Policy
	Policy ReplacementPolicy
	// Random state for random replacement (splitmix64)
	randState uint64
}

// NewLevel creates a cache level with the given parameters.
// Enforces power-of-2 on the set count for bitmask-based indexing.
func NewLevel(name string, sizeKB, assoc, lineSize, latency int, policy ReplacementPolicy) *Level {
	if lineSize <= 0 {
		lineSize = 64
	}
	if assoc <= 0 {
		assoc = 1
	}
	if sizeKB <= 0 {
		sizeKB = 1
	}

	totalBytes := sizeKB * 1024
	numSets := totalBytes / (assoc * lineSize)

	// Round down to nearest power of 2 for bitmask addressing
	numSets = roundDownPow2(numSets)
	if numSets < 1 {
		numSets = 1
	}

	// Compute bit positions
	offsetBits := log2(lineSize)
	setBits := log2(numSets)

	l := &Level{
		Name:      name,
		SizeKB:    sizeKB,
		Assoc:     assoc,
		LineSize:  lineSize,
		NumSets:   numSets,
		Latency:   latency,
		setMask:   uint64(numSets - 1),
		tagShift:  uint(offsetBits + setBits),
		Policy:    policy,
		randState: 0xdeadbeefcafe1234, // Fixed seed for determinism
	}

	// Pre-allocate all sets
	l.sets = make([][]CacheLine, numSets)
	for i := range l.sets {
		l.sets[i] = make([]CacheLine, assoc)
	}

	return l
}

// Access performs a cache lookup. Returns (hit, evictedDirty).
//
// Algorithm:
//   1. Extract set index via bitmask (O(1))
//   2. Linear scan of ways for tag match (O(assoc), assoc ≤ 16)
//   3. On hit: update LRU, return
//   4. On miss: find victim via LRU/random, install new line
//
// Hot path: ~5-20ns per access on modern hardware.
func (l *Level) Access(addr uint64, isWrite bool) (hit bool, evictedDirty bool) {
	// O(1) set index extraction via bitmask
	offsetBits := log2(l.LineSize)
	setIndex := (addr >> uint(offsetBits)) & l.setMask
	tag := addr >> l.tagShift

	set := l.sets[setIndex]

	// Linear scan for tag match — O(assoc)
	// assoc is tiny (2-16), so this is effectively constant time
	// and fits entirely in one or two L1 cache lines.
	for i := 0; i < l.Assoc; i++ {
		if set[i].Valid() && set[i].Tag == tag {
			// HIT
			l.Stats.Hits++
			if isWrite {
				set[i].setDirty()
			}
			l.updateLRU(set, i)
			return true, false
		}
	}

	// MISS
	l.Stats.Misses++

	// Find victim
	victimIdx := l.findVictim(set)
	evictedDirty = set[victimIdx].Valid() && set[victimIdx].Dirty()
	if evictedDirty {
		l.Stats.Writebacks++
	}
	if set[victimIdx].Valid() {
		l.Stats.Evictions++
	}

	// Install new line
	set[victimIdx].Tag = tag
	set[victimIdx].Meta = 0
	set[victimIdx].setValid()
	if isWrite {
		set[victimIdx].setDirty()
	}
	l.updateLRU(set, victimIdx)

	return false, evictedDirty
}

// updateLRU promotes the accessed way to MRU position.
// All other ways' ages are incremented.
// O(assoc) — but assoc ≤ 16, so this is ~16 increment ops.
func (l *Level) updateLRU(set []CacheLine, accessedWay int) {
	for i := 0; i < l.Assoc; i++ {
		if i == accessedWay {
			set[i].setLRUAge(0) // Most recently used
		} else if set[i].Valid() {
			set[i].setLRUAge(set[i].LRUAge() + 1)
		}
	}
}

// findVictim selects eviction candidate.
// Prefers invalid lines first, then applies policy.
func (l *Level) findVictim(set []CacheLine) int {
	// First: prefer invalid (empty) lines — O(assoc)
	for i := 0; i < l.Assoc; i++ {
		if !set[i].Valid() {
			return i
		}
	}

	switch l.Policy {
	case PolicyLRU:
		// Find way with highest LRU age — O(assoc)
		maxAge := uint32(0)
		victim := 0
		for i := 0; i < l.Assoc; i++ {
			age := set[i].LRUAge()
			if age >= maxAge {
				maxAge = age
				victim = i
			}
		}
		return victim

	case PolicyRandom:
		// Splitmix64 for fast deterministic random
		l.randState += 0x9e3779b97f4a7c15
		z := l.randState
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb
		z = z ^ (z >> 31)
		return int(z % uint64(l.Assoc))

	default: // PseudoLRU: use LRU as fallback (same logic, no recursion)
		maxAge := uint32(0)
		victim := 0
		for i := 0; i < l.Assoc; i++ {
			age := set[i].LRUAge()
			if age >= maxAge {
				maxAge = age
				victim = i
			}
		}
		return victim
	}
}

// Flush invalidates all lines (used on context switch simulation).
func (l *Level) Flush() {
	for i := range l.sets {
		for j := range l.sets[i] {
			l.sets[i][j].invalidate()
		}
	}
	l.Stats = CacheStats{}
}

// --- Utility functions ---

// roundDownPow2 returns the largest power of 2 ≤ n.
func roundDownPow2(n int) int {
	if n <= 0 {
		return 1
	}
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	return (n + 1) >> 1
}

// log2 returns floor(log2(n)), assumes n is a power of 2.
func log2(n int) int {
	if n <= 1 {
		return 0
	}
	r := 0
	for n > 1 {
		n >>= 1
		r++
	}
	return r
}
