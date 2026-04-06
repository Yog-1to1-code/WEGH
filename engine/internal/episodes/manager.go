// Package episodes manages per-episode DAG instances.
// Each episode gets its own graph to prevent cross-contamination.
package episodes

import (
	"fmt"
	"sync"

	"github.com/wegh/engine/internal/graph"
)

// Manager holds all active episode graphs.
type Manager struct {
	mu       sync.RWMutex
	episodes map[string]*graph.DAG
}

// NewManager creates an episode manager.
func NewManager() *Manager {
	return &Manager{
		episodes: make(map[string]*graph.DAG, 16),
	}
}

// Create initializes a new episode with a task-specific graph.
func (m *Manager) Create(episodeID string, taskID int) (*graph.DAG, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.episodes[episodeID]; exists {
		return nil, fmt.Errorf("episode '%s' already exists", episodeID)
	}

	dag := graph.BuildInitialGraph(taskID)
	m.episodes[episodeID] = dag
	return dag, nil
}

// Get returns the DAG for an episode.
func (m *Manager) Get(episodeID string) (*graph.DAG, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	dag, ok := m.episodes[episodeID]
	if !ok {
		return nil, fmt.Errorf("episode '%s' not found", episodeID)
	}
	return dag, nil
}

// Delete removes an episode's graph (called on reset/cleanup).
func (m *Manager) Delete(episodeID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.episodes, episodeID)
}

// Count returns the number of active episodes.
func (m *Manager) Count() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.episodes)
}
