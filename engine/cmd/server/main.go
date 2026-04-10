// WEGH — Go Simulation Engine & OpenEnv Server
// Self-contained HTTP server that serves all OpenEnv-standard endpoints.
// No Python framework dependency required for the core API.
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"

	"github.com/wegh/engine/api"
)

const landingHTML = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WEGH — CPU Architecture Design Environment</title>
  <style>
    :root { --bg: #0f0f0f; --fg: #e8e8e8; --accent: #00d4aa; --card: #1a1a2e; --border: #333; }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: var(--bg); color: var(--fg); padding: 2rem; max-width: 900px; margin: 0 auto; }
    h1 { color: var(--accent); font-size: 2rem; margin-bottom: 0.5rem; }
    h2 { color: var(--accent); font-size: 1.2rem; margin: 1.5rem 0 0.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }
    .subtitle { color: #888; margin-bottom: 2rem; }
    .badge { display: inline-block; background: var(--card); border: 1px solid var(--accent); color: var(--accent);
             padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; margin-right: 0.5rem; }
    table { width: 100%%; border-collapse: collapse; margin: 0.5rem 0; }
    th, td { padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }
    th { color: var(--accent); font-weight: 600; }
    code { background: var(--card); padding: 0.15rem 0.4rem; border-radius: 3px; font-size: 0.85em; }
    .task-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
                 padding: 1rem; margin: 0.5rem 0; }
    .task-card h3 { color: var(--accent); margin-bottom: 0.3rem; }
    .task-card .difficulty { font-size: 0.8rem; color: #aaa; }
  </style>
</head>
<body>
  <h1>🔬 WEGH</h1>
  <p class="subtitle">Workload Evaluation for Generative Hardware — AI-Driven CPU Architecture Design</p>
  <span class="badge">OpenEnv 0.2.3</span>
  <span class="badge">Go 1.22</span>
  <span class="badge">PyTorch Hackathon</span>

  <h2>API Endpoints</h2>
  <table>
    <tr><th>Method</th><th>Path</th><th>Description</th></tr>
    <tr><td><code>GET</code></td><td>/health</td><td>Health check</td></tr>
    <tr><td><code>GET</code></td><td>/ping</td><td>Liveness probe</td></tr>
    <tr><td><code>POST</code></td><td>/reset</td><td>Start new episode</td></tr>
    <tr><td><code>POST</code></td><td>/step</td><td>Execute architectural action</td></tr>
    <tr><td><code>GET</code></td><td>/state</td><td>Current environment state</td></tr>
    <tr><td><code>GET</code></td><td>/grade</td><td>Episode grading with sub-scores</td></tr>
    <tr><td><code>GET</code></td><td>/tasks</td><td>Available tasks</td></tr>
    <tr><td><code>GET</code></td><td>/metrics</td><td>Prometheus metrics</td></tr>
    <tr><td><code>GET</code></td><td>/replay</td><td>Episode replay data</td></tr>
  </table>

  <h2>Tasks</h2>
  <div class="task-card">
    <h3>Task 0: 8-Bit IoT Microcontroller</h3>
    <p class="difficulty">Difficulty: Easy • 20 steps</p>
    <p>Design a minimal 8-bit MCU for IoT sensors. Power budget: 50mW.</p>
  </div>
  <div class="task-card">
    <h3>Task 1: RV32IM 5-Stage Pipeline</h3>
    <p class="difficulty">Difficulty: Medium • 30 steps</p>
    <p>Design a RISC-V RV32IM core. Maximize IPC within 10mm² area budget.</p>
  </div>
  <div class="task-card">
    <h3>Task 2: M-Series Heterogeneous Superscalar</h3>
    <p class="difficulty">Difficulty: Hard • 40 steps</p>
    <p>Design an Apple M-series inspired CPU. Power density limit: 1.5 W/mm².</p>
  </div>

  <h2>Quick Start</h2>
  <p>Reset an episode: <code>curl -X POST http://localhost:7860/reset -d '{"task":"rv32im"}'</code></p>
  <p>Take a step: <code>curl -X POST http://localhost:7860/step -d '{"action":{"type":"resize","component":"l1d","param_name":"size_kb","value":32}}'</code></p>
  <p>Grade: <code>curl http://localhost:7860/grade</code></p>
</body>
</html>`

func main() {
	bind := flag.String("bind", "", "Address to bind the HTTP server (default: 0.0.0.0:7860)")
	maxMem := flag.Int("max-memory", 4096, "Max memory in MB (for logging)")
	flag.Parse()

	// Resolve bind address: flag > PORT env > default
	addr := *bind
	if addr == "" {
		port := os.Getenv("PORT")
		if port == "" {
			port = "7860"
		}
		addr = "0.0.0.0:" + port
	}

	log.SetFlags(log.Ltime | log.Lmicroseconds)
	log.Printf("WEGH Engine starting on %s (max_mem=%dMB, cpus=%d, go=%s)",
		addr, *maxMem, runtime.NumCPU(), runtime.Version())

	handler := api.NewHandler()
	mux := http.NewServeMux()
	handler.RegisterRoutes(mux)

	// Landing page
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, landingHTML)
	})

	// CORS middleware wrapper
	corsHandler := corsMiddleware(mux)

	server := &http.Server{
		Addr:    addr,
		Handler: corsHandler,
	}

	// Graceful shutdown
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down engine...")
		server.Close()
	}()

	log.Printf("WEGH OpenEnv server ready at http://%s", addr)
	log.Printf("Dashboard: http://%s/", addr)
	if err := server.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}
	log.Println("Engine stopped.")
}

// corsMiddleware adds CORS headers for evaluator and dashboard access.
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		// Request logging
		if !strings.HasPrefix(r.URL.Path, "/metrics") {
			log.Printf("[%s] %s %s", r.Method, r.URL.Path, r.RemoteAddr)
		}

		next.ServeHTTP(w, r)
	})
}
