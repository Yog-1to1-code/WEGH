// WEGH — Go Simulation Engine
// Background daemon that handles DAG state management, topological validation,
// and PPA (Power, Performance, Area) heuristic simulations.
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"syscall"

	"github.com/wegh/engine/api"
)

func main() {
	bind := flag.String("bind", "127.0.0.1:8080", "Address to bind the HTTP server")
	maxMem := flag.Int("max-memory", 4096, "Max memory in MB (for logging)")
	flag.Parse()

	log.SetFlags(log.Ltime | log.Lmicroseconds)
	log.Printf("WEGH Engine starting on %s (max_mem=%dMB, cpus=%d, go=%s)",
		*bind, *maxMem, runtime.NumCPU(), runtime.Version())

	// Set GOMAXPROCS to 1 — we're given vCPU 1, Python gets vCPU 0
	runtime.GOMAXPROCS(1)

	handler := api.NewHandler()
	mux := http.NewServeMux()
	handler.RegisterRoutes(mux)

	// Root health check (for any root ping tests)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"status":"ok","engine":"wegh"}`)
	})

	server := &http.Server{
		Addr:    *bind,
		Handler: mux,
	}

	// Graceful shutdown
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down engine...")
		server.Close()
	}()

	log.Printf("Engine ready at http://%s", *bind)
	if err := server.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}
	log.Println("Engine stopped.")
}
