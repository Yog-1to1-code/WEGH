# WEGH: Workload Evaluation for Generative Hardware (High-Level Overview)

Welcome to the WEGH project! This document explains the core concepts and mechanics of our Hackathon project, introducing the technical jargon step-by-step so you understand exactly what we built and why it's impressive.

## What does WEGH stand for?
**WEGH** stands for **Workload Evaluation for Generative Hardware**. 
* **Workload:** The specific software task the computer needs to run (like an IoT sensor vs a heavy video game).
* **Evaluation:** Measuring how well the computer runs that task.
* **Generative Hardware:** Using Artificial Intelligence (AI) to automatically *generate* and design the physical layout of the computer chip.

## The Big Idea
Designing modern computer processors (CPUs) is incredibly difficult. Engineers must balance three main metrics, known in the industry as **PPA**:
1. **Performance (Speed):** Measured in **IPC** (Instructions Per Clock) and **Throughput** (GIPS - Giga Instructions Per Second).
2. **Power (Energy):** How many milliwatts (mW) the chip consumes. More power means worse battery life.
3. **Area (Cost/Size):** How much physical silicon space (mm²) the chip takes up. Larger chips are more expensive to manufacture.

There is also a hidden fourth metric: **Thermals**. If you pack too many fast components together, it creates a "hotspot." If it gets too hot, the chip undergoes **Thermal Throttling** (slowing itself down to avoid melting).

We built WEGH as a **Reinforcement Learning Environment**. It's basically a highly accurate software playground where an AI (like an advanced LLM) can try its hand at designing a CPU layout to learn how to optimize PPA.

## How it Works: The Core Loop
The environment operates in a continuous loop:
1. **State / Observation:** The AI looks at the current blueprint of the CPU (which is represented as a network graph or "DAG"). It sees the current power usage, speed, and temperature.
2. **Action:** The AI sends a structured `JSON` command to modify the blueprint. It might say, *"Let's increase the size of the L1 Data Cache from 32KB to 64KB,"* or *"Let's add a Branch Predictor component to guess 'if-statements' faster."*
3. **Simulation (The Engine):** Our blazing-fast backend engine recalculates the physics. Does that larger cache draw too much wattage? Did the Branch Predictor speed up the IPC?
4. **Reward:** We grade the AI's move strictly between `0` and `1`. A good design gets a high score. A design that overheats or exceeds the area budget gets a low score.

## The Three Difficulty Levels (Tasks)
To prove the AI is actually smart, our framework forces it to design three completely different kinds of CPUs:

1. **The 8-Bit IoT Microcontroller (`iot_8bit`)**
   * **The Goal:** Design a tiny, cheap chip for a smart thermostat or sensor.
   * **The Challenge:** Strict Power (<50mW) and Area limits. Speed doesn't matter much. The AI must strip away unnecessary, power-hungry components.

2. **The RISC-V 5-Stage Pipelined Core (`rv32im`)**
   * **The Goal:** Build a classic mid-range chip.
   * **The Challenge:** Balance! It needs good IPC (speed) but has to stay under a 10mm² area budget. The AI has to make smart tradeoffs about cache sizes and data processing paths.

3. **The M-Series Heterogeneous Superscalar (`mseries_superscalar`)**
   * **The Goal:** Build an ultra-modern, Apple M1-style beast of a chip.
   * **The Challenge:** "Heterogeneous" means mixing massive Performance Cores (P-cores) with efficient E-cores. The AI must maximize speed without triggering the dreaded **Thermal Throttling**. Poor layout = Hotspots = Failure.

## Why Getting "Validation" Was So Hard
In Phase 2 of the hackathon, the judges used a remote automated system (the Scaler Evaluator) to essentially attack our server and test if our simulation physics and network code were bulletproof. 

We had to design an architecture that cleanly separated our **Networking Layer** (which uses WebSockets on a FastAPI server to talk to the AI) from our **Simulation Engine** (the math heavy-lifter). Our system passed perfectly—meaning our WEGH framework is officially considered a stable, highly-accurate sandbox for testing AI chip designers!
