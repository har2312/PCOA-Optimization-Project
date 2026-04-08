# Pine Cone Optimization Algorithm (PCOA) - Modified

Research project for improving the Pine Cone Optimization Algorithm (PCOA) using adaptive mechanisms, Lévy flight, and chaotic maps.

## About

The **Pine Cone Optimization Algorithm (PCOA)** is a nature-inspired metaheuristic algorithm proposed by Anaraki & Farzin (2024), modeled after the reproduction strategies of pine trees:

- **Pollination Phase** — Simulates wind-driven pollen dispersal (exploration)
- **Dispersal Phase** — Simulates gravity and animal-based cone dispersal (exploitation)

This project modifies the original PCOA to improve exploration-exploitation balance and benchmarks it against other recent metaheuristic algorithms.

## Repository Structure

```
PCOA-Optimization-Project/
├── src/                    # Source code
│   ├── pcoa.py             # Original PCOA (baseline)
│   └── mpcoa.py            # Modified PCOA (our contribution)
├── results/                # Experiment output CSV files
├── scripts/                # Analysis and plotting tools
├── docs/                   # Reference papers and notes
└── README.md
```

## Benchmark Suites

- CEC 2014
- CEC 2017
- CEC 2020
- CEC 2022

All experiments use **60,000 function evaluations** with **30 independent runs**.

## Comparison Algorithms

| # | Algorithm | Abbreviation |
|---|-----------|-------------|
| 1 | Particle Swarm Optimization | PSO |
| 2 | Grey Wolf Optimizer | GWO |
| 3 | Whale Optimization Algorithm | WOA |
| 4 | Sine Cosine Algorithm | SCA |
| 5 | Harris Hawks Optimization | HHO |
| 6 | Arithmetic Optimization Algorithm | AOA |
| 7 | Aquila Optimizer | AO |

## Reference

> Anaraki, M.V. & Farzin, S. (2024). "The Pine Cone Optimization Algorithm (PCOA)." *Biomimetics*, 9(2), 91. DOI: [10.3390/biomimetics9020091](https://doi.org/10.3390/biomimetics9020091)
