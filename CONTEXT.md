<!-- AI_AGENT_CONTEXT: read this first -->

# Context (for Codex)
- This file is authoritative for goals + constraints in this directory.
- Do not change data schemas without updating the matching context file.
- Prefer explicit, testable code. Avoid hidden state.
- Ask before large refactors.

# TrigPoint Backend — Context

This directory contains the TrigPoint backend service.

## Technology Stack

- Python 3.11+
- FastAPI
- MongoDB (Atlas + local dev)
- Pydantic / dataclasses
- NumPy for geometry and solver math

## Responsibilities

The backend is responsible for:
- Defining bike geometry schemas
- Solving suspension kinematics
- Producing clean, structured result datasets
- Serving data via REST (and later streaming)
- Authentication and user/session management

The backend is **not** responsible for:
- UI layout
- Plot styling
- Heavy frontend state management

## Solver Philosophy

- Treat the rear suspension as a constrained linkage problem
- Use explicit pivot coordinates and rigid links
- Solve travel incrementally (wheel travel or shock travel)
- Derived quantities (leverage ratio, axle path) are post-processed

Preferred solver traits:
- Deterministic
- Side-effect free
- Stateless where possible
- Vectorised with NumPy when reasonable
- Easy to unit test

## Data Structures

- Geometry inputs are explicit and serialisable
- Results are arrays of named quantities
- Avoid pandas in solver core (NumPy preferred)
- JSON output must be frontend-friendly

Example result fields:
- wheel_travel
- shock_travel
- leverage_ratio
- axle_x, axle_y
- instant_center (future)

## API Design

- Endpoints should be:
  - Predictable
  - Versionable
  - Stateless per request
- Prefer POST for solver runs
- Use clear request/response models

## Development Style

- Clear docstrings (NumPy style preferred)
- Minimal cleverness
- Explicit math over abstraction layers
- Favour clarity over micro-optimisation

When unsure:
- Ask for clarification rather than guessing solver intent