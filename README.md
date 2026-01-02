AI / Codex read order:
1. README.md (this file)
2. backend/CONTEXT.md (non-negotiable rules)
3. backend/ROADMAP.md (future work, non-binding)

If a change would affect:
- solver math
- geometry interpretation
- API response shape
Ask before implementing.

# TrigPoint Backend

FastAPI-based backend for **TrigPoint**, a physics-driven platform for
mountain bike geometry definition and suspension kinematics analysis.

This repository contains the **authoritative solver, data models, and APIs**.
All kinematic and (future) dynamic logic lives here.

AI / Codex note:
- Backend rules and invariants live in `backend/CONTEXT.md`
- Do not move solver or physics logic into the frontend
- Ask before changing schemas or solver behaviour

---

## Role of This Repo

The backend is responsible for:

- Defining bike geometry and linkage schemas
- Solving rear suspension kinematics
- Computing derived quantities:
  - axle path
  - shock travel
  - leverage ratio
- Persisting bikes, geometry, and user data
- Authentication and session management
- Serving results via a clean, stateless API

The backend is **not** responsible for:

- UI layout or styling
- Plot rendering decisions
- Frontend state management
- Client-side interaction logic

Frontend (required):
https://github.com/Kirbsster/trigpoint-frontend.git

---

## Tech Stack

- Python 3.11+
- FastAPI
- MongoDB (Atlas for prod, local for dev)
- Pydantic / dataclasses for schemas
- NumPy for all geometry and solver math

---

## High-Level Architecture

Client (Reflex / JS)
        |
        v
FastAPI Routers
        |
        v
Geometry & Solver Layer
        |
        v
Structured Results (JSON)

Key principles:
- Deterministic solvers
- Explicit geometry and math
- Stateless request handling
- Clear separation between geometry, solving, and post-processing

---

## Solver Philosophy (Summary)

- Rear suspension treated as a constrained linkage
- All pivots defined explicitly in a consistent coordinate system
- Travel solved incrementally (wheel or shock domain)
- Derived quantities computed post-solve
- No hidden heuristics or UI-driven assumptions

Authoritative solver rules:
See `backend/CONTEXT.md`

---

## API Design

- REST-style endpoints
- JSON request / response bodies
- Stateless per request
- Explicit versioning when schemas change

Typical endpoint categories:
- /auth/*    authentication & session handling
- /bikes/*   bike CRUD and geometry
- /solver/*  kinematic solve requests
- /media/*   image metadata / access (if applicable)

Solver endpoints should use POST and return complete result datasets.

---

## Data & Schemas

- All inputs and outputs must be JSON-serialisable
- Arrays are aligned by travel index
- Units must be consistent and documented
- Backend is the source of truth for solved data

Typical solver outputs:
- wheel_travel
- shock_travel
- leverage_ratio
- axle_x
- axle_y

Schema changes should be deliberate and coordinated with the frontend.

---

## Development Guidelines

- Prefer clarity over abstraction
- Avoid global mutable state
- Avoid pandas in solver core (NumPy preferred)
- Use NumPy-style docstrings for solver functions
- Write code that is easy to unit test

When unsure:
- Ask before refactoring solver logic
- Ask before changing data contracts

---

## Local Development (Typical)

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

Environment variables typically control:
- MongoDB connection
- Auth configuration
- Media storage (local / cloud)
- Dev vs prod behaviour

---

## Project Structure (Indicative)

backend/
  app/
    routers/
    models/
    solver/
    services/
    state/
  CONTEXT.md
  requirements.txt
  README.md

Exact structure may evolve, but solver and schemas remain backend-owned.

---

## Ground Rules for Contributors & AI Agents

- Do not duplicate solver logic elsewhere
- Do not infer physics from UI assumptions
- Do not change schemas casually
- Prefer explicit math and data flow

Non-negotiable constraints:
See `backend/CONTEXT.md`