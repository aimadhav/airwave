# AGENTS.md — Autonomous Agent Engineering & Development Protocols

This document establishes the mandatory engineering rules, architectural principles, and quality standards for all AI agents and developers working on this codebase.

Every agent operating within this workspace must strictly abide by these directives to ensure isolated, modular, and maintainable software development.

---

## 1. Core Architectural Directives

### 1.1 Strict Separation of Concerns (SoC) & Decoupling
- **Zero Cross-Module Entanglement**: Every assigned layer, component, or service must be fully self-contained. 
- **No Direct Concrete Imports**: Never import concrete implementations, runtime classes, or internal helpers from another module or layer into your domain logic.
- **Contract-Based Interfaces**: All cross-boundary interaction must occur exclusively through abstract interfaces (protocols, abstract base classes), standardized Data Transfer Objects (DTOs), or canonical data contracts (typed dataclasses / Pydantic schemas).
- **Zero Circular Dependencies**: Structure packages hierarchically with dependencies pointing inwards toward pure domain abstractions. Never create mutual or circular imports between components.

### 1.2 Interface-First & Contract-Driven Design
- **Define Boundaries Before Logic**: Prior to implementing core algorithms or services, define and freeze the data models (DTOs) and interface signatures.
- **Explicit Input/Output Types**: Every function and method must have explicit type annotations for all arguments and return values.
- **Canonical Schemas as Single Source of Truth**: Data passed across component boundaries must adhere to strict, validated schemas. No component may unilaterally modify shared contracts without updating the canonical specification.

### 1.3 Mock-First & Isolated Development
- **No External Runtime Blockers**: A module must never require a live external dependency (e.g., remote services, databases, cloud APIs, specialized hardware) to be executed or verified during development.
- **Mock Adapters**: Always provide mock/stub implementations for any external dependency or downstream consumer (e.g., in-memory data stores, dummy client providers, deterministic fixture generators).
- **Self-Contained Verification**: Each component must include an independent test suite that executes successfully in a sandbox environment without external infrastructure.

---

## 2. Software Engineering Fundamentals

### 2.1 SOLID Principles & Clean Architecture
- **Single Responsibility Principle (SRP)**: Each file, class, and module must have one and only one reason to change. Separate orchestration, business logic, data persistence, and external communication into distinct files.
- **Open/Closed Principle (OCP)**: Design services to be open for extension but closed for modification via dependency injection, strategy patterns, and provider interfaces.
- **Liskov Substitution Principle (LSP)**: Concrete providers must fully honor the behavioral contract of their abstract interface.
- **Interface Segregation Principle (ISP)**: Create narrow, focused interfaces rather than monolithic, bloated base classes.
- **Dependency Inversion Principle (DIP)**: High-level business logic must depend on abstractions, never on low-level storage or I/O drivers.

### 2.2 Defensive Programming & Error Handling
- **Domain-Specific Exception Hierarchy**: Define a clear, module-scoped exception tree inheriting from a common base domain error. Never raise generic unhandled exceptions across boundaries.
- **Fail Fast & Explicitly**: Validate configuration parameters, inputs, and preconditions at boundary entry points.
- **Graceful Degradation**: Catch provider or I/O failures and translate them into typed domain errors or fallback responses with descriptive error messages.

### 2.3 Configuration & Secrets Hygiene
- **Externalize Runtime Configurations**: Never hardcode runtime parameters, timeouts, thresholds, model identifiers, or URLs directly in source code.
- **Config Loaders**: Manage parameters via external configuration files (e.g., YAML / JSON) and typed configuration loaders.
- **Secrets Management**: Read API keys, credentials, and sensitive tokens strictly from environment variables or `.env` files. Never commit plaintext secrets to the repository.

---

## 3. Standard Code Organization Rules

When developing any module or component, adhere to the following directory layout convention:

```text
src/<module_name>/
├── __init__.py
├── config.py                 # Typed configuration dataclass and loader
├── models/                   # Centralized domain models and data transfer objects (DTOs)
│   ├── __init__.py
│   └── ...
├── interfaces/               # Abstract interfaces / protocols for providers and repositories
│   ├── __init__.py
│   └── ...
├── services/                 # Pure domain business logic and algorithms (Single Responsibility)
│   ├── __init__.py
│   └── ...
├── adapters/                 # Concrete implementations (local, external, third-party)
│   ├── __init__.py
│   ├── mock_adapter.py       # Deterministic mock adapter for testing
│   └── ...
└── exceptions.py             # Module-scoped exception hierarchy
```

---

## 4. Agent Execution Checklist

Before declaring any implementation task complete, every agent must verify:

1. [ ] **Isolation Check**: Does the module operate without importing private implementation details from other modules?
2. [ ] **Contract Compliance**: Do all models and interfaces match the documented shared data schemas?
3. [ ] **Mock Provider Included**: Is there a mock/stub adapter provided to allow other agents to test against this module immediately?
4. [ ] **Configuration Externalized**: Are all hyperparameters and configurations loaded via typed config objects?
5. [ ] **Type Annotations & Linting**: Are all public functions, classes, and methods fully type-annotated with clean docstrings?
6. [ ] **Unit Tests Passing**: Does the module include comprehensive unit tests verifying success paths, edge cases, and failure modes using mock fixtures?
