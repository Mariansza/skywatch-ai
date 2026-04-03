# ADR Skill — Design Spec

**Date:** 2026-04-04
**Status:** Draft

## Summary

A custom Claude Code skill (`/adr`) that guides the user through creating Architecture Decision Records via an interactive, challenging process. Designed for the SkyWatch AI project — an aerial detection & tracking system targeting defense/edge deployment.

## Goals

- Document architectural decisions as they happen, not after the fact
- Challenge the user's thinking with domain-relevant questions (edge constraints, latency, sovereignty)
- Generate well-structured ADR files in `docs/adr/` with automatic numbering
- Support the user's learning objective: understand every decision

## Non-Goals

- Automatic triggering (manual `/adr` invocation only)
- Multiple templates or template files (single format baked into the skill)
- Git operations (the skill proposes commit messages but never commits)

## Invocation

- `/adr "subject"` — starts the process with a given subject
- `/adr` — prompts the user for a subject

## Interactive Process (6 Steps)

### Step 1: Context
Ask: "What problem or need triggered this decision?"
- Reformulate the user's answer for clarity if needed
- One question, one message

### Step 2: Options Considered
Ask: "What alternatives have you identified? (minimum 2)"
- If the user only provides one, ask for at least one more
- The skill may suggest additional options relevant to the domain if the user is stuck

### Step 3: Decision Criteria
Ask: "What criteria are you comparing on?"
- The skill proposes relevant criteria based on the domain:
  - Model choice: inference latency, model size, mAP, training data requirements, license
  - Infra/format choice: hardware compatibility, quantization support, ecosystem maturity
  - Library choice: maintenance frequency, community size, vendor lock-in risk
  - Code architecture: testability, extensibility, complexity
- The user can accept, modify, or add criteria

### Step 4: Trade-offs & Challenge
- Build a comparison table: options x criteria
- Ask 1-3 challenge questions, one per message, contextual to the domain:

| Domain | Example challenges |
|---|---|
| Model (detection, tracking) | "What's your max latency budget for edge deployment?", "Have you checked this lib's license for defense use?" |
| Infra/format (ONNX, TensorRT) | "What target hardware are you deploying on?", "Do you need INT8 quantization?" |
| Library/framework | "How actively maintained is this lib?", "Is there vendor lock-in risk?" |
| Code architecture (API, patterns) | "Does this scale if you add a second detector type?", "What's the impact on testability?" |

Rules:
- Maximum 3 challenge questions
- If a challenge reveals a new criterion, add it to the comparison table
- The skill never forces a decision — it illuminates, the user decides

### Step 5: Decision
Ask: "Which option do you choose and why?"
- Capture the reasoning, not just the choice

### Step 6: Consequences
Help the user list:
- Positive consequences
- Negative consequences / trade-offs accepted
- Residual risks

## ADR File Format

Generated file: `docs/adr/NNNN-slug.md`

```markdown
# ADR-NNNN: Title of the Decision

**Date:** YYYY-MM-DD
**Status:** Accepted | Superseded by ADR-XXXX | Deprecated

## Context

Why this decision needs to be made. What problem or need triggered it.

## Options Considered

### Option 1: Name
Brief description of the option.

### Option 2: Name
Brief description of the option.

## Decision Criteria

- Criterion 1 (e.g., inference latency)
- Criterion 2 (e.g., model size)
- ...

## Comparison

| Criterion | Option 1 | Option 2 |
|-----------|----------|----------|
| Latency   | ...      | ...      |
| ...       | ...      | ...      |

## Decision

The chosen option and the reasoning behind it.

## Consequences

### Positive
- ...

### Negative
- ...

### Risks
- ...

## Defense & Edge Considerations

Constraints specific to the domain: edge deployment, memory footprint,
data sovereignty, security, offline capability, etc.
Only included when relevant to the decision.
```

### Conditional section
"Defense & Edge Considerations" is only included when the decision touches edge deployment, hardware constraints, data sovereignty, security, or offline capability. It is omitted for purely code-level decisions (e.g., linter choice).

## File Naming

### Numbering
- Scan `docs/adr/` for existing `NNNN-*.md` files
- Extract the highest number, increment by 1
- Pad to 4 digits: `0001`, `0002`, ...
- If directory is empty, start at `0001`

### Slug generation
- From the title: lowercase, spaces to hyphens, strip special characters
- Truncate at 50 characters on a word boundary
- Example: `"YOLOv8 vs DETR for aerial detection"` → `yolov8-vs-detr-for-aerial-detection`

## Post-Generation

- Display the generated ADR content to the user for review
- Propose a conventional commit message: `docs(adr): add ADR-NNNN <short title>`
- Never execute git commands (per CLAUDE.md rules)

## Language

- ADR content is written in English
- Skill interaction (questions, challenges) follows the conversation language (French or English depending on the user)

## Technical Implementation

- Single skill file installed via `skill-creator`
- No external dependencies
- Reads `docs/adr/` directory to determine next number
- Uses Glob tool to scan existing ADR files
- Uses Write tool to create the ADR file
