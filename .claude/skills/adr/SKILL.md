---
name: adr
description: Generate an Architecture Decision Record through an interactive guided process. Use this skill when the user wants to document an architectural choice, compare technical options, or create an ADR. Triggers on /adr or when the user mentions "ADR", "architecture decision", or "documenter une décision".
---

# ADR — Architecture Decision Record Generator

Guide the user through creating a well-structured Architecture Decision Record via an interactive, challenging dialogue. This skill exists because architectural decisions are easy to forget and hard to reconstruct later — capturing them in real time, with the reasoning and trade-offs, is what makes them valuable.

This project (SkyWatch AI) targets defense and edge deployment, so many decisions involve constraints like inference latency, model size, hardware compatibility, data sovereignty, and offline capability. The challenge questions should reflect this when relevant.

## Process

Follow these 6 steps in order. Ask one question per message. Wait for the user's answer before moving on.

### Step 1: Subject

If the user provided a subject with the command (e.g., `/adr YOLOv8 vs DETR`), use $ARGUMENTS. Otherwise ask:

> What architectural decision do you need to document?

### Step 2: Context

Ask:

> What problem or need triggered this decision? What's the situation that makes this choice necessary right now?

Reformulate the user's answer for clarity if it's vague — confirm your understanding before moving on.

### Step 3: Options Considered

Ask:

> What alternatives have you identified? (at least 2)

If the user provides only one option, ask for at least one more. If they're stuck, suggest relevant alternatives based on the domain — you know the SkyWatch stack (YOLOv8, ByteTrack, ONNX Runtime, FastAPI, React) and can propose alternatives that make sense in this context.

### Step 4: Decision Criteria

Ask:

> What criteria matter most for comparing these options?

Then propose additional criteria relevant to the domain. Tailor suggestions to what's being decided:

- **Model choice**: inference latency, model size (params/FLOPs), mAP on relevant benchmarks, training data requirements, license compatibility
- **Infrastructure/format**: hardware compatibility, quantization support (INT8/FP16), ecosystem maturity, deployment complexity
- **Library/framework**: maintenance activity, community size, documentation quality, vendor lock-in risk
- **Code architecture**: testability, extensibility, implementation complexity, consistency with existing patterns

The user can accept, modify, or add to your suggestions.

### Step 5: Trade-offs & Challenge

Build a comparison table (options as columns, criteria as rows) and present it.

Then challenge the user's thinking with 1-3 questions, one per message. These questions should surface blind spots — things the user might not have considered. Pick questions that are relevant to the specific decision, not generic. Examples by domain:

| Domain | Example challenges |
|---|---|
| Model (detection, tracking) | "What's your max latency budget for edge inference?", "Have you verified this model's license allows defense use?", "How does this perform on small objects at high altitude?" |
| Infra/format (ONNX, TensorRT) | "What target hardware will this run on?", "Do you need INT8 quantization for deployment?", "Can this run offline?" |
| Library/framework | "When was the last release? Is it actively maintained?", "What happens if this project gets abandoned — how hard is it to switch?", "Does this introduce a dependency on a foreign cloud provider?" |
| Code architecture | "Does this scale if you add a second detector type later?", "How does this affect testability?", "Will this be easy to explain in a technical interview?" |

Rules:
- Maximum 3 challenge questions, one per message
- If a user's answer reveals a new criterion, add it to the comparison table
- Never force a decision — illuminate trade-offs, let the user decide

### Step 6: Decision

Ask:

> Which option do you choose, and why?

Capture the reasoning, not just the choice. If the user gives a one-word answer, ask them to elaborate on the key reasons.

### Step 7: Consequences

Help the user list consequences. Ask:

> What are the consequences of this decision? Think about:
> - What becomes easier or better (positive)
> - What trade-offs you're accepting (negative)
> - What could go wrong that you should watch for (risks)

## File Generation

After all steps are complete, generate the ADR file.

### Numbering

Use the Glob tool to scan `docs/adr/` for existing files matching `[0-9][0-9][0-9][0-9]-*.md`. Extract the highest number and increment by 1. If no files exist, start at `0001`. Pad to 4 digits.

### Slug

Generate from the title: lowercase, spaces to hyphens, strip special characters, truncate at 50 characters on a word boundary.

Example: `"YOLOv8 vs DETR for aerial detection"` → `0001-yolov8-vs-detr-for-aerial-detection.md`

### File Template

Write to `docs/adr/NNNN-slug.md` using this structure:

```
# ADR-NNNN: Title of the Decision

**Date:** YYYY-MM-DD
**Status:** Accepted

## Context

[From Step 2 — the problem/need that triggered this decision]

## Options Considered

### Option 1: Name
[Description from Step 3]

### Option 2: Name
[Description from Step 3]

## Decision Criteria

- [From Step 4]

## Comparison

| Criterion | Option 1 | Option 2 |
|-----------|----------|----------|
| ...       | ...      | ...      |

## Decision

[From Step 6 — the choice and the reasoning]

## Consequences

### Positive
- ...

### Negative
- ...

### Risks
- ...

## Defense & Edge Considerations

[Only include this section when the decision involves edge deployment, hardware constraints,
memory/latency budgets, data sovereignty, security, or offline capability.
Omit entirely for pure code-level decisions like linter configuration.]
```

### Post-Generation

1. Display the full ADR content to the user for a final review
2. Ask if they want to modify anything
3. Propose a conventional commit message: `docs(adr): add ADR-NNNN short title`
4. Never execute git commands — the user manages git themselves

## Language

- Write ADR content in **English**
- Conduct the interactive dialogue in whatever language the user is speaking (typically French for this project)

## Key Principles

- One question per message — don't overwhelm
- Challenge thoughtfully — the goal is to help the user think deeper, not to slow them down
- The "Defense & Edge Considerations" section is conditional — only include it when genuinely relevant
- Keep the ADR concise — each section should be a few sentences, not paragraphs
- The comparison table is the centerpiece — make it clear and honest
