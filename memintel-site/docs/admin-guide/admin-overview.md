---
id: admin-overview
title: Admin Guide
sidebar_label: Overview
---

# Admin Guide

This guide covers the complete admin setup for a Memintel deployment — from defining the domain vocabulary through to configuring how the system delivers decisions. Follow the sections in order. Each one builds on the previous.

---

## The Admin's Role

The admin is the domain expert who governs how Memintel operates within an application. The admin does not write evaluation logic — the compiler derives that from user intent. What the admin does is define the **boundaries within which the compiler works**:

- What signals exist and how they are typed — **primitives**
- What rules the compiler must follow when resolving intent — **guardrails**
- What the application does and what domain-specific terms mean — **application context**
- What can happen when a condition fires — **actions**

Users express intent. The compiler derives logic. The admin governs both.

---

## Setup Order

Always configure in this sequence. Each layer depends on the previous one being in place.

```
Step 1 — Application Context   POST /context
         ↓
Step 2 — Primitives Config     memintel_primitives.yaml (loaded at startup)
         ↓
Step 3 — Guardrails Config     memintel_guardrails.yaml (loaded at startup)
         ↓
Step 4 — Actions Config        memintel_actions.yaml (loaded at startup)
         ↓
Step 5 — Validate              startup checks + smoke test
```

:::tip
Application context is defined via API call (`POST /context`) and can be updated at any time. The three YAML config files are loaded at server startup — changes require a restart.
:::

---

## Quick Reference

| What | How | When to change |
|---|---|---|
| Application context | `POST /context` | When domain understanding changes |
| Primitives | `memintel_primitives.yaml` + restart | When a new signal source is available |
| Guardrails | `memintel_guardrails.yaml` + restart | When regulatory or policy constraints change |
| Actions | `memintel_actions.yaml` + restart | When delivery endpoints change |

---

## Pages in This Guide

- [Application Context](/docs/admin-guide/admin-application-context) — defining domain understanding via API
- [Primitives](/docs/admin-guide/admin-primitives) — the signal vocabulary
- [Guardrails](/docs/admin-guide/admin-guardrails) — compiler policy and constraints
- [Actions](/docs/admin-guide/admin-actions) — delivery configuration
