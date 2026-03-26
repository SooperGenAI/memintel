---
id: clinical-trials-fda
title: Clinical Trials and FDA Compliance
sidebar_label: Clinical Trials / FDA
---

# Tutorial: Clinical Trials and FDA Compliance

A walkthrough of three critical intelligence use cases in the clinical trial and regulatory approval process — safety signal monitoring, site risk-based monitoring, and regulatory submission readiness. Each addresses a different phase of the drug development lifecycle, and each demonstrates the dual memory architecture that makes Memintel particularly well-suited to regulated research environments.

:::note What you'll build
Three deterministic monitoring systems that continuously evaluate a trial's safety, operational, and regulatory state — detecting emerging signals before they become reportable events, identifying site quality deterioration before it becomes a data integrity issue, and assessing submission readiness against current FDA expectations before a Complete Response Letter arrives.
:::

---

## Why Clinical Trials Fit Memintel

The clinical trial and regulatory approval process has exactly the three properties that make Memintel's architecture most valuable.

**Dual memory is the core of the problem.** Every significant trial decision requires reconciling two continuously evolving states:

- **Internal state** — the trial's own data: enrollment numbers, adverse event counts, protocol deviations, interim analysis results, site performance metrics
- **External state** — the regulatory environment: current FDA guidance documents, ICH guidelines, precedent decisions from comparable submissions, post-market safety signals for related compounds in FAERS

Neither memory alone is sufficient. An adverse event rate is only meaningful relative to what the FDA has historically accepted for this indication. A protocol deviation only matters in the context of current GCP expectations. The value comes from the interaction between the two.

**Auditability is legally mandated and extremely specific.** When the FDA asks "why was this adverse event not escalated to a serious adverse event?", the answer must include what data was evaluated, what thresholds were applied, when the evaluation occurred, and who approved the definition. Memintel's immutable audit trail produces this documentation automatically as a byproduct of normal operation.

**The external environment evolves continuously.** FDA guidance is not static. Draft guidances become final. Advisory committee recommendations shift expectations. Post-market safety signals for related compounds create new monitoring requirements mid-trial. A trial that begins with a specific safety monitoring plan may need to evaluate its data against new FDA signals two years later — without retroactively changing what past decisions meant. This is precisely the guardrails update and recompilation workflow.

---

## The Three Roles

| Role | Who they are | What they do |
|---|---|---|
| **Data Engineer** | Clinical data / biostatistics team | Builds EDC and safety database pipelines, writes resolver functions, integrates FAERS and FDA guidance feeds |
| **Admin** | Medical Monitor / VP Regulatory Affairs | Maintains config files — primitives, guardrails — governs safety thresholds, GCP expectations, and indication-specific parameters |
| **User** | Clinical operations / pharmacovigilance / regulatory affairs | Expresses monitoring intent via internal platform — creates, manages, and reviews monitoring tasks |

:::warning Domain expertise is essential
The guardrails and parameter priors for an oncology trial are completely different from a cardiovascular or rare disease trial. The admin configuration work requires genuine regulatory expertise — not just data engineering. The primitives encode clinical meaning, and that meaning must be governed by qualified clinical and regulatory professionals.
:::

---

## Use Case 1 — Safety Signal Monitoring and Pharmacovigilance

### The Problem

*Perspective: Sponsor (Pharma / Biotech)*

Clinical trials are required to continuously monitor for adverse events (AEs), serious adverse events (SAEs), and suspected unexpected serious adverse reactions (SUSARs). Current approaches are predominantly manual — a medical monitor reviews AE listings against a static set of criteria at weekly or monthly data cuts. The critical window between data cut and medical review is where emerging safety signals go undetected.

The deeper problem is that an AE is not just a data point in isolation. It is meaningful relative to:
- The expected background rate for the indication
- The comparator arm's AE profile in the same trial
- The compound's class-level safety signal in the FDA's FAERS database
- The trial's pre-specified stopping rules
- Recent FDA safety communications for related compounds

Current systems evaluate the AE. Memintel evaluates the AE in context.

### The Architecture Boundary

```
Raw trial data    →  Signal Extraction  →  Primitives  →  Memintel
(EDC, safety DB,     (MedDRA coding,       (typed,          (deterministic
 FAERS, FDA docs)     severity scoring,     normalised)       signal evaluation)
                      relatedness NLP)
```

### Primitive Design

```yaml
# memintel_primitives_safety.yaml

primitives:

  # Patient-level AE signals
  - id: patient.ae_severity_score
    type: float
    source: safety_database
    entity: patient_id
    description: Composite AE severity score based on MedDRA grades, 0-1

  - id: patient.ae_relatedness_signal
    type: float
    source: nlp_pipeline
    entity: patient_id
    description: LLM-extracted probability of drug-relatedness from narrative, 0-1

  - id: patient.ae_relatedness_confidence
    type: float
    source: nlp_pipeline
    entity: patient_id
    description: Confidence of relatedness extraction, 0-1

  - id: patient.sae_count_30d
    type: int
    source: safety_database
    entity: patient_id
    description: Serious adverse events reported in last 30 days

  - id: patient.concomitant_medication_risk_score
    type: float
    source: safety_database
    entity: patient_id
    description: Drug-drug interaction risk score from concomitant medications, 0-1

  # Trial-level safety signals
  - id: trial.ae_incidence_rate_treatment_arm
    type: float
    source: edc_pipeline
    entity: trial_id
    description: AE incidence rate in treatment arm as percentage

  - id: trial.ae_incidence_rate_comparator_arm
    type: float
    source: edc_pipeline
    entity: trial_id
    description: AE incidence rate in comparator / placebo arm as percentage

  - id: trial.treatment_vs_comparator_ratio
    type: float
    source: edc_pipeline
    entity: trial_id
    description: Ratio of treatment AE rate to comparator AE rate

  - id: trial.stopping_rule_proximity_score
    type: float
    source: safety_monitoring_pipeline
    entity: trial_id
    description: How close current safety data is to pre-specified stopping thresholds, 0-1

  - id: trial.dsmb_flag
    type: boolean
    source: safety_monitoring_pipeline
    entity: trial_id
    description: True if DSMB has flagged this trial for enhanced monitoring

  # External regulatory signals
  - id: compound.faers_signal_score
    type: float
    source: faers_pipeline
    entity: compound_id
    description: Disproportionality signal score from FDA FAERS for this compound class, 0-1

  - id: compound.class_safety_alert_flag
    type: boolean
    source: fda_guidance_pipeline
    entity: compound_id
    description: True if FDA has issued a safety communication for this compound class in last 90 days

  - id: indication.historical_sae_threshold
    type: float
    source: fda_precedent_pipeline
    entity: indication_id
    description: Historical SAE rate accepted in FDA-approved trials for this indication

  - id: indication.recent_crl_safety_flag
    type: boolean
    source: fda_precedent_pipeline
    entity: indication_id
    description: True if any CRL in last 24 months cited safety concerns for this indication
```

### Resolver Examples

```python
# resolvers/safety_resolvers.py

@registry.resolver("patient.ae_relatedness_signal")
async def resolve_ae_relatedness(entity_id: str, timestamp: datetime) -> float:
    # Fetch narrative from safety database as of timestamp
    ae_records = await safety_db.execute("""
        SELECT ae_narrative, investigator_assessment, onset_timing_days
        FROM adverse_events
        WHERE patient_id = $1
          AND report_date <= $2
          AND is_active = true
        ORDER BY report_date DESC
        LIMIT 5
    """, entity_id, timestamp)

    if not ae_records:
        return 0.0

    # Pass to LLM for relatedness assessment — compile time only
    result = await llm_client.assess_relatedness(
        narratives=[r.ae_narrative for r in ae_records],
        investigator_assessments=[r.investigator_assessment for r in ae_records],
        onset_timings=[r.onset_timing_days for r in ae_records]
    )
    return result.relatedness_probability


@registry.resolver("compound.faers_signal_score")
async def resolve_faers_signal(entity_id: str, timestamp: datetime) -> float:
    # entity_id is compound_id — fetch FAERS disproportionality score as of timestamp
    # FAERS is updated quarterly — use the version current at evaluation time
    result = await faers_api.get_signal_score(
        compound_id=entity_id,
        as_of_date=timestamp,
        metric="proportional_reporting_ratio"
    )
    return float(result.score or 0)


@registry.resolver("trial.treatment_vs_comparator_ratio")
async def resolve_treatment_ratio(entity_id: str, timestamp: datetime) -> float:
    result = await db.execute("""
        WITH arm_rates AS (
            SELECT
                arm_type,
                COUNT(CASE WHEN has_ae THEN 1 END)::float
                / NULLIF(COUNT(*), 0) AS ae_rate
            FROM patient_trial_enrollment
            WHERE trial_id = $1
              AND enrollment_date <= $2
            GROUP BY arm_type
        )
        SELECT
            MAX(CASE WHEN arm_type = 'treatment' THEN ae_rate END)
            / NULLIF(MAX(CASE WHEN arm_type = 'comparator' THEN ae_rate END), 0)
        FROM arm_rates
    """, entity_id, timestamp)
    return float(result.scalar() or 1.0)


@registry.resolver("indication.historical_sae_threshold")
async def resolve_historical_sae_threshold(entity_id: str, timestamp: datetime) -> float:
    # Query FDA precedent database for accepted SAE rates in approved trials
    # for this indication — as of the given timestamp
    result = await fda_precedent_db.execute("""
        SELECT AVG(sae_rate_treatment_arm) AS historical_threshold
        FROM approved_trial_data
        WHERE indication_id = $1
          AND approval_date <= $2
          AND trial_phase = 3
        ORDER BY approval_date DESC
        LIMIT 10
    """, entity_id, timestamp)
    return float(result.scalar() or 0.15)  # default 15% if no precedent
```

### Guardrails

```yaml
# memintel_guardrails_safety.yaml

type_strategy_map:
  float:    [threshold, percentile, z_score]
  int:      [threshold, percentile]
  boolean:  [equals]

parameter_priors:
  patient.ae_severity_score:
    low_severity:     { threshold: 0.5 }   # Grade 2+ AEs
    medium_severity:  { threshold: 0.7 }   # Grade 3+ AEs
    high_severity:    { threshold: 0.9 }   # Grade 4/5 AEs

  patient.ae_relatedness_signal:
    low_severity:     { threshold: 0.4 }   # Possibly related
    medium_severity:  { threshold: 0.65 }  # Probably related
    high_severity:    { threshold: 0.85 }  # Definitely related

  trial.treatment_vs_comparator_ratio:
    low_severity:     { threshold: 1.5 }   # 50% higher than comparator
    medium_severity:  { threshold: 2.0 }   # 2x comparator rate
    high_severity:    { threshold: 3.0 }   # 3x comparator rate

  trial.stopping_rule_proximity_score:
    low_severity:     { threshold: 0.60 }  # 40% headroom to stopping rule
    medium_severity:  { threshold: 0.75 }  # 25% headroom
    high_severity:    { threshold: 0.90 }  # 10% headroom — urgent

  compound.faers_signal_score:
    low_severity:     { threshold: 0.5 }
    medium_severity:  { threshold: 0.7 }
    high_severity:    { threshold: 0.85 }

bias_rules:
  conservative:   high_severity    # oncology / high-risk populations
  standard:       medium_severity
  proactive:      low_severity
  early_signal:   low_severity

# Indication-specific overrides loaded from domain config
# e.g. oncology trials use different severity mappings than cardiology
indication_context: loaded_from_domain_config
```

### User Creates Tasks

```typescript
// Medical monitor creates safety monitoring tasks

// Task 1 — Patient-level SAE early signal
const patientSafetyTask = await client.tasks.create({
    intent: "Alert me when a patient shows emerging safety signals that may require SAE assessment",
    entityScope: "all_enrolled_patients",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/safety-patient-alert"
    },
    dryRun: true
});
// Resolves to composite: ae_severity_score + ae_relatedness_signal
// weighted by concomitant_medication_risk_score

// Task 2 — Trial-level treatment vs comparator signal
const trialSafetyTask = await client.tasks.create({
    intent: "Alert me when treatment arm AE rate significantly exceeds comparator arm",
    entityScope: "all_active_trials",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/safety-trial-alert"
    }
});
// Resolves to: treatment_vs_comparator_ratio > 2.0 (medium severity)

// Task 3 — External FDA class safety alert
const fdaAlertTask = await client.tasks.create({
    intent: "Alert me immediately when FDA issues a safety communication for our compound class",
    entityScope: "all_active_compounds",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/safety-fda-alert",
        priority: "immediate"
    }
});
// Resolves to: compound.class_safety_alert_flag = true
// Event-driven — fires when FDA guidance pipeline detects new communication

// Task 4 — Stopping rule proximity
const stoppingRuleTask = await client.tasks.create({
    intent: "Alert me when trial safety data is approaching pre-specified stopping thresholds",
    entityScope: "all_active_trials",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/safety-stopping-rule"
    }
});
// Resolves to: stopping_rule_proximity_score > 0.75 (25% headroom remaining)
```

### What the Alert Looks Like

```
🔴 Safety Alert — Patient SAE Assessment Required

Trial:      TRIAL-2024-007  (Phase 3, Oncology — NSCLC)
Patient:    PT-00847  (Site 12, Week 34)
Triggered:  2024-03-15 09:23 UTC

Signal breakdown:
  AE severity score:       0.82  (Grade 3 events — threshold: 0.70)
  Relatedness signal:      0.71  (Probably related — threshold: 0.65)
  Concomitant med risk:    0.58  (moderate interaction risk)

External context:
  FAERS class signal:      0.74  (elevated for checkpoint inhibitor class)
  FDA class alert:         Yes   (irAE guidance updated 2024-02-28)
  Historical SAE rate:     8.2%  (FDA precedent for NSCLC indication)
  Current trial SAE rate:  11.4% (39% above historical precedent)

Stopping rule proximity:  0.61  (39% headroom remaining)

Action required: Medical monitor review within 24 hours
→ Open SAE workflow   → Review patient narrative   → DSMB notification
```

### The Safety Monitoring Advantage

The `indication.historical_sae_threshold` primitive is what makes this genuinely intelligent rather than just threshold-based. A trial SAE rate of 11.4% evaluated in isolation may or may not be concerning — it depends entirely on what the FDA has accepted for this indication in prior approved trials. By maintaining the FDA precedent database as external memory, the system evaluates the trial's safety profile against the regulatory standard that actually matters — not against an internal threshold written at protocol design.

---

## Use Case 2 — Site Risk-Based Monitoring

### The Problem

*Perspective: Clinical Operations*

Multi-site trials distribute patients across dozens or hundreds of clinical sites. Not all sites perform equally. Some enroll slowly. Some generate disproportionate protocol deviations. Some have data quality issues that compromise the integrity of the efficacy dataset. ICH E6(R2) and FDA's risk-based monitoring guidance require sponsors to implement risk-based monitoring approaches — but current implementations are still largely calendar-driven.

The critical insight: site quality problems do not appear suddenly. They develop gradually — enrollment slows, query rates climb, deviation frequency rises, data entry latency extends. Each signal individually may be within tolerance. The trajectory across all signals is the early warning that a site is deteriorating.

### Primitive Design

```yaml
# memintel_primitives_site_monitoring.yaml

primitives:

  # Enrollment signals
  - id: site.enrollment_velocity_trend_8w
    type: time_series<float>
    source: edc_pipeline
    entity: site_id
    description: Weekly enrollment count over last 8 weeks — enables trend detection

  - id: site.enrollment_vs_target_ratio
    type: float
    source: edc_pipeline
    entity: site_id
    description: Actual cumulative enrollment / target cumulative enrollment

  - id: site.screen_failure_rate
    type: float
    source: edc_pipeline
    entity: site_id
    description: Screen failure rate — high rate may indicate eligibility issues

  # Data quality signals
  - id: site.open_query_rate
    type: float
    source: edc_pipeline
    entity: site_id
    description: Open data queries per CRF page — above peer median indicates data quality issues

  - id: site.query_resolution_days
    type: float
    source: edc_pipeline
    entity: site_id
    description: Average days to resolve open queries

  - id: site.data_entry_latency_days
    type: float
    source: edc_pipeline
    entity: site_id
    description: Average days from visit to data entry

  - id: site.missing_data_rate
    type: float
    source: edc_pipeline
    entity: site_id
    description: Percentage of required fields with missing data

  # Protocol compliance signals
  - id: site.protocol_deviation_rate_30d
    type: float
    source: safety_database
    entity: site_id
    description: Protocol deviations per patient per 30 days

  - id: site.major_deviation_count_90d
    type: int
    source: safety_database
    entity: site_id
    description: Major protocol deviations in last 90 days

  - id: site.informed_consent_deficiency_flag
    type: boolean
    source: safety_database
    entity: site_id
    description: True if any informed consent deficiencies identified in last 6 months

  # Site context
  - id: site.peer_deviation_percentile
    type: float
    source: benchmarking_pipeline
    entity: site_id
    description: Site's deviation rate percentile within trial — enables relative comparison

  - id: site.investigator_experience_score
    type: float
    source: site_qualification_pipeline
    entity: site_id
    description: Investigator's prior trial experience and performance score, 0-1

  # External regulatory signals
  - id: site.fda_inspection_history_flag
    type: boolean
    source: fda_inspection_pipeline
    entity: site_id
    description: True if site has FDA inspection findings in last 3 years

  - id: country.gcp_compliance_risk
    type: categorical
    source: regulatory_pipeline
    entity: country_id
    description: ICH GCP compliance risk classification for site country — low/medium/high
```

### Resolver Examples

```python
# resolvers/site_monitoring_resolvers.py

@registry.resolver("site.enrollment_velocity_trend_8w")
async def resolve_enrollment_trend(entity_id: str, timestamp: datetime) -> list[float]:
    result = await db.execute("""
        SELECT
            DATE_TRUNC('week', consent_date) AS week,
            COUNT(*) AS enrollments
        FROM patient_trial_enrollment
        WHERE site_id = $1
          AND consent_date BETWEEN $2 - INTERVAL '8 weeks' AND $2
        GROUP BY week
        ORDER BY week
    """, entity_id, timestamp)
    return [float(row.enrollments) for row in result.fetchall()]


@registry.resolver("site.open_query_rate")
async def resolve_query_rate(entity_id: str, timestamp: datetime) -> float:
    result = await db.execute("""
        SELECT
            COUNT(CASE WHEN status = 'open' THEN 1 END)::float
            / NULLIF(COUNT(DISTINCT crf_page_id), 0)
        FROM data_queries
        WHERE site_id = $1
          AND created_date <= $2
    """, entity_id, timestamp)
    return float(result.scalar() or 0)


@registry.resolver("site.peer_deviation_percentile")
async def resolve_peer_percentile(entity_id: str, timestamp: datetime) -> float:
    # What percentile is this site's deviation rate within the trial?
    result = await db.execute("""
        WITH site_rates AS (
            SELECT
                site_id,
                COUNT(*)::float / NULLIF(patient_count, 0) AS deviation_rate
            FROM protocol_deviations pd
            JOIN site_enrollment_counts sec ON pd.site_id = sec.site_id
            WHERE pd.trial_id = (
                SELECT trial_id FROM sites WHERE id = $1
            )
            AND pd.deviation_date <= $2
            GROUP BY pd.site_id, sec.patient_count
        )
        SELECT
            PERCENT_RANK() OVER (ORDER BY deviation_rate) AS percentile
        FROM site_rates
        WHERE site_id = $1
    """, entity_id, timestamp)
    return float(result.scalar() or 0.5)
```

### User Creates Tasks

```typescript
// Clinical operations manager creates site monitoring tasks

// Task 1 — Enrollment velocity declining
const enrollmentTask = await client.tasks.create({
    intent: "Alert me when a site's enrollment velocity is declining significantly",
    entityScope: "all_active_sites",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/site-enrollment" }
});
// Resolves to: change strategy on enrollment_velocity_trend_8w
// Fires when slope turns significantly negative — not just when enrollment is low

// Task 2 — Data quality deterioration
const dataQualityTask = await client.tasks.create({
    intent: "Alert me when a site shows signs of data quality deterioration",
    entityScope: "all_active_sites",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/site-quality" }
});
// Resolves to: composite of open_query_rate + data_entry_latency_days
// + missing_data_rate vs peer median

// Task 3 — Protocol deviation risk
const deviationTask = await client.tasks.create({
    intent: "Alert me when a site has a significantly elevated protocol deviation rate relative to peers",
    entityScope: "all_active_sites",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/site-deviation" }
});
// Resolves to: peer_deviation_percentile > 85th percentile

// Task 4 — Informed consent deficiency — immediate
const consentTask = await client.tasks.create({
    intent: "Alert me immediately when any informed consent deficiency is identified",
    entityScope: "all_active_sites",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/site-consent",
        priority: "immediate"
    }
});
// Resolves to: informed_consent_deficiency_flag = true
```

### What the Alert Looks Like

```
⚠️ Site Risk Alert — Data Quality Deterioration

Trial:   TRIAL-2024-007  (Phase 3, Oncology)
Site:    Site 23 — University Medical Center, Frankfurt
PI:      Dr. Klaus Weber
Patients enrolled: 34  |  Target: 40

Quality signals (last 60 days):
  Open query rate:       4.2 per page  (trial median: 1.8 — 2.3x elevated)
  Query resolution:      18.4 days     (trial median: 7.2 — 2.6x elevated)
  Data entry latency:    9.1 days      (trial median: 3.4 — 2.7x elevated)
  Missing data rate:     8.3%          (trial median: 2.1%)

Protocol compliance:
  Deviation rate:        84th percentile within trial
  Major deviations 90d:  3  (trial median: 0.8)

External context:
  FDA inspection history: Yes — Form 483 issued 2022 (data integrity)
  Country GCP risk:       Medium (Germany — EMA oversight)

Recommendation: Triggered monitoring visit recommended
→ Schedule SDV visit   → Notify CRA   → Issue corrective action plan
```

### The Site Monitoring Advantage

The `site.peer_deviation_percentile` primitive illustrates the cross-memory advantage clearly. A deviation rate of 4.2 deviations per patient per 30 days means nothing in isolation. Evaluated against the distribution across all sites in the trial, a site in the 84th percentile for deviations is definitively an outlier — regardless of the absolute number. The peer comparison is computed automatically from the internal trial data. No rule was written for this; the intent "significantly elevated relative to peers" compiled to a percentile strategy automatically.

---

## Use Case 3 — Regulatory Submission Readiness

### The Problem

*Perspective: Regulatory Affairs*

Before submitting an NDA or BLA, regulatory affairs teams spend months assessing whether the data package meets FDA expectations. The most expensive outcome in drug development is a Complete Response Letter — an FDA refusal to approve citing deficiencies that could have been identified and addressed before submission.

The problem is information asymmetry. FDA expectations evolve continuously through guidance documents, advisory committee decisions, approval letters, and CRLs — all of which are publicly available on FDA.gov and Drugs@FDA. But most regulatory teams do not have a systematic way to continuously monitor this corpus and evaluate their trial data against it. They rely on regulatory consultants, periodic literature reviews, and point-in-time assessments — typically done once, late in development.

Memintel with the FDA public corpus as external memory changes this: the submission package is continuously evaluated against current FDA expectations throughout development — not just in the 90 days before submission.

### Primitive Design

```yaml
# memintel_primitives_submission.yaml

primitives:

  # Efficacy endpoint signals
  - id: trial.primary_endpoint_effect_size
    type: float
    source: biostatistics_pipeline
    entity: trial_id
    description: Current estimated effect size for primary efficacy endpoint

  - id: trial.primary_endpoint_ci_lower
    type: float
    source: biostatistics_pipeline
    entity: trial_id
    description: Lower bound of 95% CI for primary endpoint — must exceed FDA minimum

  - id: indication.fda_minimum_effect_size
    type: float
    source: fda_precedent_pipeline
    entity: indication_id
    description: Minimum clinically meaningful effect size accepted by FDA for this indication

  - id: indication.recent_approval_effect_size_median
    type: float
    source: fda_precedent_pipeline
    entity: indication_id
    description: Median effect size of recent FDA approvals for this indication

  # Safety profile signals
  - id: trial.benefit_risk_score
    type: float
    source: safety_monitoring_pipeline
    entity: trial_id
    description: Computed benefit-risk ratio vs indication standard of care, 0-1

  - id: indication.crl_safety_pattern_score
    type: float
    source: fda_precedent_pipeline
    entity: indication_id
    description: Similarity of current safety profile to safety profiles that triggered CRLs, 0-1

  # Statistical and data integrity signals
  - id: trial.multiplicity_adjustment_flag
    type: boolean
    source: biostatistics_pipeline
    entity: trial_id
    description: True if multiple endpoint testing requires additional multiplicity adjustment

  - id: trial.missing_data_impact_score
    type: float
    source: biostatistics_pipeline
    entity: trial_id
    description: Estimated impact of missing data on primary endpoint — sensitivity analysis

  # Regulatory precedent signals
  - id: indication.advisory_committee_flag
    type: boolean
    source: fda_guidance_pipeline
    entity: indication_id
    description: True if FDA advisory committee meeting likely required for this indication

  - id: indication.recent_guidance_change_flag
    type: boolean
    source: fda_guidance_pipeline
    entity: indication_id
    description: True if FDA issued new guidance for this indication in last 12 months

  - id: submission.cmc_deficiency_risk_score
    type: float
    source: cmc_pipeline
    entity: submission_id
    description: Risk score for Chemistry Manufacturing and Controls deficiencies, 0-1

  - id: submission.labeling_alignment_score
    type: float
    source: regulatory_pipeline
    entity: submission_id
    description: Alignment of proposed labeling with FDA precedent for this indication, 0-1
```

### User Creates Tasks

```typescript
// Regulatory affairs creates submission readiness monitoring tasks

// Task 1 — Efficacy endpoint vs FDA minimum
const efficacyTask = await client.tasks.create({
    intent: "Alert me when the primary endpoint effect size is at risk of falling below FDA minimum requirements for this indication",
    entityScope: "all_active_submissions",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/reg-efficacy" }
});
// Resolves to: primary_endpoint_ci_lower approaching indication.fda_minimum_effect_size
// Fires when the lower CI bound is within 15% of the FDA minimum

// Task 2 — Safety profile resembles CRL patterns
const safetyProfileTask = await client.tasks.create({
    intent: "Alert me when our safety profile resembles profiles that have triggered CRLs for this indication",
    entityScope: "all_active_submissions",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/reg-safety-profile" }
});
// Resolves to: crl_safety_pattern_score > 0.65
// Most powerful early warning for submission risk

// Task 3 — New FDA guidance affecting submission
const guidanceTask = await client.tasks.create({
    intent: "Alert me immediately when FDA issues new guidance relevant to our indication or submission",
    entityScope: "all_active_indications",
    delivery: {
        type: "webhook",
        endpoint: "https://myapp.com/hooks/reg-guidance",
        priority: "immediate"
    }
});
// Event-driven — fires when FDA guidance pipeline detects new document

// Task 4 — CMC deficiency risk
const cmcTask = await client.tasks.create({
    intent: "Alert me when manufacturing data shows patterns associated with CMC deficiencies",
    entityScope: "all_active_submissions",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/reg-cmc" }
});
```

### What the Alert Looks Like

```
⚠️ Submission Readiness Alert — Safety Profile Risk

Submission:  NDA-2024-XXXXX  (Compound ABC, NSCLC 2L)
Stage:       Phase 3 ongoing  (interim analysis 2)
Triggered:   2024-03-15 11:30 UTC

Efficacy status:
  Primary endpoint effect size:  HR 0.71  (FDA minimum: HR < 0.80 ✓)
  95% CI lower bound:           HR 0.58  (above minimum ✓)
  vs recent approvals:           HR 0.71 vs median HR 0.68 (comparable ✓)

⚠️ Safety profile risk detected:
  CRL pattern similarity score:  0.71  (threshold: 0.65)

  Your safety profile resembles profiles that triggered CRLs in:
  • Compound XYZ (2022 CRL) — hepatotoxicity signal at similar incidence
  • Compound DEF (2021 CRL) — similar grade 3+ AE distribution

  Specific similarities:
  • Grade 3+ hepatic AEs:  4.2%  (CRL precedents: 3.8-4.6%)
  • Treatment discontinuation due to AEs: 18%  (CRL precedents: 15-21%)

External signals:
  Recent FDA guidance:  Yes — hepatotoxicity monitoring guidance updated 2024-01
  Advisory committee:   Likely required based on indication precedent

Recommendation: Proactive FDA meeting recommended before NDA submission
→ Schedule Type B meeting   → Review hepatotoxicity risk management plan
→ Update benefit-risk assessment
```

### The Submission Readiness Advantage

The `indication.crl_safety_pattern_score` primitive — comparing the current trial's safety profile against safety profiles that historically triggered CRLs — is the most commercially valuable primitive in this system. A CRL costs 12-18 months and tens of millions of dollars in delayed revenue. Identifying a CRL-pattern safety profile 18 months before submission, rather than in the FDA's response letter, changes the entire regulatory strategy. This is external memory — the Drugs@FDA CRL corpus — doing work that no amount of internal data analysis can replicate.

---

## The System Response Loop (All Three Use Cases)

```
Trigger: Scheduled (daily/weekly) OR event-driven
  (new AE reported, FDA guidance published, EDC data locked)
          ↓
For each entity in scope:
  Resolvers fetch point-in-time values:
    → EDC / Safety database: AE counts, enrollment, deviations, data quality
    → FAERS API: class-level safety signals (as of timestamp)
    → FDA precedent database: historical thresholds, CRL patterns (as of timestamp)
    → FDA guidance pipeline: new communications (event-driven)
  Concept computation runs
  Condition evaluates
          ↓
  If fired:
    Alert delivered with full context and recommended action
    Decision logged with complete audit trail
    Workflow integration triggered (SAE system, CTMS, eTMF)
          ↓
  If not: Decision logged. No alert.
          ↓
Next scheduled evaluation or event trigger
```

### Event-driven vs scheduled

| Task | Best trigger | Reason |
|---|---|---|
| Patient AE monitoring | **Daily** | AE data locked daily in most EDC systems |
| Trial-level safety signal | **Weekly** | Aggregate signals change weekly at data cut |
| FDA class safety alert | **Event-driven** on FDA guidance update | Federal safety obligation begins at publication |
| Site enrollment trend | **Weekly** | Trend requires accumulation of data points |
| Site data quality | **Weekly** | Quality metrics lag by one data cut cycle |
| Submission readiness | **Monthly + event-driven on guidance** | Efficacy data updates monthly; guidance is event-driven |

---

## Task Management

### User controls

```typescript
// Medical monitor views all active safety monitoring tasks
const tasks = await client.tasks.list({ owner: "current_user" });

// Pause site monitoring during planned database lock
await client.tasks.pause("tsk_site_quality_all", {
    reason: "Database lock in progress — data in flux until lock complete 2024-03-16"
});

// Check alert quality — are we generating actionable alerts?
const taskDetail = await client.tasks.get("tsk_patient_ae_monitoring");
console.log(`Fired ${taskDetail.fire_count_30d} times — ${taskDetail.fire_count_30d} SAE assessments triggered`);
console.log(`False positive rate: ${taskDetail.false_positive_rate_30d}`);
```

### Admin visibility

```typescript
// Medical monitor / VP Regulatory reviews all tasks across all trials
const allTasks = await adminClient.tasks.list();

// Version audit — critical for FDA inspection readiness
// Regulator may ask: which monitoring logic was active on date X?
const versionHistory = await adminClient.conditions.versionHistory({
    conditionId: "patient.sae_assessment_required",
    from: "2024-01-01",
    to: "2024-03-31"
});

// Which condition version was active on each date during the trial period?
// This is the Memintel audit trail — immutable, reproducible, FDA-defensible

// Suspend monitoring if safety database is under maintenance
await adminClient.tasks.suspend("tsk_patient_ae_monitoring", {
    reason: "Safety database maintenance — data integrity cannot be guaranteed"
});
```

---

## Calibration

### Safety monitoring — driven by medical monitor review outcomes

```typescript
// An AE alert fired but medical monitor determined no SAE required — false positive
await client.feedback.decision({
    conditionId: "patient.sae_assessment_required",
    conditionVersion: "1.0",
    entity: "PT-00847",
    timestamp: "2024-03-15T09:23:00Z",
    feedback: "false_positive",
    note: "AE was Grade 3 but unrelated to study drug — pre-existing condition confirmed"
});
```

### Site monitoring — driven by triggered monitoring visit outcomes

```typescript
// A site was not flagged but triggered monitoring visit found significant issues — false negative
await client.feedback.decision({
    conditionId: "site.quality_deterioration_risk",
    conditionVersion: "1.0",
    entity: "site_23",
    timestamp: "2024-01-15T00:00:00Z",
    feedback: "false_negative",
    note: "On-site visit found systematic data entry issues — not detected by query rate alone"
});
// Calibration may recommend adding data_entry_latency weight to composite
```

### Submission readiness — driven by FDA response outcomes

```typescript
// CRL received citing safety concern that was not flagged — false negative
await client.feedback.decision({
    conditionId: "submission.crl_safety_pattern_risk",
    conditionVersion: "1.0",
    entity: "nda_2024_xxxxx",
    timestamp: "2024-06-01T00:00:00Z",
    feedback: "false_negative",
    note: "CRL cited hepatotoxicity — CRL pattern score was 0.58, below 0.65 threshold"
});
// Calibration recommendation: lower crl_safety_pattern_score threshold to 0.55
```

---

## Full Lifecycle Diagram

```
SETUP (one time)
──────────────────────────────────────────────────────────
Data Engineer:   Configure EDC, safety DB, CTMS pipeline resolvers
Data Engineer:   Configure FAERS, FDA precedent, FDA guidance feeds
Data Engineer:   Build CRL pattern similarity scorer
Admin:           Define primitives → memintel_primitives_safety.yaml
                                      memintel_primitives_site_monitoring.yaml
                                      memintel_primitives_submission.yaml
Admin:           Define guardrails  → indication-specific config
                                      (oncology vs cardiology vs rare disease)
System:          Load config at startup


TASK CREATION
──────────────────────────────────────────────────────────
User (medical monitor / reg affairs):
              Express monitoring intent via clinical platform
Bot:          POST /tasks/create?dryRun=true
Memintel:     Compile intent → concept + condition
              (using indication-specific guardrails)
User:         Review compiled condition in plain English
User:         Confirm → task activated


ONGOING EVALUATION LOOP
──────────────────────────────────────────────────────────
Trigger: Scheduled (daily/weekly) OR event (FDA guidance, EDC data lock)
  → Resolvers fetch point-in-time values:
      EDC / Safety DB (as of timestamp — not current state)
      FAERS API (quarterly update version current at timestamp)
      FDA precedent database (approval history as of timestamp)
      FDA guidance pipeline (event-triggered on new document)
  → Concept computation runs
  → Condition evaluates
  → If fired: alert with context, audit log entry, workflow trigger
  → If not:   audit log entry, no alert


TASK MANAGEMENT
──────────────────────────────────────────────────────────
User:    View / pause own tasks
         (e.g. pause during database lock)
Admin:   View all tasks across all trials
Admin:   Version history for FDA inspection readiness
Admin:   Suspend tasks during system maintenance


CALIBRATION CYCLE
──────────────────────────────────────────────────────────
Ground truth: SAE review outcomes, monitoring visit findings, FDA responses
User:    Submit feedback (false positive / false negative)
Admin:   Review calibration recommendation
Admin:   Approve → new condition version (immutable)
         Document rationale for FDA inspection record
User:    Rebind task to new version (explicit, never automatic)
         ↓
Back to ONGOING LOOP with improved detection sensitivity
```

---

## Role Summary

| Step | Who | Safety Monitoring | Site Monitoring | Submission Readiness |
|---|---|---|---|---|
| Data pipeline | **Data Eng.** | EDC, safety DB, FAERS, FDA guidance | EDC, CTMS, site qualification | Biostatistics pipeline, FDA precedent DB |
| Resolvers | **Data Eng.** | AE scoring, FAERS signal, comparator ratio | Enrollment trend, peer percentile | Effect size, CRL pattern scorer |
| Primitives | **Admin** | `memintel_primitives_safety.yaml` | `memintel_primitives_site_monitoring.yaml` | `memintel_primitives_submission.yaml` |
| Guardrails | **Admin** | Indication-specific severity priors | GCP compliance thresholds | FDA minimum effect size priors |
| Task creation | **User** | Medical monitor | Clinical operations | Regulatory affairs |
| Feedback | **User** | SAE review outcomes | Monitoring visit findings | FDA CRL / approval outcomes |
| Calibration | **Admin** | Documents rationale for FDA record | Approves threshold adjustments | Approves pattern score thresholds |

---

## Why This Domain Specifically Needs Memintel

Three properties make clinical trials and FDA compliance one of the strongest structural fits for Memintel's architecture.

**The auditability requirement is legally mandated at a level that exceeds most other domains.** During an FDA inspection, the agency can request documentation of every safety monitoring decision — what data was evaluated, what thresholds were applied, who defined those thresholds, and whether the same thresholds were applied consistently across all patients. Memintel's immutable audit trail, version-pinned condition history, and reproducible evaluations satisfy this requirement automatically. A rules system can provide some of this documentation, but it requires forensic reconstruction. Memintel produces it as a byproduct of normal operation.

**The dual memory structure is the core of every meaningful decision.** An AE rate, a deviation count, a statistical result — none of these are meaningful without the external regulatory context. What has the FDA accepted for this indication? What patterns triggered CRLs in comparable programs? What new guidance has the FDA issued since the protocol was written? Memintel's primitive model treats internal trial data and external FDA regulatory data identically — both are typed signals that the evaluation engine can combine. No other architecture makes this as clean.

**The cost of a missed signal is uniquely high.** A missed early safety signal can mean patient harm. A missed submission readiness signal can mean a CRL — 12-18 months of delay and tens of millions in lost revenue. A missed site quality signal can mean data integrity findings that put the entire efficacy dataset at risk. In no other domain does the early warning capability of intent-based trajectory monitoring have higher direct commercial and human value.

---

## Next Steps

- [Core Concepts](/docs/intro/core-concepts) — understand the ψ → φ → α model in depth
- [Why not SQL and rules?](/docs/intro/why-not-rules) — the architectural case for intent-based monitoring
- [XBRL Compliance Tutorial](/docs/tutorials/xbrl-compliance) — similar dual memory architecture for SEC filings
- [Healthcare Payor-Provider](/docs/tutorials/healthcare-payor-provider) — claims fraud, network compliance, prior auth
- [API Reference](/docs/api-reference/overview) — full endpoint documentation
