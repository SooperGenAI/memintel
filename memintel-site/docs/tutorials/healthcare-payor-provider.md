---
id: healthcare-payor-provider
title: Healthcare Payor-Provider Intelligence
sidebar_label: Healthcare Payor-Provider
---

# Tutorial: Healthcare Payor-Provider Intelligence

A walkthrough of three critical intelligence use cases in the payor-provider space — claims fraud detection, provider network compliance monitoring, and prior authorization management. Each is examined from both the payor and provider perspective, and each follows the same Memintel architecture with domain-specific primitives, concepts, and conditions.

:::note What you'll build
Three deterministic monitoring systems covering the most operationally and financially significant problems at the intersection of health insurance and healthcare delivery — with every decision auditable, reproducible, and defensible under CMS, OIG, and state regulatory scrutiny.
:::

---

## Why Healthcare Needs Memintel

The payor-provider relationship is one of the most complex in any industry. Payors must evaluate billions of claims annually against evolving clinical policies, regulatory requirements, and fraud patterns. Providers must navigate authorization requirements, billing rules, and value-based contract metrics that change continuously.

Both sides share the same fundamental problem: **decisions that are made correctly today may become incorrect tomorrow** — because the regulatory environment changed, because a provider's billing pattern shifted, because a clinical policy was updated, or because a new fraud typology was published by CMS or the OIG.

Current systems handle this badly. Claims adjudication is largely rule-based and reactive. Provider credentialing is calendar-driven rather than event-driven. Prior authorization is manual and inconsistent. The result is billions of dollars in improper payments, provider abrasion from inconsistent authorization decisions, and compliance exposure on both sides.

Memintel addresses this by separating the **discovery of clinical and compliance meaning** (which requires contextual, temporal, and cross-memory reasoning) from the **execution of payor-provider decisions** (which must be deterministic, consistent, and fully auditable).

---

## The Three Roles

| Role | Who they are | What they do |
|---|---|---|
| **Data Engineer** | Health IT / data team | Builds data pipelines, writes resolver functions, delivers typed primitives |
| **Admin** | Chief Medical Officer / VP Compliance | Maintains config files — primitives, guardrails — governs clinical policy thresholds |
| **User** | Claims analyst / network manager / care manager | Expresses intent via dashboard — creates, monitors, and manages tasks |

---

## Use Case 1 — Claims Fraud Detection

### The Problem

*Perspective: Payor*

Healthcare fraud costs the US healthcare system an estimated $300 billion annually. The most sophisticated fraud is not a single anomalous claim — it is a pattern of individually plausible claims that collectively reveal upcoding, unbundling, phantom billing, or referral kickback schemes. Current rules-based systems catch obvious violations but miss the subtle patterns that account for the majority of improper payments.

The challenge is contextual: a billing pattern that looks suspicious for a primary care physician is completely normal for an oncologist. A procedure billed 40 times a month by a rural solo practitioner is an anomaly. The same procedure billed 40 times a month by a high-volume urban specialist may be entirely appropriate. Context — specialty, geography, patient population, historical pattern — determines what is normal.

### The Architecture Boundary

```
Raw claims data  →  Signal Extraction  →  Primitives  →  Memintel
(claims, EHR,        (pattern analysis,    (typed,          (deterministic
 eligibility,         LLM on notes,         normalised)       evaluation)
 provider data)       peer benchmarking)
```

### Primitive Design

```yaml
# memintel_primitives_claims_fraud.yaml

primitives:

  # Provider billing patterns
  - id: provider.procedure_volume_30d
    type: time_series<int>
    source: claims_pipeline
    entity: provider_id
    description: Daily procedure count over last 30 days — enables z_score detection

  - id: provider.upcoding_signal
    type: float
    source: claims_pipeline
    entity: provider_id
    description: Ratio of high-complexity codes to total codes vs specialty peer median, 0-1

  - id: provider.unbundling_signal
    type: float
    source: claims_pipeline
    entity: provider_id
    description: Frequency of code combinations that suggest unbundling vs peers, 0-1

  - id: provider.same_day_billing_rate
    type: float
    source: claims_pipeline
    entity: provider_id
    description: Ratio of claims with multiple procedures same day vs specialty median

  - id: provider.peer_deviation_score
    type: float
    source: benchmarking_pipeline
    entity: provider_id
    description: Overall billing pattern deviation from specialty and geography peers, 0-1

  # Patient-level signals
  - id: patient.multi_provider_same_service_flag
    type: boolean
    source: claims_pipeline
    entity: patient_id
    description: True if same service billed by multiple providers in same period

  - id: patient.impossible_day_flag
    type: boolean
    source: claims_pipeline
    entity: patient_id
    description: True if more services billed than possible in one day

  # External regulatory signals
  - id: provider.oia_watchlist_flag
    type: boolean
    source: regulatory_pipeline
    entity: provider_id
    description: True if provider appears on OIG exclusion list or active investigation registry

  - id: cpt_code.fraud_typology_score
    type: float
    source: regulatory_pipeline
    entity: cpt_code_id
    description: CMS/OIG fraud risk score for this CPT code based on recent enforcement, 0-1

  - id: provider.license_active_flag
    type: boolean
    source: credentialing_pipeline
    entity: provider_id
    description: True if provider license is current and active in billing state
```

### Resolver Examples

```python
# resolvers/claims_fraud_resolvers.py

@registry.resolver("provider.upcoding_signal")
async def resolve_upcoding_signal(entity_id: str, timestamp: datetime) -> float:
    # Compare provider's code complexity distribution to specialty peers
    result = await db.execute("""
        WITH provider_dist AS (
            SELECT
                SUM(CASE WHEN complexity_level = 'high' THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(*), 0) AS high_pct
            FROM claims
            WHERE rendering_provider_id = $1
              AND service_date BETWEEN $2 - INTERVAL '90 days' AND $2
        ),
        peer_dist AS (
            SELECT AVG(
                CASE WHEN complexity_level = 'high' THEN 1 ELSE 0 END::float
            ) AS peer_high_pct
            FROM claims c
            JOIN providers p ON c.rendering_provider_id = p.id
            WHERE p.specialty_code = (
                SELECT specialty_code FROM providers WHERE id = $1
            )
            AND c.service_date BETWEEN $2 - INTERVAL '90 days' AND $2
        )
        SELECT LEAST(provider_dist.high_pct / NULLIF(peer_dist.peer_high_pct, 0), 2.0) / 2.0
        FROM provider_dist, peer_dist
    """, entity_id, timestamp)
    return float(result.scalar() or 0)


@registry.resolver("patient.impossible_day_flag")
async def resolve_impossible_day(entity_id: str, timestamp: datetime) -> bool:
    # Check if total service hours billed exceed 24 hours on any day
    result = await db.execute("""
        SELECT EXISTS (
            SELECT service_date, SUM(service_duration_minutes)
            FROM claims
            WHERE patient_id = $1
              AND service_date <= $2
              AND service_date > $2 - INTERVAL '90 days'
            GROUP BY service_date
            HAVING SUM(service_duration_minutes) > 1440
        )
    """, entity_id, timestamp)
    return bool(result.scalar())


@registry.resolver("cpt_code.fraud_typology_score")
async def resolve_fraud_typology_score(entity_id: str, timestamp: datetime) -> float:
    # entity_id is a CPT code e.g. "99215"
    # Returns OIG/CMS risk score as of the given timestamp
    result = await db.execute("""
        SELECT risk_score
        FROM cpt_fraud_risk_scores
        WHERE cpt_code = $1
          AND effective_date <= $2
        ORDER BY effective_date DESC
        LIMIT 1
    """, entity_id, timestamp)
    return float(result.scalar() or 0)
```

### Guardrails

```yaml
# memintel_guardrails_claims_fraud.yaml

parameter_priors:
  provider.upcoding_signal:
    low_severity:     { threshold: 0.60 }
    medium_severity:  { threshold: 0.75 }
    high_severity:    { threshold: 0.88 }

  provider.peer_deviation_score:
    low_severity:     { threshold: 0.55 }
    medium_severity:  { threshold: 0.70 }
    high_severity:    { threshold: 0.85 }

  provider.procedure_volume_30d:
    low_severity:     { z_score: 2.0 }
    medium_severity:  { z_score: 2.5 }
    high_severity:    { z_score: 3.0 }

bias_rules:
  monitor:      low_severity
  investigate:  medium_severity
  urgent:       high_severity
  prepayment:   high_severity
```

### User Creates Tasks

```typescript
// Claims integrity analyst creates monitoring tasks

// Task 1 — Provider upcoding pattern
const upcodingTask = await client.tasks.create({
    intent: "Alert me when a provider shows significant upcoding relative to specialty peers",
    entityScope: "all_active_providers",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/fraud-upcoding" }
});
// Resolves to: upcoding_signal > 0.75 (medium severity)

// Task 2 — Statistically anomalous procedure volume
const volumeTask = await client.tasks.create({
    intent: "Alert me when a provider's procedure volume is statistically unusual",
    entityScope: "all_active_providers",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/fraud-volume" }
});
// Resolves to: z_score strategy on procedure_volume_30d > 2.5 std deviations

// Task 3 — Impossible day billing
const impossibleDayTask = await client.tasks.create({
    intent: "Alert me immediately when impossible day billing is detected for any patient",
    entityScope: "all_active_patients",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/fraud-impossible-day" }
});
// Resolves to: impossible_day_flag = true (immediate — no threshold)
```

### What the Alert Looks Like

```
🚨 Claims Integrity Alert — Upcoding Pattern Detected

Provider:   Dr. Sarah Chen, MD (NPI: 1234567890)
Specialty:  Internal Medicine — Suburban Chicago
Period:     Q4 2023 (Oct–Dec)

Billing Pattern vs Specialty Peers:
  99215 (High complexity): 68% of E&M claims  (peer median: 31%)
  99214 (Mod complexity):  28% of E&M claims  (peer median: 52%)
  99213 (Low complexity):   4% of E&M claims  (peer median: 17%)

Upcoding signal:   0.79  (threshold: 0.75)
Peer deviation:    0.71  (threshold: 0.70)
OIG watchlist:     No
License active:    Yes

Estimated overpayment (90 days):  $47,200

→ Request medical records   → Prepayment review   → Refer to SIU
```

### The Fraud Detection Advantage

The `provider.upcoding_signal` primitive compares the provider's billing distribution against their **specialty and geography peer group** — not against a global average. An oncologist legitimately bills a high proportion of complex codes. Comparing them to a primary care median would produce false positives constantly. The peer-adjusted comparison is what makes this genuinely useful. The primitive captures this adjustment; Memintel evaluates it deterministically.

---

## Use Case 2 — Provider Network Compliance

### The Problem

*Perspective: Payor*

Health plans are contractually and regulatorily obligated to maintain networks of qualified, accessible providers. CMS, state insurance commissioners, and accreditation bodies like NCQA audit network adequacy — whether patients have sufficient access to in-network providers across specialties and geographies. Network compliance failures result in regulatory sanctions, member grievances, and CMS star rating impacts.

At the same time, individual provider compliance within the network is a continuous obligation — licenses expire, board certifications lapse, malpractice coverage lapses, credentialing requirements change. A provider whose credentials were current at contracting may no longer meet network participation requirements twelve months later.

Current systems are predominantly calendar-driven — providers are recredentialed every two years regardless of whether their circumstances have changed. Events that should trigger immediate review (a license suspension, a malpractice judgment, an OIG exclusion) are often discovered weeks or months after they occur.

### Primitive Design

```yaml
# memintel_primitives_network_compliance.yaml

primitives:

  # Provider credentialing status
  - id: provider.license_expiry_days
    type: int
    source: credentialing_pipeline
    entity: provider_id
    description: Days until primary state license expires — negative means expired

  - id: provider.board_certification_active
    type: boolean
    source: credentialing_pipeline
    entity: provider_id
    description: True if specialty board certification is current

  - id: provider.malpractice_coverage_active
    type: boolean
    source: credentialing_pipeline
    entity: provider_id
    description: True if malpractice insurance coverage is current and meets plan minimums

  - id: provider.oig_exclusion_flag
    type: boolean
    source: regulatory_pipeline
    entity: provider_id
    description: True if provider appears on current OIG exclusion list

  - id: provider.adverse_action_flag
    type: boolean
    source: npdb_pipeline
    entity: provider_id
    description: True if new adverse action reported to NPDB in last 90 days

  # Access and availability
  - id: provider.accepting_new_patients
    type: boolean
    source: directory_pipeline
    entity: provider_id
    description: True if provider is currently accepting new patients

  - id: provider.wait_time_days
    type: int?
    source: directory_pipeline
    entity: provider_id
    description: Estimated days to next available appointment — null if unknown

  - id: specialty.network_adequacy_ratio
    type: float
    source: network_analytics_pipeline
    entity: specialty_geographic_area_id
    description: Ratio of available in-network providers to CMS minimum standard for this specialty/area

  # Contract compliance
  - id: provider.quality_score_percentile
    type: float
    source: quality_pipeline
    entity: provider_id
    description: Provider quality score percentile within specialty — from HEDIS and claims data

  - id: provider.contract_utilization_rate
    type: float
    source: claims_pipeline
    entity: provider_id
    description: Actual utilization vs contracted minimum — below 1.0 means underutilized
```

### Resolver Examples

```python
# resolvers/network_compliance_resolvers.py

@registry.resolver("provider.license_expiry_days")
async def resolve_license_expiry(entity_id: str, timestamp: datetime) -> int:
    result = await db.execute("""
        SELECT EXTRACT(DAY FROM expiry_date - $2)::int
        FROM provider_licenses
        WHERE provider_id = $1
          AND license_type = 'PRIMARY_STATE'
          AND state = (SELECT primary_billing_state FROM providers WHERE id = $1)
        ORDER BY expiry_date DESC
        LIMIT 1
    """, entity_id, timestamp)
    return int(result.scalar() or -999)


@registry.resolver("provider.oig_exclusion_flag")
async def resolve_oig_exclusion(entity_id: str, timestamp: datetime) -> bool:
    # Check OIG exclusion list as of timestamp — not current state
    # OIG list is updated monthly — use the version current on that date
    result = await oig_api.check_exclusion(
        npi=entity_id,
        as_of_date=timestamp
    )
    return result.is_excluded


@registry.resolver("specialty.network_adequacy_ratio")
async def resolve_network_adequacy(entity_id: str, timestamp: datetime) -> float:
    # entity_id is a specialty_geographic_area composite key
    # e.g. "cardiology_chicago_north"
    specialty, geo_area = entity_id.split("_", 1)
    result = await db.execute("""
        SELECT
            COUNT(CASE WHEN p.accepting_new_patients AND l.expiry_date > $3 THEN 1 END)::float
            / NULLIF(r.cms_minimum_providers, 0)
        FROM providers p
        JOIN provider_licenses l ON p.id = l.provider_id
        JOIN network_contracts nc ON p.id = nc.provider_id
        JOIN cms_adequacy_requirements r ON r.specialty = $1 AND r.geo_area = $2
        WHERE p.specialty_code = $1
          AND p.geographic_area = $2
          AND nc.contract_end_date > $3
          AND nc.effective_date <= $3
    """, specialty, geo_area, timestamp)
    return float(result.scalar() or 0)
```

### User Creates Tasks

```typescript
// Network management team creates compliance monitoring tasks

// Task 1 — License expiry early warning
const licenseTask = await client.tasks.create({
    intent: "Alert me when a provider's license is approaching expiry",
    entityScope: "all_network_providers",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/network-license" }
});
// Resolves to: license_expiry_days < 60 (medium severity — 60 days notice)

// Task 2 — OIG exclusion — immediate
const oigTask = await client.tasks.create({
    intent: "Alert me immediately when any network provider is added to the OIG exclusion list",
    entityScope: "all_network_providers",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/network-oig",
                priority: "immediate" }
});
// Resolves to: oig_exclusion_flag = true
// Event-driven — fires on OIG list update, not on schedule

// Task 3 — Network adequacy below CMS standard
const adequacyTask = await client.tasks.create({
    intent: "Alert me when network adequacy falls below CMS minimum standard in any specialty and geography",
    entityScope: "all_specialty_geographic_areas",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/network-adequacy" }
});
// Resolves to: network_adequacy_ratio < 1.0 (below CMS minimum)

// Task 4 — Adverse action reported to NPDB
const npdbTask = await client.tasks.create({
    intent: "Alert me when a new adverse action is reported against a network provider",
    entityScope: "all_network_providers",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/network-npdb",
                priority: "high" }
});
// Resolves to: adverse_action_flag = true
```

### What the Alert Looks Like — License Expiry

```
⚠️ Network Compliance Alert — License Expiry Approaching

Provider:     Dr. James Park, MD (NPI: 9876543210)
Specialty:    Cardiology
Network:      PPO Gold, HMO Standard
Members:      847 attributed members

License Expiry:   42 days  (expires: 2024-04-28)
Board Cert:       Active  (expires: 2025-09-15)
Malpractice:      Active
OIG Status:       Clear
NPDB:             No adverse actions

Action required: Provider must submit renewed license
before 2024-04-28 or will be suspended from network

→ Send provider notice   → Update credentialing portal   → Flag for follow-up
```

### What the Alert Looks Like — OIG Exclusion

```
🔴 URGENT — Network Provider OIG Exclusion

Provider:     Dr. Michael Torres, MD (NPI: 5551234567)
Specialty:    Pain Management
Exclusion:    Added to OIG LEIE 2024-03-15 14:30 UTC
Reason:       Program-related conviction

Impact:
  • Federal programs: CANNOT bill Medicare/Medicaid effective immediately
  • Claims submitted after exclusion date: subject to repayment
  • Network contract: termination required under CMS conditions

Members affected:    312 attributed
Claims in flight:    23 claims submitted today — HOLD pending review

Action required IMMEDIATELY — federal law prohibits payment
to excluded providers from date of exclusion

→ Suspend provider NOW   → Hold in-flight claims   → Notify compliance
```

### The Network Compliance Advantage

The OIG exclusion alert demonstrates the event-driven advantage. Current credentialing systems check the OIG list on a monthly or quarterly schedule. A provider excluded on March 15th might not be flagged until the next monthly batch run — in the interim, the plan may continue paying claims to an excluded provider, creating federal overpayment liability. Memintel evaluates against the OIG list as of the current timestamp, event-driven on each update, eliminating this exposure gap entirely.

---

## Use Case 3 — Prior Authorization Management

### The Problem

*Perspective: Both Payor and Provider*

Prior authorization is one of the most significant sources of friction and financial risk in the payor-provider relationship. For payors, authorization decisions must be clinically appropriate, consistent, and compliant with CMS and state timeliness requirements. For providers, authorization denials are a leading cause of claim denials, delayed care, and administrative burden.

The problem has two sides:

**Payor side:** Authorization decisions must be made consistently — the same clinical scenario must receive the same determination regardless of which reviewer handles it. Current processes are highly variable and difficult to audit.

**Provider side:** Authorization status must be actively monitored — an authorization approved today may have conditions, expiry dates, or unit limits that, if exceeded, result in denial. Providers often discover authorization issues only at the point of claim submission, after the service has already been delivered.

### Primitive Design

```yaml
# memintel_primitives_auth.yaml

primitives:

  # Authorization request signals (Payor perspective)
  - id: auth_request.clinical_criteria_match_score
    type: float
    source: clinical_review_pipeline
    entity: auth_request_id
    description: Degree to which request matches InterQual/MCG clinical criteria, 0-1

  - id: auth_request.diagnosis_procedure_alignment
    type: float
    source: clinical_review_pipeline
    entity: auth_request_id
    description: Clinical alignment between diagnosis codes and requested procedure, 0-1

  - id: auth_request.member_history_support_score
    type: float
    source: claims_pipeline
    entity: auth_request_id
    description: Degree to which member's claims history supports the request, 0-1

  - id: auth_request.similar_request_approval_rate
    type: float
    source: authorization_pipeline
    entity: auth_request_id
    description: Historical approval rate for clinically similar requests, 0-1

  # Authorization status signals (Provider perspective)
  - id: auth.days_to_expiry
    type: int
    source: authorization_pipeline
    entity: auth_id
    description: Days until authorization expires — negative means expired

  - id: auth.units_remaining
    type: int?
    source: authorization_pipeline
    entity: auth_id
    description: Authorized units remaining — null if no unit limit

  - id: auth.units_utilization_rate
    type: float
    source: authorization_pipeline
    entity: auth_id
    description: Units used / units authorized — above 1.0 means overrun

  - id: auth.pending_claims_at_risk
    type: int
    source: claims_pipeline
    entity: auth_id
    description: Count of submitted claims that may be denied if auth expires or is exhausted

  # Timeliness compliance (Payor perspective)
  - id: auth_request.hours_pending
    type: int
    source: authorization_pipeline
    entity: auth_request_id
    description: Hours since authorization request was submitted

  - id: auth_request.urgency_level
    type: categorical
    source: authorization_pipeline
    entity: auth_request_id
    description: Request urgency — standard/urgent/emergent

  # Policy alignment
  - id: procedure.policy_change_flag
    type: boolean
    source: clinical_policy_pipeline
    entity: cpt_code_id
    description: True if authorization requirements for this procedure changed in last 30 days

  - id: procedure.cms_ncd_coverage_status
    type: categorical
    source: regulatory_pipeline
    entity: cpt_code_id
    description: CMS National Coverage Determination status — covered/non_covered/conditional
```

### Resolver Examples

```python
# resolvers/auth_resolvers.py

@registry.resolver("auth.days_to_expiry")
async def resolve_auth_expiry(entity_id: str, timestamp: datetime) -> int:
    result = await db.execute("""
        SELECT EXTRACT(DAY FROM expiry_date - $2)::int
        FROM authorizations
        WHERE auth_id = $1
    """, entity_id, timestamp)
    return int(result.scalar() or -999)


@registry.resolver("auth.units_utilization_rate")
async def resolve_units_utilization(entity_id: str, timestamp: datetime) -> float:
    result = await db.execute("""
        SELECT
            SUM(c.units_billed)::float / NULLIF(a.authorized_units, 0)
        FROM authorizations a
        LEFT JOIN claims c ON c.auth_id = a.auth_id
            AND c.service_date <= $2
        WHERE a.auth_id = $1
        GROUP BY a.authorized_units
    """, entity_id, timestamp)
    return float(result.scalar() or 0)


@registry.resolver("auth_request.hours_pending")
async def resolve_hours_pending(entity_id: str, timestamp: datetime) -> int:
    result = await db.execute("""
        SELECT EXTRACT(HOUR FROM $2 - submitted_at)::int
        FROM auth_requests
        WHERE auth_request_id = $1
          AND status = 'pending'
    """, entity_id, timestamp)
    return int(result.scalar() or 0)


@registry.resolver("auth_request.similar_request_approval_rate")
async def resolve_similar_approval_rate(entity_id: str, timestamp: datetime) -> float:
    # Look at historical approval rate for clinically similar requests
    result = await db.execute("""
        SELECT
            SUM(CASE WHEN ar2.status = 'approved' THEN 1 ELSE 0 END)::float
            / NULLIF(COUNT(*), 0)
        FROM auth_requests ar1
        JOIN auth_requests ar2
            ON ar2.primary_cpt_code = ar1.primary_cpt_code
            AND ar2.primary_diagnosis_code = ar1.primary_diagnosis_code
            AND ar2.submitted_at BETWEEN $2 - INTERVAL '12 months' AND $2
            AND ar2.auth_request_id != $1
        WHERE ar1.auth_request_id = $1
    """, entity_id, timestamp)
    return float(result.scalar() or 0.5)
```

### Guardrails

```yaml
# memintel_guardrails_auth.yaml

parameter_priors:
  auth.days_to_expiry:
    low_severity:     { threshold: 14 }   # 2 weeks notice
    medium_severity:  { threshold: 7  }   # 1 week notice
    high_severity:    { threshold: 2  }   # 48 hours — urgent

  auth.units_utilization_rate:
    low_severity:     { threshold: 0.75 }  # 75% of units used
    medium_severity:  { threshold: 0.90 }  # 90% of units used
    high_severity:    { threshold: 1.00 }  # fully exhausted

  auth_request.hours_pending:
    # CMS urgent care: 72 hours / Standard: 14 calendar days
    # Guardrails set per urgency category
    urgent_standard:   { threshold: 48  }  # alert at 48h (CMS limit: 72h)
    standard_standard: { threshold: 240 }  # alert at 10 days (CMS limit: 14 days)

bias_rules:
  urgent:     high_severity
  expiring:   high_severity
  approaching: medium_severity
  monitor:    low_severity
```

### User Creates Tasks — Payor Perspective

```typescript
// Authorization operations manager (payor) creates tasks

// Task 1 — Timeliness compliance — urgent requests
const urgentTimeliness = await client.tasks.create({
    intent: "Alert me when an urgent prior authorization request is approaching the CMS 72-hour timeliness limit",
    entityScope: "all_pending_urgent_auth_requests",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/auth-timeliness" }
});
// Resolves to: hours_pending > 48 AND urgency_level = urgent

// Task 2 — Policy change impact on pending auths
const policyChange = await client.tasks.create({
    intent: "Alert me when a clinical policy change affects procedures with pending authorizations",
    entityScope: "all_active_procedures",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/auth-policy-change" }
});
// Resolves to: procedure.policy_change_flag = true
// Event-driven — fires when policy is updated
```

### User Creates Tasks — Provider Perspective

```typescript
// Revenue cycle manager (provider) creates tasks

// Task 1 — Authorization expiry warning
const authExpiry = await client.tasks.create({
    intent: "Alert me when an active authorization is approaching expiry with pending services",
    entityScope: "all_active_authorizations",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/auth-expiry" }
});
// Resolves to: days_to_expiry < 7 AND pending_claims_at_risk > 0

// Task 2 — Unit exhaustion warning
const unitExhaustion = await client.tasks.create({
    intent: "Alert me when authorization units are nearly exhausted",
    entityScope: "all_active_authorizations",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/auth-units" }
});
// Resolves to: units_utilization_rate > 0.90

// Task 3 — Authorization required but missing
const missingAuth = await client.tasks.create({
    intent: "Alert me when a scheduled procedure requires authorization and none is on file",
    entityScope: "all_scheduled_procedures",
    delivery: { type: "webhook", endpoint: "https://myapp.com/hooks/auth-missing" }
});
```

### What the Alerts Look Like — Provider Side

```
⚠️ Authorization Alert — Units Nearly Exhausted

Patient:        Mary Johnson  (MRN: 4821039)
Authorization:  AUTH-2024-88823
Service:        Physical Therapy (CPT 97110)
Authorized:     24 units   |   Used: 22 units   |   Remaining: 2 units
Expiry:         2024-05-31  (67 days remaining)

Scheduled upcoming sessions: 4
Units at risk: 2 sessions will exceed authorization

Action required: Submit extension request before next session
→ Request extension   → Contact payor   → Update schedule

---

⚠️ Authorization Alert — Expiry with Claims at Risk

Patient:        Robert Chen  (MRN: 7723041)
Authorization:  AUTH-2024-44201
Service:        Home Health (G0151)
Expiry:         5 days  (expires: 2024-03-20)

Pending claims at risk: 3 claims ($4,200 estimated)
Last renewal submitted: Not yet submitted

→ Submit renewal NOW   → Contact case manager   → Review schedule
```

### The Prior Auth Advantage

The `auth.pending_claims_at_risk` primitive is what makes provider-side monitoring genuinely actionable. Most authorization management systems show you that an auth is expiring. This system shows you that an expiring auth has 3 pending claims worth $4,200 that will be denied if the auth is not renewed — quantifying the financial impact and creating urgency proportional to actual risk. The Concept layer computes this combination automatically from the expiry and pending claims primitives.

---

## The System Response Loop (All Three Use Cases)

```
Trigger: Schedule (hourly/daily) OR event-driven
  (new claim submitted, OIG list updated, auth status changed,
   policy updated, NPDB report received)
          ↓
For each entity in scope:
  Memintel calls resolvers with (entity_id, timestamp)
  Resolvers fetch point-in-time values:
    → Claims DB: billing patterns, procedure volumes
    → Credentialing system: license and certification status
    → OIG/NPDB APIs: exclusion and adverse action status
    → Authorization system: expiry dates, unit utilization
    → Clinical policy engine: criteria match scores
  Concept computation runs
  Condition evaluates
          ↓
  If condition fires:
    Alert delivered to analyst / care manager / revenue cycle team
    Decision logged with full audit trail
    Workflow action triggered (hold claims, send notice, escalate)
          ↓
  If not: Decision logged. No action.
          ↓
Next trigger
```

### Event-driven vs scheduled evaluation

| Use Case | Best trigger | Reason |
|---|---|---|
| Claims fraud | **Daily batch** for pattern analysis; **real-time** for impossible day flags | Patterns require accumulation; impossible day flags are immediate |
| Network compliance — OIG | **Event-driven** on OIG list publication | Federal liability begins at exclusion date, not detection date |
| Network compliance — license | **Daily** with **30/60/90 day** early warning | License expiry is gradual but non-negotiable |
| Prior auth — provider | **Daily** + **event-driven** on claim submission | Expiry is gradual; missing auth is immediate |
| Prior auth — payor timeliness | **Hourly** for urgent requests | CMS timeliness penalties are time-critical |

---

## Task Management

### User controls

```typescript
// View all active compliance monitoring tasks
const tasks = await client.tasks.list({ owner: "current_user" });

// Pause fraud detection during claims system migration
await client.tasks.pause("tsk_fraud_upcoding", {
    reason: "Claims system migration — data quality unreliable 2024-03-15 to 2024-03-17"
});

// Check alert performance
const taskDetail = await client.tasks.get("tsk_auth_expiry");
console.log(`Fired ${taskDetail.fire_count_30d} times in last 30 days`);
console.log(`Avg days lead time before auth expiry: ${taskDetail.avg_lead_time_days}`);
```

### Admin visibility

```typescript
// Admin view — all tasks across all departments
const allTasks = await adminClient.tasks.list();

// Compliance audit — which condition version was active on a specific date?
// Critical for CMS audit response
const versionHistory = await adminClient.conditions.versionHistory({
    conditionId: "auth_request.timeliness_breach_risk",
    from: "2024-01-01",
    to: "2024-03-31"
});
// Returns: which version was active on each date during the audit period

// Suspend tasks if a data source is unreliable
await adminClient.tasks.suspend("tsk_network_adequacy", {
    reason: "Provider directory API outage — adequacy data unreliable"
});
```

---

## Calibration

### Claims fraud — driven by SIU investigation outcomes

```typescript
// Provider was flagged but investigation found no fraud — false positive
await client.feedback.decision({
    conditionId: "provider.high_fraud_risk",
    conditionVersion: "1.1",
    entity: "provider_npi_1234567890",
    timestamp: "2024-01-15T00:00:00Z",
    feedback: "false_positive",
    note: "SIU investigation closed — provider serves complex patient population justifying high complexity codes"
});
```

### Network compliance — driven by regulatory audit outcomes

```typescript
// CMS audit found network adequacy gap that was not flagged — false negative
await client.feedback.decision({
    conditionId: "specialty_area.network_adequacy_insufficient",
    conditionVersion: "1.0",
    entity: "cardiology_rural_southwest",
    timestamp: "2023-12-31T00:00:00Z",
    feedback: "false_negative",
    note: "CMS audit found inadequacy — threshold too conservative for rural geographies"
});
```

### Prior auth — driven by denial rates and appeals outcomes

```typescript
// Auth was not flagged as at-risk but claim was denied — false negative
await client.feedback.decision({
    conditionId: "auth.expiry_at_risk",
    conditionVersion: "1.0",
    entity: "auth_2024_44201",
    timestamp: "2024-03-10T00:00:00Z",
    feedback: "false_negative",
    note: "Auth expired and claim denied — 7-day warning was insufficient for this service type"
});
// Calibration may recommend: extend warning window for home health services to 14 days
```

---

## Full Lifecycle Diagram

```
SETUP (one time)
──────────────────────────────────────────────────────────
Data Engineer:   Configure claims, credentialing, auth pipelines
Data Engineer:   Configure OIG/NPDB/CMS regulatory feeds
Data Engineer:   Write resolver functions (application code)
Admin:           Define primitives → memintel_primitives_[domain].yaml
Admin:           Define guardrails  → memintel_guardrails_[domain].yaml
System:          Load config at startup


TASK CREATION
──────────────────────────────────────────────────────────
User (analyst / care manager / rev cycle):
              Express intent via compliance dashboard or bot
Bot:          POST /tasks/create?dryRun=true
Memintel:     Compile intent → concept + condition
Bot:          Show plain-English preview + estimated alert volume
User:         Confirm
Bot:          POST /tasks/create → task activated


ONGOING EVALUATION LOOP
──────────────────────────────────────────────────────────
Trigger: Schedule OR event (claim submitted, OIG updated, auth changed)
  → Resolvers fetch point-in-time values from clinical systems
  → Concept computation runs (peer comparison, trend analysis, clinical criteria)
  → Condition evaluates
  → If fired: alert delivered, workflow triggered, decision logged
  → If not:   decision logged, no action


TASK MANAGEMENT
──────────────────────────────────────────────────────────
User:    View / pause / resume own tasks
Admin:   View all tasks, version history for audit response
Admin:   Suspend tasks during data source outages
Admin:   Deprecate condition versions after policy updates


CALIBRATION CYCLE
──────────────────────────────────────────────────────────
Ground truth: SIU outcome / CMS audit finding / denial appeal outcome
User:    Submit feedback (false positive / false negative / correct)
Admin:   Request calibration recommendation
Admin:   Review impact — sensitivity vs specificity tradeoff
Admin:   Approve → new condition version (immutable)
User:    Rebind task to new version (explicit, never automatic)
         ↓
Back to ONGOING EVALUATION LOOP with recalibrated thresholds
```

---

## Role Summary

| Step | Who | Claims Fraud | Network Compliance | Prior Auth |
|---|---|---|---|---|
| Data pipeline | **Data Eng.** | Claims feeds, NLP scoring | Credentialing, OIG/NPDB feeds | Auth system, clinical criteria |
| Resolvers | **Data Eng.** | Peer benchmarking, pattern computation | License/cert status, OIG API | Expiry, unit utilization, pending claims |
| Primitives | **Admin** | `memintel_primitives_claims_fraud.yaml` | `memintel_primitives_network_compliance.yaml` | `memintel_primitives_auth.yaml` |
| Guardrails | **Admin** | `memintel_guardrails_claims_fraud.yaml` | `memintel_guardrails_network_compliance.yaml` | `memintel_guardrails_auth.yaml` |
| Task creation | **User** | Claims integrity analyst | Network management team | Rev cycle mgr (provider) / Auth ops (payor) |
| Feedback | **User** | SIU investigation outcomes | CMS audit findings | Denial rates and appeal outcomes |
| Calibration | **Admin** | Approves threshold adjustments | Approves threshold adjustments | Approves threshold adjustments |

---

## Why This Architecture Fits Healthcare

All three use cases share the same structural property: **the compliance determination requires evaluating internal clinical and operational state against an evolving external regulatory and clinical environment.**

Claims fraud requires evaluating a provider's billing pattern against both their own history (internal) and their specialty peer group and current OIG typologies (external). A rule fires on a single transaction. Memintel detects a pattern.

Network compliance requires evaluating a provider's credentialing state against both their current credentials (internal) and the current OIG exclusion list, NPDB reports, and CMS adequacy standards (external). A calendar-based system recredentials every two years. Memintel detects the moment something changes.

Prior authorization requires evaluating authorization status against both current utilization (internal) and current clinical policy requirements, CMS timeliness rules, and procedure coverage determinations (external). A claims system discovers the problem at denial. Memintel surfaces it before the service is delivered.

The auditability requirement is especially significant in healthcare. CMS audits, OIG investigations, and state insurance commissioner reviews all require documentation of when decisions were made, on what basis, and whether processes were consistently followed. Memintel's immutable audit trail — logging every evaluation with its exact concept version, condition version, primitive values, and timestamp — satisfies this documentation requirement automatically, without requiring forensic reconstruction after the fact.

---

## Next Steps

- [Core Concepts](/docs/intro/core-concepts) — understand the ψ → φ → α model in depth
- [Financial Risk Monitoring](/docs/tutorials/financial-risk-monitoring) — AML, credit risk, capital adequacy
- [XBRL Compliance Tutorial](/docs/tutorials/xbrl-compliance) — SEC filing compliance
- [Guardrails System](/docs/intro/guardrails) — how admins configure the policy layer
- [API Reference](/docs/api-reference/overview) — full endpoint documentation
