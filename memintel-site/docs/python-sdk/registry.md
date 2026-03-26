---
id: registry
title: Registry
sidebar_label: Registry
---

# Registry

The registry is the source of truth for all definitions — Concepts, Primitives, Conditions, and Features. Definitions are immutable once registered. Updates require a new version. The registry enforces uniqueness at `(id, version)`.

---

## Register Definition

```
POST /registry/definitions
```

Stores a validated definition in the registry under its namespace. Returns HTTP 409 if the same `id+version` already exists. Definitions are immutable — never mutate an existing version.

### Request — RegisterRequest

| Field | Type | Required | Description |
|---|---|---|---|
| `definition` | dict | **Required** | A Concept, Primitive, or Condition. Should have passed `/definitions/validate`. |
| `namespace` | str | **Required** | `'personal'` \| `'team'` \| `'org'` \| `'global'`. Must match the id prefix. |
| `metadata` | dict | Optional | `{ description, tags, owner }`. |

### Response Codes

| Status | Description |
|---|---|
| **200** | `DefinitionResponse`: `{ id, version, status, concept_hash, semantic_hash }`. |
| **400** | Validation error. |
| **401** | Unauthorised. |
| **409** | Already exists at this `id + version`. Create a new version instead. |

---

## List Definitions

```
GET /registry/definitions
```

Returns a paginated list of definitions. Filter by type, namespace, or tags. For semantic free-text search, use `GET /registry/search`.

### Query Parameters

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | str | Optional | `'concept'` \| `'primitive'` \| `'condition'` \| `'feature'`. |
| `namespace` | str | Optional | `'personal'` \| `'team'` \| `'org'` \| `'global'`. |
| `tags` | str | Optional | Comma-separated tag filter. |
| `limit` | int | Optional | Default `20`, max `100`. |
| `cursor` | str | Optional | Pagination cursor. |

### Response Codes

| Status | Description |
|---|---|
| **200** | Paginated list with `items[]`, `has_more`, `next_cursor`, `total_count`. |
| **401** | Unauthorised. |

---

## Search Definitions

```
GET /registry/search
```

Free-text and structured semantic search across registry definitions. Searches `description`, `meaning`, `subject`, `intent`, `tags`, and `semantic_type` fields. Results ranked by relevance.

### Query Parameters

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | str | Optional | Free-text search. |
| `type` | str | Optional | `'concept'` \| `'primitive'` \| `'condition'` \| `'feature'`. |
| `namespace` | str | Optional | `'personal'` \| `'team'` \| `'org'` \| `'global'`. |
| `limit` | int | Optional | Default `20`, max `100`. |
| `cursor` | str | Optional | Pagination cursor. |

### Response Codes

| Status | Description |
|---|---|
| **200** | Paginated `SearchResult`. |
| **401** | Unauthorised. |

---

## List Definition Versions

```
GET /registry/definitions/{id}/versions
```

Returns version history for a definition, newest-first. Includes deprecation status per version. Use to discover the current version before constructing execute or evaluate calls.

### Response Codes

| Status | Description |
|---|---|
| **200** | `VersionListResult`: `{ id, versions: VersionSummary[] }`. |
| **401** | Unauthorised. |
| **404** | Not found. |

---

## Get Definition Lineage

```
GET /registry/definitions/{id}/lineage
```

Returns the full dependency graph of a definition — both what it depends on (`dependencies`) and what depends on it (`dependents`). Use for impact analysis before modifying shared definitions.

### Query Parameters

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | str | Required (path) | Fully qualified definition id. |
| `version` | str | Required (query) | Explicit version. |

### Response Codes

| Status | Description |
|---|---|
| **200** | `LineageResult`: `{ id, version, dependencies[], dependents[] }`. |
| **401** | Unauthorised. |
| **404** | Not found. |

---

## Semantic Diff

```
GET /registry/definitions/{id}/semantic-diff
```

Returns a meaning-level diff between two versions. Equivalence status drives governance decisions: `equivalent` = safe to promote; `compatible` = review recommended; `breaking` = governance required; `unknown` = treat as breaking.

### Query Parameters

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | str | Required (path) | Fully qualified definition id. |
| `version_from` | str | Required (query) | Baseline version. |
| `version_to` | str | Required (query) | Target version. |

### Response Codes

| Status | Description |
|---|---|
| **200** | `SemanticDiffResult`: `{ equivalence_status, changes, affected_downstream, summary }`. |
| **401** | Unauthorised. |
| **404** | One or both versions not found. |

---

## Deprecate Definition

```
POST /registry/definitions/{id}/deprecate
```

Marks a version as deprecated. Existing references continue to work. New registrations referencing this version receive a warning. Reversible by an admin.

### Request Body

| Field | Type | Required | Description |
|---|---|---|---|
| `version` | str | **Required** | Version to deprecate. |
| `replacement_version` | str | Optional | Recommended replacement. |
| `reason` | str | Optional | Human-readable reason. |

### Response Codes

| Status | Description |
|---|---|
| **200** | Deprecation confirmed. |
| **401** | Unauthorised. |
| **404** | Not found. |

---

## Promote Definition

```
POST /registry/definitions/{id}/promote
```

Promotes a definition to a higher namespace. Path: `personal → team → org → global`. Promoting to `global` requires elevated API key permissions. Run `semantic-diff` first to check for breaking changes.

### Request Body

| Field | Type | Required | Description |
|---|---|---|---|
| `version` | str | **Required** | Version to promote. |
| `from_namespace` | str | **Required** | `'personal'` \| `'team'` \| `'org'`. Source namespace. |
| `to_namespace` | str | **Required** | `'team'` \| `'org'` \| `'global'`. Target namespace. |

### Response Codes

| Status | Description |
|---|---|
| **200** | Promotion confirmed. |
| **401** | Unauthorised. |
| **403** | Insufficient permissions for target namespace. |
| **404** | Not found. |

---

## Register Feature

```
POST /registry/features
```

Registers a Feature as a first-class entity. The compiler computes `meaning_hash` on registration and scans for duplicates. `on_duplicate` controls behaviour on collision: `'warn'` (default) registers and lists duplicates, `'reject'` returns HTTP 409, `'merge'` returns the existing feature with the same `meaning_hash`.

### Response Codes

| Status | Description |
|---|---|
| **200** | Feature registered, or existing feature returned on `'merge'`. |
| **400** | Validation error. |
| **401** | Unauthorised. |
| **409** | Duplicate found and `on_duplicate='reject'`. |

---

## Get Feature Usages

```
GET /registry/features/{id}/usages
```

Returns every Concept that references this feature. Essential for impact analysis before deprecating or modifying a shared feature.

### Response Codes

| Status | Description |
|---|---|
| **200** | `{ feature_id, feature_version, used_by: [{ concept_id, concept_version, layer }] }`. |
| **401** | Unauthorised. |
| **404** | Feature not found. |
