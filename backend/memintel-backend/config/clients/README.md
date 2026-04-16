# Client Primitive Configs

Add one YAML file per client. Memintel loads all `*.yaml` files in this
directory on startup and merges their `primitives:` and `primitive_sources:`
blocks into the base config.

## Format

Each file must contain only these two top-level keys (both optional):

```yaml
primitives:
  - name: <namespace>.<field>
    type: <memintel-type>
    missing_data_policy: null | zero | forward_fill | backward_fill
    source:
      type: database | api | stream
      identifier: <connector-name>   # must exist in memintel_config.yaml connectors:
      field: <column-or-response-field>
      access:
        method: sql | rest
        query: >
          SELECT ... FROM ... WHERE id = :entity_id AND ts <= :as_of

primitive_sources:
  <namespace>.<field>:
    connector: <connector-name>      # must exist in memintel_config.yaml connectors:
    query: >
      SELECT ... FROM ... WHERE id = :entity_id AND ts <= :as_of
```

## Rules

- Connector names referenced here (`identifier:`, `connector:`) **must** be
  declared in `memintel_config.yaml` under `connectors:`. Client files cannot
  define new connectors — add those to the base config.
- Primitive names must be globally unique across all client files and the base
  config. Duplicates will cause a startup error.
- Files are loaded in alphabetical order. Last writer wins for
  `primitive_sources` key collisions.

## Current clients

| File              | Client       | Primitives                                      |
|-------------------|--------------|-------------------------------------------------|
| `acme_bank.yaml`  | Acme Bank    | `payment.*`, `customer.*` (credit metrics)      |
| `xbrl_client.yaml`| XBRL Client  | `filing.*` (XBRL filing quality metrics)        |
