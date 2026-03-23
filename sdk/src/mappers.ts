import {
  Action,
  ActionResult,
  ActionTriggered,
  AgentDefineResponse,
  AgentQueryResponse,
  ApplyCalibrationResult,
  BatchExecuteItem,
  BatchExecuteResult,
  CalibrationImpact,
  CalibrationResult,
  ConditionDefinition,
  ConditionExplanation,
  ConstraintsConfig,
  DecisionExplanation,
  DecisionExplanationDriver,
  DecisionResult,
  DeliveryConfig,
  DryRunResult,
  Explanation,
  ExplanationNode,
  FeedbackResponse,
  FeatureSearchResult,
  FullPipelineResult,
  Job,
  JobResult,
  RegisteredFeature,
  Result,
  SearchResult,
  StrategyDefinition,
  Task,
  TaskList,
  ValidationResult,
  VersionListResult,
} from './types';

type Raw = Record<string, unknown>;

// ── Delivery ──────────────────────────────────────────────────────────────────

export function mapDeliveryToSnake(d: DeliveryConfig): Raw {
  const out: Raw = { type: d.type };
  if (d.endpoint !== undefined) out.endpoint = d.endpoint;
  if (d.channel !== undefined) out.channel = d.channel;
  if (d.workflowId !== undefined) out.workflow_id = d.workflowId;
  return out;
}

function mapDeliveryFromSnake(raw: unknown): DeliveryConfig {
  const r = raw as Raw;
  const d: DeliveryConfig = { type: r.type as DeliveryConfig['type'] };
  if (r.endpoint !== undefined) d.endpoint = r.endpoint as string;
  if (r.channel !== undefined) d.channel = r.channel as string;
  if (r.workflow_id !== undefined) d.workflowId = r.workflow_id as string;
  return d;
}

export function mapConstraintsToSnake(c: ConstraintsConfig): Raw {
  const out: Raw = {};
  if (c.sensitivity !== undefined) out.sensitivity = c.sensitivity;
  if (c.namespace !== undefined) out.namespace = c.namespace;
  return out;
}

// ── Task ──────────────────────────────────────────────────────────────────────

export function mapTask(raw: unknown): Task {
  const r = raw as Raw;
  return {
    taskId:           r.task_id as string | undefined,
    intent:           r.intent as string,
    conceptId:        r.concept_id as string,
    conceptVersion:   r.concept_version as string,
    conditionId:      r.condition_id as string,
    conditionVersion: r.condition_version as string,
    actionId:         r.action_id as string,
    actionVersion:    r.action_version as string,
    entityScope:      r.entity_scope as string,
    delivery:         mapDeliveryFromSnake(r.delivery),
    status:           r.status as Task['status'],
    createdAt:        r.created_at as string | undefined,
    lastTriggeredAt:  r.last_triggered_at as string | null | undefined,
  };
}

export function mapTaskList(raw: unknown): TaskList {
  const r = raw as Raw;
  return {
    items:       (r.items as unknown[] ?? []).map(mapTask),
    hasMore:     r.has_more as boolean,
    nextCursor:  r.next_cursor as string | null,
    totalCount:  r.total_count as number | undefined,
  };
}

// ── Execution ─────────────────────────────────────────────────────────────────

function mapExplanationNode(raw: unknown): ExplanationNode {
  const r = raw as Raw;
  return {
    nodeId:      r.node_id as string,
    op:          r.op as string,
    inputs:      (r.inputs as Record<string, unknown>) ?? {},
    params:      (r.params as Record<string, unknown>) ?? {},
    outputValue: r.output_value as number | boolean | string,
    outputType:  r.output_type as string,
  };
}

function mapExplanation(raw: unknown): Explanation | null {
  if (!raw) return null;
  const r = raw as Raw;
  return {
    output:        r.output as number | boolean | string,
    contributions: (r.contributions as Record<string, number>) ?? {},
    nodes:         (r.nodes as unknown[] ?? []).map(mapExplanationNode),
    trace:         (r.trace as Array<Record<string, unknown>>) ?? [],
  };
}

function mapActionTriggered(raw: unknown): ActionTriggered {
  const r = raw as Raw;
  return {
    actionId:    r.action_id as string,
    actionVersion: r.action_version as string,
    status:      r.status as ActionTriggered['status'],
    payloadSent: r.payload_sent as Record<string, unknown> | null | undefined,
    error:       r.error as Record<string, unknown> | null | undefined,
  };
}

export function mapResult(raw: unknown): Result {
  const r = raw as Raw;
  const out: Result = {
    value:         r.value as number | boolean | string,
    type:          r.type as Result['type'],
    entity:        r.entity as string,
    version:       r.version as string,
    deterministic: r.deterministic as boolean,
    timestamp:     r.timestamp as string | null | undefined,
  };
  if ('explanation' in r) {
    out.explanation = r.explanation ? mapExplanation(r.explanation) : null;
  }
  return out;
}

export function mapDecisionResult(raw: unknown): DecisionResult {
  const r = raw as Raw;
  return {
    value:            r.value as boolean | string,
    type:             r.type as DecisionResult['type'],
    entity:           r.entity as string,
    conditionId:      r.condition_id as string,
    conditionVersion: r.condition_version as string,
    timestamp:        r.timestamp as string | null | undefined,
    actionsTriggered: (r.actions_triggered as unknown[] ?? []).map(mapActionTriggered),
  };
}

export function mapFullPipelineResult(raw: unknown): FullPipelineResult {
  const r = raw as Raw;
  return {
    result:    mapResult(r.result),
    decision:  mapDecisionResult(r.decision),
    dryRun:    r.dry_run as boolean | undefined,
    entity:    r.entity as string,
    timestamp: r.timestamp as string | undefined,
  };
}

export function mapBatchExecuteResult(raw: unknown): BatchExecuteResult {
  const r = raw as Raw;
  return {
    total:  r.total as number,
    failed: r.failed as number,
    items:  (r.items as unknown[] ?? []).map((item) => {
      const i = item as Raw;
      const out: BatchExecuteItem = { entity: i.entity as string };
      if (i.result) out.result = mapResult(i.result);
      if (i.error) out.error = i.error as string;
      return out;
    }),
  };
}

export function mapJob(raw: unknown): Job {
  const r = raw as Raw;
  return {
    jobId:               r.job_id as string,
    status:              r.status as Job['status'],
    pollIntervalSeconds: r.poll_interval_seconds as number | undefined,
  };
}

export function mapJobResult(raw: unknown): JobResult {
  const r = raw as Raw;
  const out: JobResult = {
    jobId:  r.job_id as string,
    status: r.status as JobResult['status'],
  };
  if (r.result) out.result = mapResult(r.result);
  if (r.error) out.error = r.error as JobResult['error'];
  return out;
}

// ── Condition ─────────────────────────────────────────────────────────────────

function mapStrategyDefinition(raw: unknown): StrategyDefinition {
  const r = raw as Raw;
  return {
    type:   r.type as StrategyDefinition['type'],
    params: (r.params as Record<string, unknown>) ?? {},
  };
}

export function mapConditionDefinition(raw: unknown): ConditionDefinition {
  const r = raw as Raw;
  return {
    conditionId:   r.condition_id as string,
    version:       r.version as string,
    conceptId:     r.concept_id as string,
    conceptVersion: r.concept_version as string,
    strategy:      mapStrategyDefinition(r.strategy),
    namespace:     r.namespace as string | undefined,
    createdAt:     r.created_at as string | undefined,
    deprecated:    r.deprecated as boolean | undefined,
  };
}

export function mapConditionExplanation(raw: unknown): ConditionExplanation {
  const r = raw as Raw;
  return {
    conditionId:           r.condition_id as string,
    conditionVersion:      r.condition_version as string,
    strategy:              mapStrategyDefinition(r.strategy),
    conceptId:             r.concept_id as string,
    conceptVersion:        r.concept_version as string,
    naturalLanguageSummary: r.natural_language_summary as string | undefined,
    parameterRationale:    r.parameter_rationale as string | undefined,
  };
}

function mapCalibrationImpact(raw: unknown): CalibrationImpact {
  const r = raw as Raw;
  return {
    deltaAlerts: r.delta_alerts as number,
    direction:   r.direction as CalibrationImpact['direction'],
  };
}

export function mapCalibrationResult(raw: unknown): CalibrationResult {
  const r = raw as Raw;
  return {
    status:                 r.status as CalibrationResult['status'],
    recommendedParams:      r.recommended_params as Record<string, unknown> | null | undefined,
    calibrationToken:       r.calibration_token as string | null | undefined,
    currentParams:          (r.current_params as Record<string, unknown>) ?? {},
    impact:                 r.impact ? mapCalibrationImpact(r.impact) : undefined,
    noRecommendationReason: r.no_recommendation_reason as CalibrationResult['noRecommendationReason'],
  };
}

export function mapApplyCalibrationResult(raw: unknown): ApplyCalibrationResult {
  const r = raw as Raw;
  return {
    conditionId:        r.condition_id as string,
    previousVersion:    r.previous_version as string,
    newVersion:         r.new_version as string,
    paramsApplied:      (r.params_applied as Record<string, unknown>) ?? {},
    tasksPendingRebind: (r.tasks_pending_rebind as unknown[] ?? []).map((t) => {
      const item = t as Raw;
      return { taskId: item.task_id as string, intent: item.intent as string };
    }),
  };
}

// ── Decisions / Feedback ──────────────────────────────────────────────────────

export function mapDecisionExplanation(raw: unknown): DecisionExplanation {
  const r = raw as Raw;
  return {
    conditionId:      r.condition_id as string,
    conditionVersion: r.condition_version as string,
    entity:           r.entity as string,
    timestamp:        r.timestamp as string,
    decision:         r.decision as boolean | string,
    decisionType:     r.decision_type as DecisionExplanation['decisionType'],
    conceptValue:     r.concept_value as number | string,
    strategyType:     r.strategy_type as DecisionExplanation['strategyType'],
    thresholdApplied: r.threshold_applied as number | null | undefined,
    labelMatched:     r.label_matched as string | null | undefined,
    drivers: (r.drivers as unknown[] ?? []).map((d): DecisionExplanationDriver => {
      const driver = d as Raw;
      return {
        signal:       driver.signal as string,
        contribution: driver.contribution as number,
        value:        driver.value as number | string | boolean,
      };
    }),
  };
}

export function mapFeedbackResponse(raw: unknown): FeedbackResponse {
  const r = raw as Raw;
  return {
    status:     r.status as 'recorded',
    feedbackId: r.feedback_id as string,
  };
}

// ── Validation ────────────────────────────────────────────────────────────────

export function mapValidationResult(raw: unknown): ValidationResult {
  const r = raw as Raw;
  return {
    valid:    r.valid as boolean,
    errors:   (r.errors as unknown[] ?? []).map((e) => {
      const err = e as Raw;
      return {
        type:       err.type as string,
        message:    err.message as string,
        location:   err.location as string | null | undefined,
        suggestion: err.suggestion as string | null | undefined,
      };
    }),
    warnings: (r.warnings as Array<{ type: string; message: string }>) ?? [],
  };
}

// ── Actions ───────────────────────────────────────────────────────────────────

export function mapAction(raw: unknown): Action {
  const r = raw as Raw;
  return {
    actionId:      r.action_id as string,
    actionVersion: r.action_version as string,
    name:          r.name as string | undefined,
    description:   r.description as string | undefined,
  };
}

export function mapActionResult(raw: unknown): ActionResult {
  const r = raw as Raw;
  return {
    status:      r.status as ActionResult['status'],
    payloadSent: r.payload_sent as Record<string, unknown> | null | undefined,
    error:       r.error as string | null | undefined,
  };
}

// ── Registry ──────────────────────────────────────────────────────────────────

export function mapSearchResult(raw: unknown): SearchResult {
  const r = raw as Raw;
  return {
    items: (r.items as Array<Record<string, unknown>>) ?? [],
    total: r.total as number | undefined,
  };
}

export function mapVersionListResult(raw: unknown): VersionListResult {
  const r = raw as Raw;
  return {
    versions: (r.versions as Array<{ version: string; created_at?: string; deprecated?: boolean }> ?? []).map((v) => ({
      version:    v.version,
      createdAt:  v.created_at,
      deprecated: v.deprecated,
    })),
  };
}

// ── Features ──────────────────────────────────────────────────────────────────

export function mapFeatureSearchResult(raw: unknown): FeatureSearchResult {
  const r = raw as Raw;
  return {
    items: (r.items as Array<Record<string, unknown>>) ?? [],
    total: r.total as number | undefined,
  };
}

export function mapRegisteredFeature(raw: unknown): RegisteredFeature {
  const r = raw as Raw;
  return {
    featureId:   r.feature_id as string,
    name:        r.name as string | undefined,
    description: r.description as string | undefined,
    type:        r.type as string | undefined,
  };
}

// ── Agents ────────────────────────────────────────────────────────────────────

export function mapAgentQueryResponse(raw: unknown): AgentQueryResponse {
  const r = raw as Raw;
  return {
    response:    r.response as string,
    suggestions: r.suggestions as Array<Record<string, unknown>> | undefined,
  };
}

export function mapAgentDefineResponse(raw: unknown): AgentDefineResponse {
  const r = raw as Raw;
  return {
    definition:  r.definition as Record<string, unknown> | undefined,
    conceptId:   r.concept_id as string | undefined,
    conditionId: r.condition_id as string | undefined,
    version:     r.version as string | undefined,
  };
}

// ── DryRunResult ──────────────────────────────────────────────────────────────

export function mapDryRunResult(raw: unknown): DryRunResult {
  const r = raw as Raw;
  const out: DryRunResult = {};
  if (r.concept !== undefined) out.concept = r.concept as Record<string, unknown>;
  if (r.condition !== undefined) out.condition = mapConditionDefinition(r.condition);
  if (r.action_id !== undefined) out.actionId = r.action_id as string;
  if (r.action_version !== undefined) out.actionVersion = r.action_version as string;
  if (r.validation !== undefined) out.validation = mapValidationResult(r.validation);
  if (r.would_trigger !== undefined) out.wouldTrigger = r.would_trigger as boolean | null;
  return out;
}
