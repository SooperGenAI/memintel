// ── Primitive types ───────────────────────────────────────────────────────────

export type MissingDataPolicy = 'null' | 'zero' | 'forward_fill' | 'backward_fill';
export type ExplainMode = 'summary' | 'full' | 'debug';
export type TaskStatus = 'active' | 'paused' | 'deleted' | 'preview';
export type FeedbackValue = 'false_positive' | 'false_negative' | 'correct';
export type ConditionStrategyType =
  | 'threshold'
  | 'percentile'
  | 'z_score'
  | 'change'
  | 'equals'
  | 'composite';
export type CalibrationStatus = 'recommendation_available' | 'no_recommendation';
export type NoRecommendationReason =
  | 'bounds_exceeded'
  | 'not_applicable_strategy'
  | 'insufficient_data';
export type ErrorType =
  | 'syntax_error'
  | 'type_error'
  | 'semantic_error'
  | 'reference_error'
  | 'parameter_error'
  | 'graph_error'
  | 'execution_error'
  | 'execution_timeout'
  | 'auth_error'
  | 'not_found'
  | 'conflict'
  | 'rate_limit_exceeded'
  | 'bounds_exceeded'
  | 'action_binding_failed';

// ── Config ────────────────────────────────────────────────────────────────────

export type DeliveryConfig = {
  type: 'webhook' | 'notification' | 'email' | 'workflow';
  endpoint?: string;
  channel?: string;
  workflowId?: string;
};

export type ConstraintsConfig = {
  sensitivity?: 'low' | 'medium' | 'high';
  namespace?: 'personal' | 'team' | 'org' | 'global';
};

// ── Task ──────────────────────────────────────────────────────────────────────

export type Task = {
  taskId?: string;
  intent: string;
  conceptId: string;
  conceptVersion: string;
  conditionId: string;
  conditionVersion: string;
  actionId: string;
  actionVersion: string;
  entityScope: string;
  delivery: DeliveryConfig;
  status: TaskStatus;
  createdAt?: string;
  lastTriggeredAt?: string | null;
};

export type TaskList = {
  items: Task[];
  hasMore: boolean;
  nextCursor: string | null;
  totalCount?: number;
};

// ── Execution ─────────────────────────────────────────────────────────────────

export type ActionTriggered = {
  actionId: string;
  actionVersion: string;
  status: 'triggered' | 'skipped' | 'failed' | 'would_trigger';
  payloadSent?: Record<string, unknown> | null;
  error?: Record<string, unknown> | null;
};

export type ExplanationNode = {
  nodeId: string;
  op: string;
  inputs: Record<string, unknown>;
  params: Record<string, unknown>;
  outputValue: number | boolean | string;
  outputType: string;
};

export type Explanation = {
  output: number | boolean | string;
  contributions: Record<string, number>;
  nodes: ExplanationNode[];
  trace: Array<Record<string, unknown>>;
};

export type Result = {
  value: number | boolean | string;
  type: 'float' | 'boolean' | 'categorical';
  entity: string;
  version: string;
  deterministic: boolean;
  timestamp?: string | null;
  explanation?: Explanation | null;
};

export type DecisionResult = {
  value: boolean | string;
  type: 'boolean' | 'categorical';
  entity: string;
  conditionId: string;
  conditionVersion: string;
  timestamp?: string | null;
  actionsTriggered: ActionTriggered[];
};

export type FullPipelineResult = {
  result: Result;
  decision: DecisionResult;
  dryRun?: boolean;
  entity: string;
  timestamp?: string;
};

export type BatchExecuteItem = {
  entity: string;
  result?: Result;
  error?: string;
};

export type BatchExecuteResult = {
  total: number;
  failed: number;
  items: BatchExecuteItem[];
};

export type Job = {
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  pollIntervalSeconds?: number;
};

export type JobResult = {
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  result?: Result;
  error?: ErrorResponse;
};

// ── Condition / Calibration ───────────────────────────────────────────────────

export type StrategyDefinition = {
  type: ConditionStrategyType;
  params: Record<string, unknown>;
};

export type ConditionDefinition = {
  conditionId: string;
  version: string;
  conceptId: string;
  conceptVersion: string;
  strategy: StrategyDefinition;
  namespace?: string;
  createdAt?: string;
  deprecated?: boolean;
};

export type ConditionExplanation = {
  conditionId: string;
  conditionVersion: string;
  strategy: StrategyDefinition;
  conceptId: string;
  conceptVersion: string;
  naturalLanguageSummary?: string;
  parameterRationale?: string;
};

export type CalibrationImpact = {
  deltaAlerts: number;
  direction: 'increase' | 'decrease' | 'no_change';
};

export type CalibrationResult = {
  status: CalibrationStatus;
  recommendedParams?: Record<string, unknown> | null;
  calibrationToken?: string | null;
  currentParams: Record<string, unknown>;
  impact?: CalibrationImpact | null;
  noRecommendationReason?: NoRecommendationReason | null;
};

export type ApplyCalibrationResult = {
  conditionId: string;
  previousVersion: string;
  newVersion: string;
  paramsApplied: Record<string, unknown>;
  tasksPendingRebind: Array<{ taskId: string; intent: string }>;
};

// ── Decisions / Feedback ──────────────────────────────────────────────────────

export type DecisionExplanationDriver = {
  signal: string;
  contribution: number;
  value: number | string | boolean;
};

export type DecisionExplanation = {
  conditionId: string;
  conditionVersion: string;
  entity: string;
  timestamp: string;
  decision: boolean | string;
  decisionType: 'boolean' | 'categorical';
  conceptValue: number | string;
  strategyType: ConditionStrategyType;
  thresholdApplied?: number | null;
  labelMatched?: string | null;
  drivers: DecisionExplanationDriver[];
};

export type FeedbackResponse = {
  status: 'recorded';
  feedbackId: string;
};

// ── Validation ────────────────────────────────────────────────────────────────

export type ValidationError = {
  type: string;
  message: string;
  location?: string | null;
  suggestion?: string | null;
};

export type ValidationResult = {
  valid: boolean;
  errors: ValidationError[];
  warnings: Array<{ type: string; message: string }>;
};

// ── Actions ───────────────────────────────────────────────────────────────────

export type Action = {
  actionId: string;
  actionVersion: string;
  name?: string;
  description?: string;
};

export type ActionResult = {
  status: 'triggered' | 'failed';
  payloadSent?: Record<string, unknown> | null;
  error?: string | null;
};

// ── Registry / Features ───────────────────────────────────────────────────────

export type SearchResult = {
  items: Array<Record<string, unknown>>;
  total?: number;
};

export type VersionListResult = {
  versions: Array<{ version: string; createdAt?: string; deprecated?: boolean }>;
};

export type FeatureSearchResult = {
  items: Array<Record<string, unknown>>;
  total?: number;
};

export type RegisteredFeature = {
  featureId: string;
  name?: string;
  description?: string;
  type?: string;
};

// ── Agents ────────────────────────────────────────────────────────────────────

export type AgentQueryResponse = {
  response: string;
  suggestions?: Array<Record<string, unknown>>;
};

export type AgentDefineResponse = {
  definition?: Record<string, unknown>;
  conceptId?: string;
  conditionId?: string;
  version?: string;
};

// ── Errors ────────────────────────────────────────────────────────────────────

export type ErrorResponse = {
  error: {
    type: ErrorType;
    message: string;
    location?: string | null;
    suggestion?: string | null;
  };
};

// ── Request param types ───────────────────────────────────────────────────────

export type MemintelConfig = {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
};

export type EvaluateFullParams = {
  conceptId: string;
  conceptVersion: string;
  conditionId: string;
  conditionVersion: string;
  entity: string;
  timestamp?: string;
  dryRun?: boolean;
  explain?: boolean;
  explainMode?: ExplainMode;
  missingDataPolicy?: MissingDataPolicy;
};

export type ExecuteParams = {
  id: string;
  version: string;
  entity: string;
  timestamp?: string;
  explain?: boolean;
  dryRun?: boolean;
  missingDataPolicy?: MissingDataPolicy;
};

export type EvaluateConditionParams = {
  conditionId: string;
  conditionVersion: string;
  entity: string;
  timestamp?: string;
  explain?: boolean;
};

export type EvaluateConditionBatchParams = {
  conditionId: string;
  conditionVersion: string;
  entities: string[];
  timestamp?: string;
};

export type ExecuteBatchParams = {
  id: string;
  version: string;
  entities: string[];
  timestamp?: string;
  explain?: boolean;
  dryRun?: boolean;
};

export type ExecuteRangeParams = {
  id: string;
  version: string;
  entity: string;
  fromTimestamp: string;
  toTimestamp: string;
  interval: string;
  explain?: boolean;
};

export type ExplainParams = {
  id: string;
  version: string;
  entity: string;
  timestamp?: string;
};

export type ValidateParams = {
  definition: Record<string, unknown>;
};

export type ConditionImpactParams = Record<string, unknown>;

export type CreateTaskParams = {
  intent: string;
  entityScope: string;
  delivery: DeliveryConfig;
  constraints?: ConstraintsConfig;
  dryRun?: boolean;
};

export type UpdateTaskParams = {
  conditionVersion?: string;
  delivery?: DeliveryConfig;
  entityScope?: string;
  status?: 'active' | 'paused';
};

export type CalibrateParams = {
  conditionId: string;
  conditionVersion: string;
  feedbackType?: FeedbackValue;
  feedbackDirection?: 'tighten' | 'relax';
  target?: { alertsPerDay: number };
  context?: { entity?: string; timestamp?: string };
};

export type ApplyCalibrationParams = {
  calibrationToken: string;
  newVersion?: string;
};

export type FeedbackParams = {
  conditionId: string;
  conditionVersion: string;
  entity: string;
  timestamp: string;
  feedbackType: FeedbackValue;
  note?: string;
};

export type DecisionExplainParams = {
  conditionId: string;
  conditionVersion: string;
  entity: string;
  timestamp: string;
};

export type AgentQueryParams = {
  query: string;
  context?: Record<string, unknown>;
};

export type AgentDefineParams = {
  intent: string;
  context?: Record<string, unknown>;
};

export type AgentDefineConditionParams = {
  intent: string;
  conceptId: string;
  conceptVersion: string;
  context?: Record<string, unknown>;
};

export type ActionTriggerParams = {
  entity: string;
  payload?: Record<string, unknown>;
};

export type DryRunResult = {
  concept?: Record<string, unknown>;
  condition?: ConditionDefinition;
  actionId?: string;
  actionVersion?: string;
  validation?: ValidationResult;
  wouldTrigger?: boolean | null;
};
