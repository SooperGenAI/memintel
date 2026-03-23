"""
app/models/__init__.py
──────────────────────────────────────────────────────────────────────────────
Re-exports every public name from the models layer.

Import from this package rather than individual modules:
    from app.models import Task, ConditionDefinition, ErrorType
    from app.models import MemintelError, NotFoundError
"""

# ── errors ────────────────────────────────────────────────────────────────────
from app.models.errors import (
    ErrorType,
    ErrorDetail,
    ErrorResponse,
    ValidationErrorItem,
    MemintelError,
    NotFoundError,
    ConflictError,
    ValidationError,
    CompilerInvariantError,
    AuthError,
    RateLimitError,
    BoundsExceededError,
    ExecutionTimeoutError,
    http_status_for,
    memintel_error_handler,
)

# ── task ──────────────────────────────────────────────────────────────────────
from app.models.task import (
    DeliveryType,
    TaskStatus,
    MutableTaskStatus,
    Namespace,
    Sensitivity,
    IMMUTABLE_TASK_FIELDS,
    DeliveryConfig,
    ConstraintsConfig,
    Task,
    CreateTaskRequest,
    TaskUpdateRequest,
    TaskList,
)

# ── condition ─────────────────────────────────────────────────────────────────
from app.models.condition import (
    StrategyType,
    DecisionType,
    ThresholdDirection,
    PercentileDirection,
    ZScoreDirection,
    ChangeDirection,
    CompositeOperator,
    ThresholdParams,
    PercentileParams,
    ZScoreParams,
    ChangeParams,
    EqualsParams,
    CompositeParams,
    ThresholdStrategy,
    PercentileStrategy,
    ZScoreStrategy,
    ChangeStrategy,
    EqualsStrategy,
    CompositeStrategy,
    StrategyDefinition,
    BOOLEAN_STRATEGIES,
    CATEGORICAL_STRATEGIES,
    TYPE_STRATEGY_COMPATIBILITY,
    ConditionDefinition,
    DecisionValue,
    ConditionExplanation,
    DriverContribution,
    DecisionExplanation,
)

# ── result ────────────────────────────────────────────────────────────────────
from app.models.result import (
    ConceptOutputType,
    ExplainMode,
    MissingDataPolicy,
    ActionTriggeredStatus,
    JobStatus,
    TERMINAL_JOB_STATUSES,
    VALID_JOB_TRANSITIONS,
    ExecuteRequest,
    ExecuteGraphRequest,
    NodeTrace,
    ConceptExplanation,
    ConceptResult,
    ActionTriggered,
    DecisionResult,
    FullPipelineResult,
    ValidationWarning,
    ValidationResult,
    DryRunResult,
    Job,
    JobResult,
    BatchExecuteItem,
    BatchExecuteResult,
)

# ── concept ───────────────────────────────────────────────────────────────────
from app.models.concept import (
    MemintelType,
    PrimitiveRef,
    FeatureNode,
    ConceptDefinition,
    GraphNode,
    GraphEdge,
    ExecutionGraph,
    SemanticGraph,
    ExecutionPlan,
    DefinitionResponse,
    VersionSummary,
    SearchResult,
    SemanticDiffResult,
    LineageResult,
)

# ── action ────────────────────────────────────────────────────────────────────
from app.models.action import (
    ActionType,
    FireOn,
    WebhookActionConfig,
    NotificationActionConfig,
    WorkflowActionConfig,
    RegisterActionConfig,
    ActionConfig,
    TriggerConfig,
    ActionDefinition,
    ActionTriggerRequest,
    ActionResult,
    ActionList,
)

# ── calibration ───────────────────────────────────────────────────────────────
from app.models.calibration import (
    MIN_FEEDBACK_THRESHOLD,
    ImpactDirection,
    FeedbackValue,
    CalibrationStatus,
    NoRecommendationReason,
    FeedbackRequest,
    FeedbackRecord,
    FeedbackResponse,
    CalibrationToken,
    TargetConfig,
    CalibrationImpact,
    CalibrateRequest,
    CalibrationResult,
    TaskPendingRebind,
    ApplyCalibrationRequest,
    ApplyCalibrationResult,
    CalibrationTelemetry,
)

# ── config ────────────────────────────────────────────────────────────────────
from app.models.config import (
    ENV_VAR_PATTERN,
    VALID_PRIMITIVE_TYPES,
    VALID_LLM_PROVIDERS,
    AccessConfig,
    SourceConfig,
    PrimitiveConfig,
    ConnectorConfig,
    LLMConfig,
    RateLimitConfig,
    ExecutionConfig,
    EnvironmentConfig,
    ConfigSchema,
    ApplicationContext,
    ApplyResult,
    PrimitiveValue,
)

# ── guardrails ────────────────────────────────────────────────────────────────
from app.models.guardrails import (
    StrategyParameterConstraints,
    StrategyParameter,
    StrategyRegistryEntry,
    TypeCompatibilityEntry,
    SeverityLevel,
    SeverityResolutionRule,
    SeverityVocabulary,
    PrimitiveStrategyHints,
    PrimitiveHint,
    StrategyPreferences,
    SeverityPriors,
    MappingCondition,
    MappingRule,
    ThresholdBounds,
    MaxComplexity,
    StrategyVersionPolicy,
    GuardrailConstraints,
    StrategyPriorities,
    BiasEffect,
    ParameterBiasRule,
    StrategyBiasSemantics,
    ConflictResolutionRule,
    ConflictResolution,
    BiasApplicationRuleDetail,
    BiasApplicationRuleEntry,
    Guardrails,
)

# ── llm ───────────────────────────────────────────────────────────────────────
from app.models.llm import (
    MAX_CONTEXT_CHARS,
    SYSTEM_PROMPT,
    CONCEPT_GENERATION_PROMPT,
    CONDITION_GENERATION_PROMPT,
    REFINEMENT_PROMPT,
    LLMFeatureRef,
    LLMComputeSpec,
    LLMConceptOutput,
    LLMConditionOutput,
    LLMActionOutput,
    LLMTaskOutput,
    ParseError,
    LLMCallError,
    LLMAuthError,
    LLMGenerationError,
    AgentQueryRequest,
    AgentQueryResponse,
    AgentDefineRequest,
    AgentDefineResponse,
    SemanticRefineRequest,
    SemanticRefineResponse,
)

__all__ = [
    # errors
    "ErrorType", "ErrorDetail", "ErrorResponse", "ValidationErrorItem",
    "MemintelError", "NotFoundError", "ConflictError", "ValidationError",
    "CompilerInvariantError", "AuthError", "RateLimitError",
    "BoundsExceededError", "ExecutionTimeoutError",
    "http_status_for", "memintel_error_handler",
    # task
    "DeliveryType", "TaskStatus", "MutableTaskStatus", "Namespace",
    "Sensitivity", "IMMUTABLE_TASK_FIELDS",
    "DeliveryConfig", "ConstraintsConfig", "Task",
    "CreateTaskRequest", "TaskUpdateRequest", "TaskList",
    # condition
    "StrategyType", "DecisionType",
    "ThresholdDirection", "PercentileDirection", "ZScoreDirection",
    "ChangeDirection", "CompositeOperator",
    "ThresholdParams", "PercentileParams", "ZScoreParams",
    "ChangeParams", "EqualsParams", "CompositeParams",
    "ThresholdStrategy", "PercentileStrategy", "ZScoreStrategy",
    "ChangeStrategy", "EqualsStrategy", "CompositeStrategy",
    "StrategyDefinition",
    "BOOLEAN_STRATEGIES", "CATEGORICAL_STRATEGIES", "TYPE_STRATEGY_COMPATIBILITY",
    "ConditionDefinition", "DecisionValue",
    "ConditionExplanation", "DriverContribution", "DecisionExplanation",
    # result
    "ConceptOutputType", "ExplainMode", "MissingDataPolicy",
    "ActionTriggeredStatus", "JobStatus",
    "TERMINAL_JOB_STATUSES", "VALID_JOB_TRANSITIONS",
    "ExecuteRequest", "ExecuteGraphRequest",
    "NodeTrace", "ConceptExplanation", "ConceptResult",
    "ActionTriggered", "DecisionResult", "FullPipelineResult",
    "ValidationWarning", "ValidationResult", "DryRunResult",
    "Job", "JobResult", "BatchExecuteItem", "BatchExecuteResult",
    # concept
    "MemintelType", "PrimitiveRef", "FeatureNode", "ConceptDefinition",
    "GraphNode", "GraphEdge", "ExecutionGraph",
    "SemanticGraph", "ExecutionPlan",
    "DefinitionResponse", "VersionSummary", "SearchResult",
    "SemanticDiffResult", "LineageResult",
    # action
    "ActionType", "FireOn",
    "WebhookActionConfig", "NotificationActionConfig",
    "WorkflowActionConfig", "RegisterActionConfig", "ActionConfig",
    "TriggerConfig", "ActionDefinition",
    "ActionTriggerRequest", "ActionResult", "ActionList",
    # calibration
    "MIN_FEEDBACK_THRESHOLD",
    "ImpactDirection", "FeedbackValue", "CalibrationStatus", "NoRecommendationReason",
    "FeedbackRequest", "FeedbackRecord", "FeedbackResponse", "CalibrationToken",
    "TargetConfig", "CalibrationImpact",
    "CalibrateRequest", "CalibrationResult",
    "TaskPendingRebind", "ApplyCalibrationRequest", "ApplyCalibrationResult",
    "CalibrationTelemetry",
    # config
    "ENV_VAR_PATTERN", "VALID_PRIMITIVE_TYPES", "VALID_LLM_PROVIDERS",
    "AccessConfig", "SourceConfig", "PrimitiveConfig", "ConnectorConfig",
    "LLMConfig", "RateLimitConfig", "ExecutionConfig", "EnvironmentConfig",
    "ConfigSchema", "ApplicationContext", "ApplyResult", "PrimitiveValue",
    # guardrails
    "StrategyParameterConstraints", "StrategyParameter", "StrategyRegistryEntry",
    "TypeCompatibilityEntry",
    "SeverityLevel", "SeverityResolutionRule", "SeverityVocabulary",
    "PrimitiveStrategyHints", "PrimitiveHint",
    "StrategyPreferences", "SeverityPriors",
    "MappingCondition", "MappingRule",
    "ThresholdBounds", "MaxComplexity", "StrategyVersionPolicy",
    "GuardrailConstraints", "StrategyPriorities",
    "BiasEffect", "ParameterBiasRule", "StrategyBiasSemantics",
    "ConflictResolutionRule", "ConflictResolution",
    "BiasApplicationRuleDetail", "BiasApplicationRuleEntry",
    "Guardrails",
    # llm
    "MAX_CONTEXT_CHARS",
    "SYSTEM_PROMPT", "CONCEPT_GENERATION_PROMPT",
    "CONDITION_GENERATION_PROMPT", "REFINEMENT_PROMPT",
    "LLMFeatureRef", "LLMComputeSpec",
    "LLMConceptOutput", "LLMConditionOutput", "LLMActionOutput", "LLMTaskOutput",
    "ParseError", "LLMCallError", "LLMAuthError", "LLMGenerationError",
    "AgentQueryRequest", "AgentQueryResponse",
    "AgentDefineRequest", "AgentDefineResponse",
    "SemanticRefineRequest", "SemanticRefineResponse",
]
