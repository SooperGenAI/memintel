import { MemintelError } from './error';
import { HttpClient } from './http';
import {
  mapAgentQueryResponse,
  mapBatchExecuteResult,
  mapDecisionResult,
  mapFullPipelineResult,
  mapJob,
  mapJobResult,
  mapResult,
  mapValidationResult,
} from './mappers';
import {
  BatchExecuteResult,
  ConditionImpactParams,
  DecisionResult,
  EvaluateConditionBatchParams,
  EvaluateConditionParams,
  EvaluateFullParams,
  ExecuteBatchParams,
  ExecuteParams,
  ExecuteRangeParams,
  ExplainParams,
  FullPipelineResult,
  Job,
  JobResult,
  MemintelConfig,
  Result,
  ValidateParams,
  ValidationResult,
} from './types';
import { ActionsClient } from './clients/actions';
import { AgentsClient } from './clients/agents';
import { ConditionsClient } from './clients/conditions';
import { DecisionsClient } from './clients/decisions';
import { FeaturesClient } from './clients/features';
import { FeedbackClient } from './clients/feedback';
import { RegistryClient } from './clients/registry';
import { TasksClient } from './clients/tasks';

export { MemintelError } from './error';
export * from './types';

export class Memintel {
  private http: HttpClient;

  // ── Sub-clients ──────────────────────────────────────────────────────────────
  readonly tasks: TasksClient;
  readonly conditions: ConditionsClient;
  readonly decisions: DecisionsClient;
  readonly feedback: FeedbackClient;
  readonly registry: RegistryClient;
  readonly features: FeaturesClient;
  readonly actions: ActionsClient;
  readonly agents: AgentsClient;

  constructor(config: MemintelConfig) {
    if (!config.apiKey) {
      throw new MemintelError('auth_error', 'apiKey is required');
    }
    const baseUrl = config.baseUrl ?? 'https://api.memsdl.ai/v1';
    const timeout = config.timeout ?? 30_000;

    this.http = new HttpClient(baseUrl, config.apiKey, timeout);

    this.tasks      = new TasksClient(this.http);
    this.conditions = new ConditionsClient(this.http);
    this.decisions  = new DecisionsClient(this.http);
    this.feedback   = new FeedbackClient(this.http);
    this.registry   = new RegistryClient(this.http);
    this.features   = new FeaturesClient(this.http);
    this.actions    = new ActionsClient(this.http);
    this.agents     = new AgentsClient(this.http);
  }

  // ── evaluateFull ─────────────────────────────────────────────────────────────

  async evaluateFull(params: EvaluateFullParams): Promise<FullPipelineResult> {
    const { conceptId, conceptVersion, conditionId, conditionVersion, entity } = params;
    if (!conceptId || !conceptVersion || !conditionId || !conditionVersion || !entity) {
      throw new MemintelError(
        'parameter_error',
        'conceptId, conceptVersion, conditionId, conditionVersion, and entity are all required',
      );
    }

    const body: Record<string, unknown> = {
      concept_id:        conceptId,
      concept_version:   conceptVersion,
      condition_id:      conditionId,
      condition_version: conditionVersion,
      entity,
    };
    if (params.timestamp !== undefined)           body.timestamp            = params.timestamp;
    if (params.dryRun !== undefined)              body.dry_run              = params.dryRun;
    if (params.explain !== undefined)             body.explain              = params.explain;
    if (params.explainMode !== undefined)         body.explain_mode         = params.explainMode;
    if (params.missingDataPolicy !== undefined)   body.missing_data_policy  = params.missingDataPolicy;

    const raw = await this.http.post('/evaluate/full', body);
    return mapFullPipelineResult(raw);
  }

  // ── execute ──────────────────────────────────────────────────────────────────

  async execute(params: ExecuteParams): Promise<Result> {
    if (!params.id || !params.version || !params.entity) {
      throw new MemintelError('parameter_error', 'id, version, and entity are required');
    }
    const body: Record<string, unknown> = {
      id:      params.id,
      version: params.version,
      entity:  params.entity,
    };
    if (params.timestamp !== undefined)          body.timestamp            = params.timestamp;
    if (params.explain !== undefined)            body.explain              = params.explain;
    if (params.dryRun !== undefined)             body.dry_run              = params.dryRun;
    if (params.missingDataPolicy !== undefined)  body.missing_data_policy  = params.missingDataPolicy;

    const raw = await this.http.post('/execute', body);
    return mapResult(raw);
  }

  // ── evaluateCondition ─────────────────────────────────────────────────────────

  async evaluateCondition(params: EvaluateConditionParams): Promise<DecisionResult> {
    if (!params.conditionId || !params.conditionVersion || !params.entity) {
      throw new MemintelError('parameter_error', 'conditionId, conditionVersion, and entity are required');
    }
    const body: Record<string, unknown> = {
      condition_id:      params.conditionId,
      condition_version: params.conditionVersion,
      entity:            params.entity,
    };
    if (params.timestamp !== undefined) body.timestamp = params.timestamp;
    if (params.explain !== undefined)   body.explain   = params.explain;

    const raw = await this.http.post('/evaluate/condition', body);
    return mapDecisionResult(raw);
  }

  // ── evaluateConditionBatch ────────────────────────────────────────────────────

  async evaluateConditionBatch(params: EvaluateConditionBatchParams): Promise<DecisionResult[]> {
    if (!params.conditionId || !params.conditionVersion || !params.entities?.length) {
      throw new MemintelError('parameter_error', 'conditionId, conditionVersion, and entities are required');
    }
    const body: Record<string, unknown> = {
      condition_id:      params.conditionId,
      condition_version: params.conditionVersion,
      entities:          params.entities,
    };
    if (params.timestamp !== undefined) body.timestamp = params.timestamp;

    const raw = await this.http.post('/evaluate/condition/batch', body) as unknown[];
    return raw.map(mapDecisionResult);
  }

  // ── executeBatch ──────────────────────────────────────────────────────────────

  async executeBatch(params: ExecuteBatchParams): Promise<BatchExecuteResult> {
    if (!params.id || !params.version || !params.entities?.length) {
      throw new MemintelError('parameter_error', 'id, version, and entities are required');
    }
    const body: Record<string, unknown> = {
      id:       params.id,
      version:  params.version,
      entities: params.entities,
    };
    if (params.timestamp !== undefined) body.timestamp = params.timestamp;
    if (params.explain !== undefined)   body.explain   = params.explain;
    if (params.dryRun !== undefined)    body.dry_run   = params.dryRun;

    const raw = await this.http.post('/execute/batch', body);
    return mapBatchExecuteResult(raw);
  }

  // ── executeRange ──────────────────────────────────────────────────────────────

  async executeRange(params: ExecuteRangeParams): Promise<Result[]> {
    if (!params.id || !params.version || !params.entity) {
      throw new MemintelError('parameter_error', 'id, version, and entity are required');
    }
    const body: Record<string, unknown> = {
      id:             params.id,
      version:        params.version,
      entity:         params.entity,
      from_timestamp: params.fromTimestamp,
      to_timestamp:   params.toTimestamp,
      interval:       params.interval,
    };
    if (params.explain !== undefined) body.explain = params.explain;

    const raw = await this.http.post('/execute/range', body) as unknown[];
    return raw.map(mapResult);
  }

  // ── executeAsync ──────────────────────────────────────────────────────────────

  async executeAsync(params: ExecuteParams): Promise<Job> {
    if (!params.id || !params.version || !params.entity) {
      throw new MemintelError('parameter_error', 'id, version, and entity are required');
    }
    const body: Record<string, unknown> = {
      id:      params.id,
      version: params.version,
      entity:  params.entity,
    };
    if (params.timestamp !== undefined)          body.timestamp           = params.timestamp;
    if (params.explain !== undefined)            body.explain             = params.explain;
    if (params.dryRun !== undefined)             body.dry_run             = params.dryRun;
    if (params.missingDataPolicy !== undefined)  body.missing_data_policy = params.missingDataPolicy;

    const raw = await this.http.post('/execute/async', body);
    return mapJob(raw);
  }

  // ── getJob / cancelJob ────────────────────────────────────────────────────────

  async getJob(jobId: string): Promise<JobResult> {
    const raw = await this.http.get(`/jobs/${jobId}`);
    return mapJobResult(raw);
  }

  async cancelJob(jobId: string): Promise<JobResult> {
    const raw = await this.http.delete(`/jobs/${jobId}`);
    return mapJobResult(raw);
  }

  // ── explain ───────────────────────────────────────────────────────────────────

  async explain(params: ExplainParams): Promise<Result> {
    if (!params.id || !params.version || !params.entity) {
      throw new MemintelError('parameter_error', 'id, version, and entity are required');
    }
    const body: Record<string, unknown> = {
      id:      params.id,
      version: params.version,
      entity:  params.entity,
    };
    if (params.timestamp !== undefined) body.timestamp = params.timestamp;

    const raw = await this.http.post('/explain', body);
    return mapResult(raw);
  }

  // ── validate ──────────────────────────────────────────────────────────────────

  async validate(params: ValidateParams): Promise<ValidationResult> {
    const raw = await this.http.post('/definitions/validate', params.definition);
    return mapValidationResult(raw);
  }

  // ── conditionImpact ───────────────────────────────────────────────────────────

  async conditionImpact(params: ConditionImpactParams): Promise<Record<string, unknown>> {
    const raw = await this.http.post('/intelligence/condition-impact', params);
    return raw as Record<string, unknown>;
  }
}

export default Memintel;
