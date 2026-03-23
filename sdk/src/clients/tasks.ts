import { HttpClient } from '../http';
import {
  CreateTaskParams,
  DryRunResult,
  Task,
  TaskList,
  TaskStatus,
  UpdateTaskParams,
} from '../types';
import {
  mapConstraintsToSnake,
  mapDeliveryToSnake,
  mapDryRunResult,
  mapTask,
  mapTaskList,
} from '../mappers';

export class TasksClient {
  constructor(private http: HttpClient) {}

  async create(
    params: CreateTaskParams,
    options?: { idempotencyKey?: string },
  ): Promise<Task | DryRunResult> {
    const body: Record<string, unknown> = {
      intent:       params.intent,
      entity_scope: params.entityScope,
      delivery:     mapDeliveryToSnake(params.delivery),
    };
    if (params.constraints) body.constraints = mapConstraintsToSnake(params.constraints);
    if (params.dryRun !== undefined) body.dry_run = params.dryRun;

    const headers: Record<string, string> = {};
    if (options?.idempotencyKey) {
      headers['Idempotency-Key'] = options.idempotencyKey;
    }

    const raw = await this.http.post('/tasks', body, headers);
    // dry_run responses return a DryRunResult (no task_id, no concept_id at top level)
    if (params.dryRun) {
      return mapDryRunResult(raw);
    }
    return mapTask(raw);
  }

  async list(params?: {
    status?: TaskStatus;
    limit?: number;
    cursor?: string;
  }): Promise<TaskList> {
    const raw = await this.http.get('/tasks', params as Record<string, string | number | boolean | undefined>);
    return mapTaskList(raw);
  }

  async get(id: string): Promise<Task> {
    const raw = await this.http.get(`/tasks/${id}`);
    return mapTask(raw);
  }

  async update(
    id: string,
    params: UpdateTaskParams,
    options?: { idempotencyKey?: string },
  ): Promise<Task> {
    if (Object.keys(params).length === 0) {
      throw new (await import('../error')).MemintelError(
        'parameter_error',
        'At least one field must be provided for update',
      );
    }
    const body: Record<string, unknown> = {};
    if (params.conditionVersion !== undefined) body.condition_version = params.conditionVersion;
    if (params.delivery !== undefined) body.delivery = mapDeliveryToSnake(params.delivery);
    if (params.entityScope !== undefined) body.entity_scope = params.entityScope;
    if (params.status !== undefined) body.status = params.status;

    const headers: Record<string, string> = {};
    if (options?.idempotencyKey) headers['Idempotency-Key'] = options.idempotencyKey;

    const raw = await this.http.patch(`/tasks/${id}`, body, headers);
    return mapTask(raw);
  }

  async delete(id: string, options?: { idempotencyKey?: string }): Promise<Task> {
    const headers: Record<string, string> = {};
    if (options?.idempotencyKey) headers['Idempotency-Key'] = options.idempotencyKey;
    const raw = await this.http.delete(`/tasks/${id}`, headers);
    return mapTask(raw);
  }
}
