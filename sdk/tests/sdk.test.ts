/**
 * @memintel/sdk — unit tests
 *
 * Covers:
 *  1. evaluateFull returns result.decision.actionsTriggered (not top-level)
 *  2. MemintelError.type is one of the ErrorType enum values
 *  3. tasks.create() accepts second options arg with idempotencyKey
 *  4. FeedbackValue rejects anything other than the three valid values
 *  5. All camelCase fields map correctly from snake_case responses
 */

import Memintel, { MemintelError } from '../src/index';
import {
  ErrorType,
  FeedbackValue,
  FullPipelineResult,
  Task,
  TaskList,
} from '../src/types';
import {
  mapApplyCalibrationResult,
  mapCalibrationResult,
  mapConditionDefinition,
  mapDecisionExplanation,
  mapDecisionResult,
  mapFullPipelineResult,
  mapResult,
  mapTask,
  mapTaskList,
} from '../src/mappers';

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeFetchMock(body: unknown, status = 200): jest.Mock {
  return jest.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    headers: { get: () => null },
    json: () => Promise.resolve(body),
  });
}

// ── 1. evaluateFull — actionsTriggered lives only under decision ───────────────

describe('evaluateFull — actionsTriggered placement', () => {
  test('actionsTriggered is nested inside decision, not at FullPipelineResult top level', async () => {
    const rawResponse = {
      entity: 'user_abc',
      timestamp: '2024-03-15T09:00:00Z',
      dry_run: false,
      result: {
        value: 0.87,
        type: 'float',
        entity: 'user_abc',
        version: '1.2',
        deterministic: true,
        timestamp: '2024-03-15T09:00:00Z',
        explanation: null,
      },
      decision: {
        value: true,
        type: 'boolean',
        entity: 'user_abc',
        condition_id: 'org.high_churn',
        condition_version: '1.0',
        timestamp: '2024-03-15T09:00:00Z',
        actions_triggered: [
          {
            action_id: 'webhook_alert',
            action_version: '1.0',
            status: 'triggered',
            payload_sent: { entity: 'user_abc', value: 0.87 },
            error: null,
          },
        ],
      },
    };

    const result = mapFullPipelineResult(rawResponse);

    // actionsTriggered is ONLY on decision
    expect(result.decision.actionsTriggered).toBeDefined();
    expect(result.decision.actionsTriggered).toHaveLength(1);
    expect(result.decision.actionsTriggered[0].actionId).toBe('webhook_alert');
    expect(result.decision.actionsTriggered[0].actionVersion).toBe('1.0');
    expect(result.decision.actionsTriggered[0].status).toBe('triggered');

    // NOT at top level
    expect((result as Record<string, unknown>).actionsTriggered).toBeUndefined();
    // NOT on result
    expect((result.result as Record<string, unknown>).actionsTriggered).toBeUndefined();
  });

  test('evaluateFull with live HTTP call routes to /evaluate/full and maps decision.actionsTriggered', async () => {
    const rawResponse = {
      entity: 'user_abc',
      timestamp: '2024-03-15T09:00:00Z',
      dry_run: false,
      result: {
        value: 0.9,
        type: 'float',
        entity: 'user_abc',
        version: '1.0',
        deterministic: true,
        timestamp: '2024-03-15T09:00:00Z',
        explanation: null,
      },
      decision: {
        value: true,
        type: 'boolean',
        entity: 'user_abc',
        condition_id: 'org.high_churn',
        condition_version: '1.0',
        timestamp: '2024-03-15T09:00:00Z',
        actions_triggered: [
          {
            action_id: 'notify_ops',
            action_version: '2.0',
            status: 'would_trigger',
            payload_sent: null,
            error: null,
          },
        ],
      },
    };

    global.fetch = makeFetchMock(rawResponse);

    const client = new Memintel({ apiKey: 'test-key' });
    const result: FullPipelineResult = await client.evaluateFull({
      conceptId: 'org.churn_risk',
      conceptVersion: '1.0',
      conditionId: 'org.high_churn',
      conditionVersion: '1.0',
      entity: 'user_abc',
    });

    expect(result.decision.actionsTriggered).toHaveLength(1);
    expect(result.decision.actionsTriggered[0].actionId).toBe('notify_ops');
    // No top-level actionsTriggered
    expect((result as Record<string, unknown>).actionsTriggered).toBeUndefined();
  });
});

// ── 2. MemintelError.type is one of the ErrorType values ──────────────────────

describe('MemintelError', () => {
  const validTypes: ErrorType[] = [
    'syntax_error', 'type_error', 'semantic_error', 'reference_error',
    'parameter_error', 'graph_error', 'execution_error', 'execution_timeout',
    'auth_error', 'not_found', 'conflict', 'rate_limit_exceeded',
    'bounds_exceeded', 'action_binding_failed',
  ];

  test.each(validTypes)('MemintelError with type=%s has correct .type', (errType) => {
    const err = new MemintelError(errType, 'test message');
    expect(err.type).toBe(errType);
    expect(err instanceof MemintelError).toBe(true);
    expect(err instanceof Error).toBe(true);
    expect(err.message).toBe('test message');
  });

  test('MemintelError populates optional fields', () => {
    const err = new MemintelError('rate_limit_exceeded', 'Rate limit hit', {
      location: 'entity',
      suggestion: 'Wait before retrying',
      retryAfterSeconds: 30,
    });
    expect(err.type).toBe('rate_limit_exceeded');
    expect(err.location).toBe('entity');
    expect(err.suggestion).toBe('Wait before retrying');
    expect(err.retryAfterSeconds).toBe(30);
  });

  test('MemintelError from HTTP 401 response has auth_error type', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 401,
      statusText: 'Unauthorized',
      headers: { get: () => null },
      json: () => Promise.resolve({
        error: { type: 'auth_error', message: 'Invalid API key' },
      }),
    });

    const client = new Memintel({ apiKey: 'bad-key' });
    await expect(
      client.evaluateFull({
        conceptId: 'c', conceptVersion: '1', conditionId: 'x',
        conditionVersion: '1', entity: 'e',
      }),
    ).rejects.toMatchObject({
      type: 'auth_error',
      message: 'Invalid API key',
    });
  });

  test('MemintelError from HTTP 429 includes retryAfterSeconds', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 429,
      statusText: 'Too Many Requests',
      headers: { get: (h: string) => h === 'Retry-After' ? '60' : null },
      json: () => Promise.resolve({
        error: { type: 'rate_limit_exceeded', message: 'Rate limit exceeded' },
      }),
    });

    const client = new Memintel({ apiKey: 'test-key' });
    let caught: MemintelError | undefined;
    try {
      await client.execute({ id: 'x', version: '1', entity: 'e' });
    } catch (e) {
      caught = e as MemintelError;
    }

    expect(caught).toBeDefined();
    expect(caught?.type).toBe('rate_limit_exceeded');
    expect(caught?.retryAfterSeconds).toBe(60);
  });

  test('Memintel constructor throws auth_error when apiKey is missing', () => {
    expect(() => new Memintel({ apiKey: '' })).toThrow(MemintelError);
    try {
      new Memintel({ apiKey: '' });
    } catch (e) {
      expect((e as MemintelError).type).toBe('auth_error');
    }
  });
});

// ── 3. tasks.create() — idempotency key support ───────────────────────────────

describe('tasks.create() — idempotency key', () => {
  const taskResponse: Task = {
    taskId: 'task_123',
    intent: 'Alert when churn exceeds 0.8',
    conceptId: 'org.churn_risk',
    conceptVersion: '1.0',
    conditionId: 'org.high_churn',
    conditionVersion: '1.0',
    actionId: 'webhook_alert',
    actionVersion: '1.0',
    entityScope: 'user',
    delivery: { type: 'webhook', endpoint: 'https://app.example.com/hooks' },
    status: 'active',
    createdAt: '2024-01-01T00:00:00Z',
    lastTriggeredAt: null,
  };

  const snakeResponse = {
    task_id: 'task_123',
    intent: 'Alert when churn exceeds 0.8',
    concept_id: 'org.churn_risk',
    concept_version: '1.0',
    condition_id: 'org.high_churn',
    condition_version: '1.0',
    action_id: 'webhook_alert',
    action_version: '1.0',
    entity_scope: 'user',
    delivery: { type: 'webhook', endpoint: 'https://app.example.com/hooks' },
    status: 'active',
    created_at: '2024-01-01T00:00:00Z',
    last_triggered_at: null,
  };

  test('tasks.create() sends Idempotency-Key header when idempotencyKey provided', async () => {
    global.fetch = makeFetchMock(snakeResponse);

    const client = new Memintel({ apiKey: 'test-key' });
    await client.tasks.create(
      {
        intent: 'Alert when churn exceeds 0.8',
        entityScope: 'user',
        delivery: { type: 'webhook', endpoint: 'https://app.example.com/hooks' },
      },
      { idempotencyKey: 'idem-key-abc123' },
    );

    const callArgs = (global.fetch as jest.Mock).mock.calls[0];
    const headers = callArgs[1].headers as Record<string, string>;
    expect(headers['Idempotency-Key']).toBe('idem-key-abc123');
  });

  test('tasks.create() does NOT send Idempotency-Key when no options provided', async () => {
    global.fetch = makeFetchMock(snakeResponse);

    const client = new Memintel({ apiKey: 'test-key' });
    await client.tasks.create({
      intent: 'Alert when churn exceeds 0.8',
      entityScope: 'user',
      delivery: { type: 'webhook', endpoint: 'https://app.example.com/hooks' },
    });

    const callArgs = (global.fetch as jest.Mock).mock.calls[0];
    const headers = callArgs[1].headers as Record<string, string>;
    expect(headers['Idempotency-Key']).toBeUndefined();
  });

  test('tasks.create() returns mapped Task with camelCase fields', async () => {
    global.fetch = makeFetchMock(snakeResponse);

    const client = new Memintel({ apiKey: 'test-key' });
    const task = await client.tasks.create({
      intent: 'Alert when churn exceeds 0.8',
      entityScope: 'user',
      delivery: { type: 'webhook', endpoint: 'https://app.example.com/hooks' },
    });

    // Task should be returned (not DryRunResult) for non-dry-run calls
    const t = task as Task;
    expect(t.taskId).toBe('task_123');
    expect(t.conceptId).toBe('org.churn_risk');
    expect(t.conditionVersion).toBe('1.0');
    expect(t.entityScope).toBe('user');
    // snake_case fields must NOT exist on the result
    expect((t as Record<string, unknown>).task_id).toBeUndefined();
    expect((t as Record<string, unknown>).concept_id).toBeUndefined();
  });
});

// ── 4. FeedbackValue — type enforcement ───────────────────────────────────────

describe('FeedbackClient — FeedbackValue validation', () => {
  test('feedback.submit() throws parameter_error for invalid value "useful"', async () => {
    const client = new Memintel({ apiKey: 'test-key' });

    await expect(
      client.feedback.submit({
        conditionId: 'org.high_churn',
        conditionVersion: '1.0',
        entity: 'user_abc',
        timestamp: '2024-03-15T09:00:00Z',
        feedbackType: 'useful' as FeedbackValue, // invalid
      }),
    ).rejects.toMatchObject({
      type: 'parameter_error',
    });
  });

  test('feedback.submit() throws parameter_error for invalid value "not_useful"', async () => {
    const client = new Memintel({ apiKey: 'test-key' });

    await expect(
      client.feedback.submit({
        conditionId: 'org.high_churn',
        conditionVersion: '1.0',
        entity: 'user_abc',
        timestamp: '2024-03-15T09:00:00Z',
        feedbackType: 'not_useful' as FeedbackValue, // invalid
      }),
    ).rejects.toMatchObject({
      type: 'parameter_error',
    });
  });

  test('feedback.submit() throws BEFORE making HTTP call for invalid value', async () => {
    global.fetch = jest.fn();
    const client = new Memintel({ apiKey: 'test-key' });

    try {
      await client.feedback.submit({
        conditionId: 'cond',
        conditionVersion: '1.0',
        entity: 'user',
        timestamp: '2024-01-01T00:00:00Z',
        feedbackType: 'thumbs_up' as FeedbackValue,
      });
    } catch {
      // expected
    }

    expect(global.fetch).not.toHaveBeenCalled();
  });

  test('feedback.submit() accepts all three valid FeedbackValue values', async () => {
    const validValues: FeedbackValue[] = ['false_positive', 'false_negative', 'correct'];

    for (const feedbackType of validValues) {
      global.fetch = makeFetchMock({ status: 'recorded', feedback_id: 'fb_123' });
      const client = new Memintel({ apiKey: 'test-key' });

      const result = await client.feedback.submit({
        conditionId: 'org.high_churn',
        conditionVersion: '1.0',
        entity: 'user_abc',
        timestamp: '2024-03-15T09:00:00Z',
        feedbackType,
      });

      expect(result.status).toBe('recorded');
      expect(result.feedbackId).toBe('fb_123');
    }
  });
});

// ── 5. camelCase ↔ snake_case field mapping ───────────────────────────────────

describe('camelCase field mapping from snake_case responses', () => {
  test('mapTask maps all snake_case fields to camelCase', () => {
    const raw = {
      task_id: 'task_xyz',
      intent: 'Monitor churn',
      concept_id: 'org.churn_risk',
      concept_version: '2.0',
      condition_id: 'org.high_churn',
      condition_version: '1.1',
      action_id: 'slack_notify',
      action_version: '1.0',
      entity_scope: 'user',
      delivery: { type: 'notification', channel: 'ops-alerts' },
      status: 'active',
      created_at: '2024-01-01T00:00:00Z',
      last_triggered_at: '2024-03-01T12:00:00Z',
    };

    const task = mapTask(raw);

    expect(task.taskId).toBe('task_xyz');
    expect(task.conceptId).toBe('org.churn_risk');
    expect(task.conceptVersion).toBe('2.0');
    expect(task.conditionId).toBe('org.high_churn');
    expect(task.conditionVersion).toBe('1.1');
    expect(task.actionId).toBe('slack_notify');
    expect(task.actionVersion).toBe('1.0');
    expect(task.entityScope).toBe('user');
    expect(task.createdAt).toBe('2024-01-01T00:00:00Z');
    expect(task.lastTriggeredAt).toBe('2024-03-01T12:00:00Z');
    // delivery workflowId mapping
    expect(task.delivery.type).toBe('notification');
    expect(task.delivery.channel).toBe('ops-alerts');
    // snake_case fields NOT present
    expect((task as Record<string, unknown>).task_id).toBeUndefined();
    expect((task as Record<string, unknown>).concept_id).toBeUndefined();
    expect((task as Record<string, unknown>).entity_scope).toBeUndefined();
    expect((task as Record<string, unknown>).last_triggered_at).toBeUndefined();
  });

  test('mapTaskList maps items, hasMore, nextCursor, totalCount', () => {
    const raw = {
      items: [
        {
          task_id: 't1', intent: 'i1', concept_id: 'c1', concept_version: '1',
          condition_id: 'cond1', condition_version: '1', action_id: 'a1', action_version: '1',
          entity_scope: 'user', delivery: { type: 'webhook', endpoint: 'https://x.com' },
          status: 'active',
        },
      ],
      has_more: true,
      next_cursor: 'cursor_abc',
      total_count: 42,
    };

    const list: TaskList = mapTaskList(raw);

    expect(list.hasMore).toBe(true);
    expect(list.nextCursor).toBe('cursor_abc');
    expect(list.totalCount).toBe(42);
    expect(list.items).toHaveLength(1);
    expect(list.items[0].taskId).toBe('t1');
    // snake_case NOT present
    expect((list as Record<string, unknown>).has_more).toBeUndefined();
    expect((list as Record<string, unknown>).next_cursor).toBeUndefined();
  });

  test('mapDecisionResult maps condition_id, condition_version, actions_triggered → actionsTriggered', () => {
    const raw = {
      value: true,
      type: 'boolean',
      entity: 'user_abc',
      condition_id: 'org.risk',
      condition_version: '1.0',
      timestamp: '2024-01-01T00:00:00Z',
      actions_triggered: [
        { action_id: 'a1', action_version: '1', status: 'triggered', payload_sent: null, error: null },
      ],
    };

    const result = mapDecisionResult(raw);

    expect(result.conditionId).toBe('org.risk');
    expect(result.conditionVersion).toBe('1.0');
    expect(result.actionsTriggered).toHaveLength(1);
    expect(result.actionsTriggered[0].actionId).toBe('a1');
    expect(result.actionsTriggered[0].payloadSent).toBeNull();
    // snake_case NOT present
    expect((result as Record<string, unknown>).condition_id).toBeUndefined();
    expect((result as Record<string, unknown>).actions_triggered).toBeUndefined();
  });

  test('mapCalibrationResult maps all calibration snake_case fields', () => {
    const raw = {
      status: 'recommendation_available',
      recommended_params: { value: 0.75 },
      calibration_token: 'tok_abc',
      current_params: { value: 0.8 },
      no_recommendation_reason: null,
      impact: { delta_alerts: -2.5, direction: 'decrease' },
    };

    const result = mapCalibrationResult(raw);

    expect(result.recommendedParams).toEqual({ value: 0.75 });
    expect(result.calibrationToken).toBe('tok_abc');
    expect(result.currentParams).toEqual({ value: 0.8 });
    expect(result.noRecommendationReason).toBeNull();
    expect(result.impact?.deltaAlerts).toBe(-2.5);
    expect(result.impact?.direction).toBe('decrease');
    // snake_case NOT present
    expect((result as Record<string, unknown>).recommended_params).toBeUndefined();
    expect((result as Record<string, unknown>).calibration_token).toBeUndefined();
    expect((result as Record<string, unknown>).no_recommendation_reason).toBeUndefined();
  });

  test('mapApplyCalibrationResult maps tasks_pending_rebind → tasksPendingRebind', () => {
    const raw = {
      condition_id: 'org.high_churn',
      previous_version: '1.0',
      new_version: '1.1',
      params_applied: { value: 0.75 },
      tasks_pending_rebind: [
        { task_id: 'task_001', intent: 'Monitor churn for VIPs' },
        { task_id: 'task_002', intent: 'Alert on high-value accounts' },
      ],
    };

    const result = mapApplyCalibrationResult(raw);

    expect(result.conditionId).toBe('org.high_churn');
    expect(result.previousVersion).toBe('1.0');
    expect(result.newVersion).toBe('1.1');
    expect(result.paramsApplied).toEqual({ value: 0.75 });
    expect(result.tasksPendingRebind).toHaveLength(2);
    expect(result.tasksPendingRebind[0].taskId).toBe('task_001');
    expect(result.tasksPendingRebind[1].taskId).toBe('task_002');
    // snake_case NOT present
    expect((result as Record<string, unknown>).previous_version).toBeUndefined();
    expect((result as Record<string, unknown>).tasks_pending_rebind).toBeUndefined();
    expect((result as Record<string, unknown>).params_applied).toBeUndefined();
  });

  test('mapDecisionExplanation maps all snake_case fields', () => {
    const raw = {
      condition_id: 'org.high_churn',
      condition_version: '1.0',
      entity: 'user_abc',
      timestamp: '2024-03-15T09:00:00Z',
      decision: true,
      decision_type: 'boolean',
      concept_value: 0.87,
      strategy_type: 'threshold',
      threshold_applied: 0.8,
      label_matched: null,
      drivers: [
        { signal: 'logins_last_7d', contribution: 0.6, value: 2 },
        { signal: 'support_tickets', contribution: 0.4, value: 5 },
      ],
    };

    const result = mapDecisionExplanation(raw);

    expect(result.conditionId).toBe('org.high_churn');
    expect(result.conditionVersion).toBe('1.0');
    expect(result.decisionType).toBe('boolean');
    expect(result.conceptValue).toBe(0.87);
    expect(result.strategyType).toBe('threshold');
    expect(result.thresholdApplied).toBe(0.8);
    expect(result.labelMatched).toBeNull();
    expect(result.drivers).toHaveLength(2);
    expect(result.drivers[0].signal).toBe('logins_last_7d');
    expect(result.drivers[0].contribution).toBe(0.6);
    // snake_case NOT present
    expect((result as Record<string, unknown>).condition_id).toBeUndefined();
    expect((result as Record<string, unknown>).decision_type).toBeUndefined();
    expect((result as Record<string, unknown>).concept_value).toBeUndefined();
    expect((result as Record<string, unknown>).strategy_type).toBeUndefined();
    expect((result as Record<string, unknown>).threshold_applied).toBeUndefined();
    expect((result as Record<string, unknown>).label_matched).toBeUndefined();
  });

  test('mapResult maps concept result correctly', () => {
    const raw = {
      value: 0.91,
      type: 'float',
      entity: 'user_abc',
      version: '1.0',
      deterministic: true,
      timestamp: '2024-03-15T09:00:00Z',
      explanation: null,
    };

    const result = mapResult(raw);

    expect(result.value).toBe(0.91);
    expect(result.type).toBe('float');
    expect(result.deterministic).toBe(true);
    expect(result.explanation).toBeNull();
  });

  test('mapConditionDefinition maps snake_case fields correctly', () => {
    const raw = {
      condition_id: 'org.high_churn',
      version: '1.0',
      concept_id: 'org.churn_risk',
      concept_version: '1.2',
      strategy: { type: 'threshold', params: { direction: 'above', value: 0.8 } },
      namespace: 'org',
      created_at: '2024-01-01T00:00:00Z',
      deprecated: false,
    };

    const result = mapConditionDefinition(raw);

    expect(result.conditionId).toBe('org.high_churn');
    expect(result.conceptId).toBe('org.churn_risk');
    expect(result.conceptVersion).toBe('1.2');
    expect(result.strategy.type).toBe('threshold');
    expect(result.strategy.params).toEqual({ direction: 'above', value: 0.8 });
    expect((result as Record<string, unknown>).condition_id).toBeUndefined();
    expect((result as Record<string, unknown>).concept_id).toBeUndefined();
  });

  test('DeliveryConfig maps workflow_id → workflowId', () => {
    const raw = {
      task_id: 'task_w1',
      intent: 'Workflow task',
      concept_id: 'c1', concept_version: '1',
      condition_id: 'cond1', condition_version: '1',
      action_id: 'a1', action_version: '1',
      entity_scope: 'order',
      delivery: { type: 'workflow', workflow_id: 'wf_abc' },
      status: 'active',
    };

    const task = mapTask(raw);
    expect(task.delivery.workflowId).toBe('wf_abc');
    expect((task.delivery as Record<string, unknown>).workflow_id).toBeUndefined();
  });
});

// ── 6. tasks.update() — empty patch throws parameter_error ───────────────────

describe('tasks.update() validation', () => {
  test('throws parameter_error when no fields provided', async () => {
    const client = new Memintel({ apiKey: 'test-key' });
    await expect(
      client.tasks.update('task_123', {}),
    ).rejects.toMatchObject({ type: 'parameter_error' });
  });
});

// ── 7. evaluateFull validates required fields ─────────────────────────────────

describe('evaluateFull — required field validation', () => {
  test('throws parameter_error when entity is missing', async () => {
    const client = new Memintel({ apiKey: 'test-key' });
    await expect(
      client.evaluateFull({
        conceptId: 'c', conceptVersion: '1',
        conditionId: 'x', conditionVersion: '1',
        entity: '', // empty
      }),
    ).rejects.toMatchObject({ type: 'parameter_error' });
  });
});
