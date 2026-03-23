/**
 * tests/sdk_scenarios.test.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * SDK integration scenario tests — all in-process using fetch mocks.
 *
 * Scenarios
 * ─────────
 *  1. Route table      — every SDK method calls the correct HTTP method + path
 *  2. Error surfacing  — 400/404/409/422 responses → typed MemintelError
 *  3. dryRun param     — evaluateFull/executeBatch send dry_run in body
 *  4. Pagination cursor — cursor forwarded as query param; nextCursor usable
 *  5. Timeout          — SDK configures AbortSignal; timeout propagates as
 *                        clean rejection.
 *                        GAP: raw AbortError is NOT wrapped as MemintelError.
 */

import Memintel, { MemintelError } from '../src/index';

// ── Shared helpers ─────────────────────────────────────────────────────────────

/** Minimal fake responses per return type, keyed by shape. */
const STUBS = {
  fullPipeline: {
    entity: 'e', timestamp: null, dry_run: false,
    result: { value: 0.5, type: 'float', entity: 'e', version: '1', deterministic: true, timestamp: null },
    decision: { value: true, type: 'boolean', entity: 'e', condition_id: 'c', condition_version: '1', timestamp: null, actions_triggered: [] },
  },
  result: { value: 0.5, type: 'float', entity: 'e', version: '1', deterministic: true, timestamp: null },
  decisionResult: { value: true, type: 'boolean', entity: 'e', condition_id: 'c', condition_version: '1', timestamp: null, actions_triggered: [] },
  batchResult: { total: 0, failed: 0, items: [] },
  job: { job_id: 'job_1', status: 'queued' },
  jobResult: { job_id: 'job_1', status: 'completed' },
  task: {
    task_id: 'task_1', intent: 'test', concept_id: 'c', concept_version: '1',
    condition_id: 'cond', condition_version: '1', action_id: 'a', action_version: '1',
    entity_scope: 'user', delivery: { type: 'webhook', endpoint: 'https://x.com' }, status: 'active',
  },
  taskList: { items: [], has_more: false, next_cursor: null, total_count: 0 },
  conditionDef: {
    condition_id: 'cond', version: '1', concept_id: 'c', concept_version: '1',
    strategy: { type: 'threshold', params: { direction: 'above', value: 0.7 } }, namespace: 'org',
  },
  conditionExplanation: {
    condition_id: 'cond', condition_version: '1', strategy: { type: 'threshold', params: {} },
    concept_id: 'c', concept_version: '1', natural_language_summary: 's', parameter_rationale: 'r',
  },
  calibrationResult: {
    status: 'no_recommendation', current_params: { value: 0.7 },
    no_recommendation_reason: 'insufficient_data',
  },
  applyCalibrationResult: {
    condition_id: 'cond', previous_version: '1.0', new_version: '1.1',
    params_applied: { value: 0.75 }, tasks_pending_rebind: [],
  },
  decisionExplanation: {
    condition_id: 'cond', condition_version: '1', entity: 'e', timestamp: null,
    decision: true, decision_type: 'boolean', concept_value: 0.5,
    strategy_type: 'threshold', threshold_applied: 0.7, label_matched: null, drivers: [],
  },
  feedback: { status: 'recorded', feedback_id: 'fb_1' },
  searchResult: { items: [], total: 0 },
  versionList: { versions: [] },
  featureSearch: { items: [], total: 0 },
  registeredFeature: { feature_id: 'f1' },
  action: { action_id: 'a1', action_version: '1.0' },
  actionList: { items: [{ action_id: 'a1', action_version: '1.0' }] },
  actionResult: { status: 'triggered' },
  agentQuery: { response: 'ok' },
  agentDefine: { definition: {} },
  validationResult: { valid: true, errors: [], warnings: [] },
  conditionImpact: {},
};

function ok(body: unknown): jest.Mock {
  return jest.fn().mockResolvedValue({
    ok: true, status: 200, statusText: 'OK',
    headers: { get: () => null },
    json: () => Promise.resolve(body),
  });
}

function errorResponse(status: number, errorType: string, message = 'err',
    extras?: { location?: string; suggestion?: string }): jest.Mock {
  return jest.fn().mockResolvedValue({
    ok: false, status, statusText: 'Error',
    headers: { get: () => null },
    json: () => Promise.resolve({ error: { type: errorType, message, ...extras } }),
  });
}

/** Extract the URL string and RequestInit from the first fetch call. */
function firstCall(): [string, RequestInit] {
  const mock = global.fetch as jest.Mock;
  expect(mock).toHaveBeenCalledTimes(1);
  return [mock.mock.calls[0][0] as string, mock.mock.calls[0][1] as RequestInit];
}

const BASE = 'https://api.memsdl.ai/v1';

let client: Memintel;
beforeEach(() => {
  client = new Memintel({ apiKey: 'test-key' });
});
afterEach(() => {
  jest.restoreAllMocks();
});

// ══════════════════════════════════════════════════════════════════════════════
// Scenario 1 — Route table: correct HTTP method + path for every SDK method
// ══════════════════════════════════════════════════════════════════════════════

describe('Scenario 1 — Route table', () => {

  // ── Top-level methods ────────────────────────────────────────────────────────

  test('evaluateFull → POST /evaluate/full', async () => {
    global.fetch = ok(STUBS.fullPipeline);
    await client.evaluateFull({ conceptId: 'c', conceptVersion: '1', conditionId: 'x', conditionVersion: '1', entity: 'e' });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/evaluate/full`);
    expect(init.method).toBe('POST');
  });

  test('execute → POST /execute', async () => {
    global.fetch = ok(STUBS.result);
    await client.execute({ id: 'c', version: '1', entity: 'e' });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/execute`);
    expect(init.method).toBe('POST');
  });

  test('evaluateCondition → POST /evaluate/condition', async () => {
    global.fetch = ok(STUBS.decisionResult);
    await client.evaluateCondition({ conditionId: 'c', conditionVersion: '1', entity: 'e' });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/evaluate/condition`);
    expect(init.method).toBe('POST');
  });

  test('evaluateConditionBatch → POST /evaluate/condition/batch', async () => {
    global.fetch = ok([]);
    await client.evaluateConditionBatch({ conditionId: 'c', conditionVersion: '1', entities: ['e1'] });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/evaluate/condition/batch`);
    expect(init.method).toBe('POST');
  });

  test('executeBatch → POST /execute/batch', async () => {
    global.fetch = ok(STUBS.batchResult);
    await client.executeBatch({ id: 'c', version: '1', entities: ['e1'] });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/execute/batch`);
    expect(init.method).toBe('POST');
  });

  test('executeRange → POST /execute/range', async () => {
    global.fetch = ok([]);
    await client.executeRange({
      id: 'c', version: '1', entity: 'e',
      fromTimestamp: '2024-01-01T00:00:00Z', toTimestamp: '2024-01-07T00:00:00Z', interval: '1d',
    });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/execute/range`);
    expect(init.method).toBe('POST');
  });

  test('executeAsync → POST /execute/async', async () => {
    global.fetch = ok(STUBS.job);
    await client.executeAsync({ id: 'c', version: '1', entity: 'e' });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/execute/async`);
    expect(init.method).toBe('POST');
  });

  test('getJob → GET /jobs/{jobId}', async () => {
    global.fetch = ok(STUBS.jobResult);
    await client.getJob('job_abc');
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/jobs/job_abc`);
    expect(init.method).toBe('GET');
  });

  test('cancelJob → DELETE /jobs/{jobId}', async () => {
    global.fetch = ok(STUBS.jobResult);
    await client.cancelJob('job_abc');
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/jobs/job_abc`);
    expect(init.method).toBe('DELETE');
  });

  test('explain → POST /explain', async () => {
    global.fetch = ok(STUBS.result);
    await client.explain({ id: 'c', version: '1', entity: 'e' });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/explain`);
    expect(init.method).toBe('POST');
  });

  test('validate → POST /definitions/validate', async () => {
    global.fetch = ok(STUBS.validationResult);
    await client.validate({ definition: {} });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/definitions/validate`);
    expect(init.method).toBe('POST');
  });

  test('conditionImpact → POST /intelligence/condition-impact', async () => {
    global.fetch = ok(STUBS.conditionImpact);
    await client.conditionImpact({ conditionId: 'c', conditionVersion: '1' } as never);
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/intelligence/condition-impact`);
    expect(init.method).toBe('POST');
  });

  // ── Tasks sub-client ─────────────────────────────────────────────────────────

  test('tasks.create → POST /tasks', async () => {
    global.fetch = ok(STUBS.task);
    await client.tasks.create({ intent: 'i', entityScope: 'user', delivery: { type: 'webhook', endpoint: 'https://x.com' } });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/tasks`);
    expect(init.method).toBe('POST');
  });

  test('tasks.list → GET /tasks', async () => {
    global.fetch = ok(STUBS.taskList);
    await client.tasks.list();
    const [url, init] = firstCall();
    expect(url).toMatch(/^https:\/\/api\.memsdl\.ai\/v1\/tasks(\?.*)?$/);
    expect(init.method).toBe('GET');
  });

  test('tasks.get → GET /tasks/{id}', async () => {
    global.fetch = ok(STUBS.task);
    await client.tasks.get('task_xyz');
    const [url, init] = firstCall();
    expect(url).toMatch(/\/tasks\/task_xyz$/);
    expect(init.method).toBe('GET');
  });

  test('tasks.update → PATCH /tasks/{id}', async () => {
    global.fetch = ok(STUBS.task);
    await client.tasks.update('task_xyz', { status: 'paused' });
    const [url, init] = firstCall();
    expect(url).toMatch(/\/tasks\/task_xyz$/);
    expect(init.method).toBe('PATCH');
  });

  test('tasks.delete → DELETE /tasks/{id}', async () => {
    global.fetch = ok(STUBS.task);
    await client.tasks.delete('task_xyz');
    const [url, init] = firstCall();
    expect(url).toMatch(/\/tasks\/task_xyz$/);
    expect(init.method).toBe('DELETE');
  });

  // ── Conditions sub-client ────────────────────────────────────────────────────

  test('conditions.get → GET /conditions/{id}?version=...', async () => {
    global.fetch = ok(STUBS.conditionDef);
    await client.conditions.get('org.churn', '1.0');
    const [url, init] = firstCall();
    expect(url).toMatch(/\/conditions\/org\.churn\?version=1\.0$/);
    expect(init.method).toBe('GET');
  });

  test('conditions.explain → POST /conditions/explain', async () => {
    global.fetch = ok(STUBS.conditionExplanation);
    await client.conditions.explain({ conditionId: 'c', conditionVersion: '1' });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/conditions/explain`);
    expect(init.method).toBe('POST');
  });

  test('conditions.calibrate → POST /conditions/calibrate', async () => {
    global.fetch = ok(STUBS.calibrationResult);
    await client.conditions.calibrate({ conditionId: 'c', conditionVersion: '1' });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/conditions/calibrate`);
    expect(init.method).toBe('POST');
  });

  test('conditions.applyCalibration → POST /conditions/apply-calibration', async () => {
    global.fetch = ok(STUBS.applyCalibrationResult);
    await client.conditions.applyCalibration({ calibrationToken: 'tok' });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/conditions/apply-calibration`);
    expect(init.method).toBe('POST');
  });

  // ── Other sub-clients ────────────────────────────────────────────────────────

  test('decisions.explain → POST /decisions/explain', async () => {
    global.fetch = ok(STUBS.decisionExplanation);
    await client.decisions.explain({ conditionId: 'c', conditionVersion: '1', entity: 'e', timestamp: '2024-01-01T00:00:00Z' });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/decisions/explain`);
    expect(init.method).toBe('POST');
  });

  test('feedback.submit → POST /feedback/decision', async () => {
    global.fetch = ok(STUBS.feedback);
    await client.feedback.submit({ conditionId: 'c', conditionVersion: '1', entity: 'e', timestamp: '2024-01-01T00:00:00Z', feedbackType: 'correct' });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/feedback/decision`);
    expect(init.method).toBe('POST');
  });

  test('registry.list → GET /registry/definitions', async () => {
    global.fetch = ok(STUBS.searchResult);
    await client.registry.list();
    const [url, init] = firstCall();
    expect(url).toMatch(/\/registry\/definitions(\?.*)?$/);
    expect(init.method).toBe('GET');
  });

  test('registry.search → GET /registry/search', async () => {
    global.fetch = ok(STUBS.searchResult);
    await client.registry.search({ query: 'churn' });
    const [url, init] = firstCall();
    expect(url).toMatch(/\/registry\/search\?/);
    expect(init.method).toBe('GET');
  });

  test('registry.versions → GET /registry/definitions/{id}/versions', async () => {
    global.fetch = ok(STUBS.versionList);
    await client.registry.versions('org.churn_risk');
    const [url, init] = firstCall();
    expect(url).toMatch(/\/registry\/definitions\/org\.churn_risk\/versions$/);
    expect(init.method).toBe('GET');
  });

  test('actions.list → GET /actions', async () => {
    global.fetch = ok(STUBS.actionList);
    await client.actions.list();
    const [url, init] = firstCall();
    expect(url).toMatch(/\/actions(\?.*)?$/);
    expect(init.method).toBe('GET');
  });

  test('actions.get → GET /actions/{id}', async () => {
    global.fetch = ok(STUBS.action);
    await client.actions.get('act_1');
    const [url, init] = firstCall();
    expect(url).toMatch(/\/actions\/act_1$/);
    expect(init.method).toBe('GET');
  });

  test('actions.trigger → POST /actions/{id}/trigger', async () => {
    global.fetch = ok(STUBS.actionResult);
    await client.actions.trigger('act_1', { entity: 'e' });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/actions/act_1/trigger`);
    expect(init.method).toBe('POST');
  });

  test('features.search → GET /registry/features', async () => {
    global.fetch = ok(STUBS.featureSearch);
    await client.features.search();
    const [url, init] = firstCall();
    expect(url).toMatch(/\/registry\/features(\?.*)?$/);
    expect(init.method).toBe('GET');
  });

  test('features.get → GET /registry/features/{id}', async () => {
    global.fetch = ok(STUBS.registeredFeature);
    await client.features.get('feat_1');
    const [url, init] = firstCall();
    expect(url).toMatch(/\/registry\/features\/feat_1$/);
    expect(init.method).toBe('GET');
  });

  test('agents.query → POST /agents/query', async () => {
    global.fetch = ok(STUBS.agentQuery);
    await client.agents.query({ query: 'q' });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/agents/query`);
    expect(init.method).toBe('POST');
  });

  test('agents.define → POST /agents/define', async () => {
    global.fetch = ok(STUBS.agentDefine);
    await client.agents.define({ intent: 'i' });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/agents/define`);
    expect(init.method).toBe('POST');
  });

  test('agents.defineCondition → POST /agents/define-condition', async () => {
    global.fetch = ok(STUBS.agentDefine);
    await client.agents.defineCondition({ intent: 'i', conceptId: 'c', conceptVersion: '1' });
    const [url, init] = firstCall();
    expect(url).toBe(`${BASE}/agents/define-condition`);
    expect(init.method).toBe('POST');
  });

  // ── Request body field names (snake_case) ────────────────────────────────────

  test('evaluateFull sends snake_case body fields to API', async () => {
    global.fetch = ok(STUBS.fullPipeline);
    await client.evaluateFull({
      conceptId: 'org.churn', conceptVersion: '1.2',
      conditionId: 'org.risk', conditionVersion: '2.0', entity: 'user_1',
    });
    const [, init] = firstCall();
    const body = JSON.parse(init.body as string);
    expect(body.concept_id).toBe('org.churn');
    expect(body.concept_version).toBe('1.2');
    expect(body.condition_id).toBe('org.risk');
    expect(body.condition_version).toBe('2.0');
    expect(body.entity).toBe('user_1');
    // camelCase must NOT appear in the body
    expect(body.conceptId).toBeUndefined();
    expect(body.conditionId).toBeUndefined();
  });

  test('tasks.create sends entity_scope and dry_run snake_case', async () => {
    global.fetch = ok(STUBS.task);
    await client.tasks.create({
      intent: 'Monitor churn',
      entityScope: 'user',
      delivery: { type: 'webhook', endpoint: 'https://x.com' },
      dryRun: false,
    });
    const [, init] = firstCall();
    const body = JSON.parse(init.body as string);
    expect(body.entity_scope).toBe('user');
    expect(body.dry_run).toBe(false);
    expect(body.entityScope).toBeUndefined();
    expect(body.dryRun).toBeUndefined();
  });

  test('conditions.calibrate sends feedback_direction snake_case', async () => {
    global.fetch = ok(STUBS.calibrationResult);
    await client.conditions.calibrate({
      conditionId: 'c', conditionVersion: '1',
      feedbackDirection: 'tighten',
    });
    const [, init] = firstCall();
    const body = JSON.parse(init.body as string);
    expect(body.feedback_direction).toBe('tighten');
    expect(body.condition_id).toBe('c');
    expect(body.condition_version).toBe('1');
    expect(body.feedbackDirection).toBeUndefined();
  });

  test('conditions.applyCalibration sends calibration_token snake_case', async () => {
    global.fetch = ok(STUBS.applyCalibrationResult);
    await client.conditions.applyCalibration({
      calibrationToken: 'tok_abc123',
      newVersion: '2.0',
    });
    const [, init] = firstCall();
    const body = JSON.parse(init.body as string);
    expect(body.calibration_token).toBe('tok_abc123');
    expect(body.new_version).toBe('2.0');
    expect(body.calibrationToken).toBeUndefined();
    expect(body.newVersion).toBeUndefined();
  });

  test('X-API-Key header is sent on every request type', async () => {
    // POST
    global.fetch = ok(STUBS.result);
    await client.execute({ id: 'c', version: '1', entity: 'e' });
    expect((firstCall()[1].headers as Record<string, string>)['X-API-Key']).toBe('test-key');

    // GET
    (global.fetch as jest.Mock).mockClear();
    global.fetch = ok(STUBS.taskList);
    await client.tasks.list();
    expect((firstCall()[1].headers as Record<string, string>)['X-API-Key']).toBe('test-key');

    // PATCH
    (global.fetch as jest.Mock).mockClear();
    global.fetch = ok(STUBS.task);
    await client.tasks.update('t', { status: 'paused' });
    expect((firstCall()[1].headers as Record<string, string>)['X-API-Key']).toBe('test-key');

    // DELETE
    (global.fetch as jest.Mock).mockClear();
    global.fetch = ok(STUBS.task);
    await client.tasks.delete('t');
    expect((firstCall()[1].headers as Record<string, string>)['X-API-Key']).toBe('test-key');
  });
});


// ══════════════════════════════════════════════════════════════════════════════
// Scenario 2 — Error surfacing: 400/404/409/422 → typed MemintelError
// ══════════════════════════════════════════════════════════════════════════════

describe('Scenario 2 — Error surfacing from HTTP error responses', () => {

  const FULL_PARAMS = {
    conceptId: 'c', conceptVersion: '1',
    conditionId: 'x', conditionVersion: '1', entity: 'e',
  };

  test('HTTP 400 with parameter_error → MemintelError.type === "parameter_error"', async () => {
    global.fetch = errorResponse(400, 'parameter_error', 'Bad request value');
    await expect(client.evaluateFull(FULL_PARAMS)).rejects.toMatchObject({
      type: 'parameter_error',
      message: 'Bad request value',
    });
  });

  test('HTTP 400 with parameter_error → error is instanceof MemintelError', async () => {
    global.fetch = errorResponse(400, 'parameter_error', 'Validation failed');
    let caught: unknown;
    try { await client.evaluateFull(FULL_PARAMS); } catch (e) { caught = e; }
    expect(caught).toBeInstanceOf(MemintelError);
    expect((caught as MemintelError).type).toBe('parameter_error');
  });

  test('HTTP 404 with not_found → MemintelError.type === "not_found"', async () => {
    global.fetch = errorResponse(404, 'not_found', 'Concept not found');
    await expect(client.evaluateFull(FULL_PARAMS)).rejects.toMatchObject({
      type: 'not_found',
      message: 'Concept not found',
    });
  });

  test('HTTP 409 with conflict → MemintelError.type === "conflict"', async () => {
    global.fetch = errorResponse(409, 'conflict', 'Version already exists');
    await expect(
      client.conditions.applyCalibration({ calibrationToken: 'tok' })
    ).rejects.toMatchObject({
      type: 'conflict',
      message: 'Version already exists',
    });
  });

  test('HTTP 422 with parameter_error → MemintelError.type === "parameter_error"', async () => {
    global.fetch = errorResponse(422, 'parameter_error', 'Unprocessable entity');
    await expect(client.evaluateFull(FULL_PARAMS)).rejects.toMatchObject({
      type: 'parameter_error',
      message: 'Unprocessable entity',
    });
  });

  test('HTTP 422 with semantic_error → MemintelError.type === "semantic_error"', async () => {
    global.fetch = errorResponse(422, 'semantic_error', 'Intent could not be resolved');
    await expect(
      client.tasks.create({ intent: 'i', entityScope: 'u', delivery: { type: 'webhook', endpoint: 'https://x.com' } })
    ).rejects.toMatchObject({
      type: 'semantic_error',
      message: 'Intent could not be resolved',
    });
  });

  test('HTTP 422 with action_binding_failed → MemintelError.type === "action_binding_failed"', async () => {
    global.fetch = errorResponse(422, 'action_binding_failed', 'No action matched');
    await expect(
      client.tasks.create({ intent: 'i', entityScope: 'u', delivery: { type: 'webhook', endpoint: 'https://x.com' } })
    ).rejects.toMatchObject({ type: 'action_binding_failed' });
  });

  test('Error response forwards location and suggestion fields', async () => {
    global.fetch = errorResponse(404, 'not_found', 'Task not found', {
      location: 'task_id',
      suggestion: 'Verify the task_id is correct',
    });
    let caught: MemintelError | undefined;
    try { await client.tasks.get('missing_task'); } catch (e) { caught = e as MemintelError; }
    expect(caught).toBeInstanceOf(MemintelError);
    expect(caught?.location).toBe('task_id');
    expect(caught?.suggestion).toBe('Verify the task_id is correct');
  });

  test('HTTP 400 on tasks.update with conflict → MemintelError.type === "conflict"', async () => {
    global.fetch = errorResponse(409, 'conflict', 'Task is deleted — cannot update');
    await expect(
      client.tasks.update('task_deleted', { status: 'active' })
    ).rejects.toMatchObject({ type: 'conflict' });
  });

  test('HTTP 400 on conditions.calibrate with parameter_error → typed error', async () => {
    global.fetch = errorResponse(400, 'parameter_error', 'Invalid token');
    await expect(
      client.conditions.calibrate({ conditionId: 'c', conditionVersion: '1' })
    ).rejects.toMatchObject({ type: 'parameter_error' });
  });

  test('Non-JSON error response falls back to execution_error type', async () => {
    // API returns non-JSON body on error (e.g. nginx 502)
    global.fetch = jest.fn().mockResolvedValue({
      ok: false, status: 502, statusText: 'Bad Gateway',
      headers: { get: () => null },
      json: () => Promise.reject(new SyntaxError('Unexpected token')),
    });
    let caught: MemintelError | undefined;
    try { await client.evaluateFull(FULL_PARAMS); } catch (e) { caught = e as MemintelError; }
    expect(caught).toBeInstanceOf(MemintelError);
    // Falls back to 'execution_error' as specified in http.ts
    expect(caught?.type).toBe('execution_error');
  });
});


// ══════════════════════════════════════════════════════════════════════════════
// Scenario 3 — dryRun parameter propagation
// ══════════════════════════════════════════════════════════════════════════════

describe('Scenario 3 — dryRun parameter reaches API as dry_run', () => {

  const BASE_FULL = {
    conceptId: 'c', conceptVersion: '1',
    conditionId: 'x', conditionVersion: '1', entity: 'e',
  };

  test('evaluateFull: dryRun=true → body.dry_run === true', async () => {
    global.fetch = ok(STUBS.fullPipeline);
    await client.evaluateFull({ ...BASE_FULL, dryRun: true });
    const body = JSON.parse(firstCall()[1].body as string);
    expect(body.dry_run).toBe(true);
  });

  test('evaluateFull: dryRun=false → body.dry_run === false', async () => {
    global.fetch = ok(STUBS.fullPipeline);
    await client.evaluateFull({ ...BASE_FULL, dryRun: false });
    const body = JSON.parse(firstCall()[1].body as string);
    expect(body.dry_run).toBe(false);
  });

  test('evaluateFull: dryRun omitted → dry_run NOT in body', async () => {
    global.fetch = ok(STUBS.fullPipeline);
    await client.evaluateFull(BASE_FULL);
    const body = JSON.parse(firstCall()[1].body as string);
    expect(body).not.toHaveProperty('dry_run');
  });

  test('executeBatch: dryRun=true → body.dry_run === true', async () => {
    global.fetch = ok(STUBS.batchResult);
    await client.executeBatch({ id: 'c', version: '1', entities: ['e1'], dryRun: true });
    const body = JSON.parse(firstCall()[1].body as string);
    expect(body.dry_run).toBe(true);
  });

  test('executeBatch: dryRun=false → body.dry_run === false', async () => {
    global.fetch = ok(STUBS.batchResult);
    await client.executeBatch({ id: 'c', version: '1', entities: ['e1'], dryRun: false });
    const body = JSON.parse(firstCall()[1].body as string);
    expect(body.dry_run).toBe(false);
  });

  test('executeBatch: dryRun omitted → dry_run NOT in body', async () => {
    global.fetch = ok(STUBS.batchResult);
    await client.executeBatch({ id: 'c', version: '1', entities: ['e1'] });
    const body = JSON.parse(firstCall()[1].body as string);
    expect(body).not.toHaveProperty('dry_run');
  });

  test('execute: dryRun=true → body.dry_run === true', async () => {
    global.fetch = ok(STUBS.result);
    await client.execute({ id: 'c', version: '1', entity: 'e', dryRun: true });
    const body = JSON.parse(firstCall()[1].body as string);
    expect(body.dry_run).toBe(true);
  });

  test('tasks.create: dryRun=true → body.dry_run === true, returns DryRunResult', async () => {
    // DryRunResult shape (not a Task)
    const dryRunStub = { concept: {}, action_id: 'a1', action_version: '1', validation: { valid: true, errors: [], warnings: [] }, would_trigger: null };
    global.fetch = ok(dryRunStub);
    const result = await client.tasks.create({
      intent: 'i', entityScope: 'u',
      delivery: { type: 'webhook', endpoint: 'https://x.com' },
      dryRun: true,
    });
    const body = JSON.parse(firstCall()[1].body as string);
    expect(body.dry_run).toBe(true);
    // Result is a DryRunResult (has actionId, not taskId)
    expect((result as Record<string, unknown>).taskId).toBeUndefined();
  });

  test('tasks.create: dryRun omitted → dry_run NOT in body', async () => {
    global.fetch = ok(STUBS.task);
    await client.tasks.create({ intent: 'i', entityScope: 'u', delivery: { type: 'webhook', endpoint: 'https://x.com' } });
    const body = JSON.parse(firstCall()[1].body as string);
    expect(body).not.toHaveProperty('dry_run');
  });

  test('evaluateFull: dryRun=true is reflected in response.dryRun (camelCase)', async () => {
    global.fetch = ok({ ...STUBS.fullPipeline, dry_run: true });
    const result = await client.evaluateFull({ ...BASE_FULL, dryRun: true });
    // Response field dry_run → dryRun in mapped result
    expect(result.dryRun).toBe(true);
    expect((result as Record<string, unknown>).dry_run).toBeUndefined();
  });
});


// ══════════════════════════════════════════════════════════════════════════════
// Scenario 4 — Pagination cursor forwarded correctly
// ══════════════════════════════════════════════════════════════════════════════

describe('Scenario 4 — Pagination cursor', () => {

  function getQueryParams(url: string): URLSearchParams {
    return new URL(url).searchParams;
  }

  test('tasks.list: cursor appears as query param', async () => {
    global.fetch = ok(STUBS.taskList);
    await client.tasks.list({ cursor: 'cursor_page2' });
    const [url] = firstCall();
    expect(getQueryParams(url).get('cursor')).toBe('cursor_page2');
  });

  test('tasks.list: limit appears as query param', async () => {
    global.fetch = ok(STUBS.taskList);
    await client.tasks.list({ limit: 5 });
    const [url] = firstCall();
    expect(getQueryParams(url).get('limit')).toBe('5');
  });

  test('tasks.list: status and cursor both forwarded', async () => {
    global.fetch = ok(STUBS.taskList);
    await client.tasks.list({ status: 'active', limit: 10, cursor: 'tok_abc' });
    const [url] = firstCall();
    const params = getQueryParams(url);
    expect(params.get('status')).toBe('active');
    expect(params.get('limit')).toBe('10');
    expect(params.get('cursor')).toBe('tok_abc');
  });

  test('tasks.list: no cursor param when cursor not provided', async () => {
    global.fetch = ok(STUBS.taskList);
    await client.tasks.list({ limit: 20 });
    const [url] = firstCall();
    expect(getQueryParams(url).has('cursor')).toBe(false);
  });

  test('tasks.list: nextCursor in response is accessible as camelCase', async () => {
    const page1 = { items: [], has_more: true, next_cursor: 'page2_cursor', total_count: 50 };
    global.fetch = ok(page1);
    const result = await client.tasks.list();
    expect(result.hasMore).toBe(true);
    expect(result.nextCursor).toBe('page2_cursor');
    expect(result.totalCount).toBe(50);
    // snake_case NOT present
    expect((result as Record<string, unknown>).next_cursor).toBeUndefined();
    expect((result as Record<string, unknown>).has_more).toBeUndefined();
  });

  test('tasks.list: nextCursor from first response can be used as cursor in second call', async () => {
    const page1 = { items: [], has_more: true, next_cursor: 'cursor_after_task_50', total_count: 100 };
    const page2 = { items: [], has_more: false, next_cursor: null, total_count: 100 };

    global.fetch = jest.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, statusText: 'OK', headers: { get: () => null }, json: () => Promise.resolve(page1) })
      .mockResolvedValueOnce({ ok: true, status: 200, statusText: 'OK', headers: { get: () => null }, json: () => Promise.resolve(page2) });

    // First call — no cursor
    const result1 = await client.tasks.list({ limit: 50 });
    expect(result1.hasMore).toBe(true);
    expect(result1.nextCursor).toBe('cursor_after_task_50');

    // Second call — pass nextCursor as cursor
    const result2 = await client.tasks.list({ limit: 50, cursor: result1.nextCursor! });
    expect(result2.hasMore).toBe(false);
    expect(result2.nextCursor).toBeNull();

    // Verify second call URL has the cursor from first page
    const secondCallUrl = (global.fetch as jest.Mock).mock.calls[1][0] as string;
    expect(getQueryParams(secondCallUrl).get('cursor')).toBe('cursor_after_task_50');
  });

  test('registry.list: cursor forwarded as query param', async () => {
    global.fetch = ok(STUBS.searchResult);
    await client.registry.list({ cursor: 'reg_cursor_xyz', namespace: 'org' });
    const [url] = firstCall();
    const params = getQueryParams(url);
    expect(params.get('cursor')).toBe('reg_cursor_xyz');
    expect(params.get('namespace')).toBe('org');
  });

  test('features.search: cursor forwarded as query param', async () => {
    global.fetch = ok(STUBS.featureSearch);
    await client.features.search({ cursor: 'feat_cursor_abc', limit: 15 });
    const [url] = firstCall();
    const params = getQueryParams(url);
    expect(params.get('cursor')).toBe('feat_cursor_abc');
    expect(params.get('limit')).toBe('15');
  });

  test('actions.list: cursor forwarded as query param', async () => {
    global.fetch = ok(STUBS.actionList);
    await client.actions.list({ cursor: 'act_cursor_def', limit: 25 });
    const [url] = firstCall();
    const params = getQueryParams(url);
    expect(params.get('cursor')).toBe('act_cursor_def');
    expect(params.get('limit')).toBe('25');
  });
});


// ══════════════════════════════════════════════════════════════════════════════
// Scenario 5 — Timeout behaviour
// ══════════════════════════════════════════════════════════════════════════════

describe('Scenario 5 — Timeout configuration and rejection', () => {

  const BASE_FULL = {
    conceptId: 'c', conceptVersion: '1',
    conditionId: 'x', conditionVersion: '1', entity: 'e',
  };

  test('SDK attaches AbortSignal to every fetch call (timeout is configured)', async () => {
    global.fetch = ok(STUBS.fullPipeline);
    await client.evaluateFull(BASE_FULL);
    const [, init] = firstCall();
    // AbortSignal.timeout() sets a signal on the request options
    expect(init.signal).toBeDefined();
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  test('SDK attaches AbortSignal to GET requests', async () => {
    global.fetch = ok(STUBS.taskList);
    await client.tasks.list();
    const [, init] = firstCall();
    expect(init.signal).toBeDefined();
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  test('SDK attaches AbortSignal to PATCH and DELETE requests', async () => {
    // PATCH
    global.fetch = ok(STUBS.task);
    await client.tasks.update('t', { status: 'paused' });
    expect(firstCall()[1].signal).toBeInstanceOf(AbortSignal);

    // DELETE
    (global.fetch as jest.Mock).mockClear();
    global.fetch = ok(STUBS.task);
    await client.tasks.delete('t');
    expect(firstCall()[1].signal).toBeInstanceOf(AbortSignal);
  });

  test('custom timeout is wired through from constructor config', async () => {
    // A client with timeout=1ms; fetch mock succeeds immediately so no actual timeout fires.
    const fastClient = new Memintel({ apiKey: 'k', timeout: 1 });
    global.fetch = ok(STUBS.result);
    await fastClient.execute({ id: 'c', version: '1', entity: 'e' });
    const [, init] = firstCall();
    // Signal is present and is an AbortSignal (the exact timeout value is internal)
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  test('when fetch throws AbortError (timeout fires), the Promise rejects — no unhandled rejection', async () => {
    // Simulate what happens when AbortSignal.timeout() fires:
    // fetch throws DOMException with name='TimeoutError' in Node 18+.
    const timeoutErr = Object.assign(new Error('The operation was aborted'), {
      name: 'TimeoutError',
    });
    global.fetch = jest.fn().mockRejectedValue(timeoutErr);

    let thrown: unknown;
    try {
      await client.evaluateFull(BASE_FULL);
    } catch (e) {
      thrown = e;
    }

    // The Promise DOES reject — it is not an unhandled rejection
    expect(thrown).toBeDefined();
    // The error is a clean Error, not an unhandled rejection (test receiving it proves it)
    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).name).toBe('TimeoutError');
  });

  test('GAP — timeout AbortError is NOT wrapped as MemintelError', async () => {
    /**
     * DOCUMENTED GAP: The SDK does not wrap AbortError/TimeoutError from fetch
     * into a MemintelError. When the configured timeout fires:
     *
     *   http.ts:  const res = await fetch(url, { signal: AbortSignal.timeout(timeout) });
     *             return this.handleResponse(res);   // ← never reached on timeout
     *
     * The fetch() call throws directly, bypassing handleResponse(). No MemintelError
     * is constructed. The caller's `catch (e) { if (e instanceof MemintelError) {...} }`
     * branch is never entered for timeout errors.
     *
     * Fix: wrap the fetch() call in try/catch and re-throw as:
     *   throw new MemintelError('execution_timeout', 'Request timed out after ${timeout}ms')
     *
     * This aligns with the TypeScript error handling pattern in ts-instructions.md
     * (see: case 'execution_timeout': start async job instead).
     */
    const timeoutErr = Object.assign(new Error('The operation was aborted'), {
      name: 'TimeoutError',
    });
    global.fetch = jest.fn().mockRejectedValue(timeoutErr);

    let thrown: unknown;
    try {
      await client.evaluateFull(BASE_FULL);
    } catch (e) {
      thrown = e;
    }

    // GAP: thrown error is NOT a MemintelError
    expect(thrown instanceof MemintelError).toBe(false);

    // Confirm the error IS identifiable (has name 'TimeoutError'), so it's
    // a clean rejection — just not a typed MemintelError. Callers who check
    // `err instanceof MemintelError` before branching will miss timeout errors.
    expect((thrown as Error).name).toBe('TimeoutError');
  });

  test('AbortError variant (name=AbortError) also propagates without wrapping', async () => {
    // In some environments the abort error name is 'AbortError' rather than 'TimeoutError'
    const abortErr = Object.assign(new Error('The operation was aborted'), {
      name: 'AbortError',
    });
    global.fetch = jest.fn().mockRejectedValue(abortErr);

    let thrown: unknown;
    try { await client.execute({ id: 'c', version: '1', entity: 'e' }); } catch (e) { thrown = e; }

    expect(thrown).toBeInstanceOf(Error);
    expect(thrown instanceof MemintelError).toBe(false);
    expect((thrown as Error).name).toBe('AbortError');
  });
});
