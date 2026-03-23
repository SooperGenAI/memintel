import { HttpClient } from '../http';
import { Action, ActionResult, ActionTriggerParams } from '../types';
import { mapAction, mapActionResult } from '../mappers';

export class ActionsClient {
  constructor(private http: HttpClient) {}

  async list(params?: { limit?: number; cursor?: string }): Promise<Action[]> {
    const raw = await this.http.get('/actions', params as Record<string, string | number | boolean | undefined>) as { items?: unknown[] } | unknown[];
    const items = Array.isArray(raw) ? raw : (raw as { items?: unknown[] }).items ?? [];
    return items.map(mapAction);
  }

  async get(id: string): Promise<Action> {
    const raw = await this.http.get(`/actions/${id}`);
    return mapAction(raw);
  }

  async trigger(id: string, params: ActionTriggerParams): Promise<ActionResult> {
    const body: Record<string, unknown> = { entity: params.entity };
    if (params.payload !== undefined) body.payload = params.payload;
    const raw = await this.http.post(`/actions/${id}/trigger`, body);
    return mapActionResult(raw);
  }
}
