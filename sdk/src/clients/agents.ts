import { HttpClient } from '../http';
import {
  AgentDefineConditionParams,
  AgentDefineParams,
  AgentDefineResponse,
  AgentQueryParams,
  AgentQueryResponse,
} from '../types';
import { mapAgentDefineResponse, mapAgentQueryResponse } from '../mappers';

export class AgentsClient {
  constructor(private http: HttpClient) {}

  async query(params: AgentQueryParams): Promise<AgentQueryResponse> {
    const body: Record<string, unknown> = { query: params.query };
    if (params.context !== undefined) body.context = params.context;
    const raw = await this.http.post('/agents/query', body);
    return mapAgentQueryResponse(raw);
  }

  async define(params: AgentDefineParams): Promise<AgentDefineResponse> {
    const body: Record<string, unknown> = { intent: params.intent };
    if (params.context !== undefined) body.context = params.context;
    const raw = await this.http.post('/agents/define', body);
    return mapAgentDefineResponse(raw);
  }

  async defineCondition(params: AgentDefineConditionParams): Promise<AgentDefineResponse> {
    const body: Record<string, unknown> = {
      intent:          params.intent,
      concept_id:      params.conceptId,
      concept_version: params.conceptVersion,
    };
    if (params.context !== undefined) body.context = params.context;
    const raw = await this.http.post('/agents/define-condition', body);
    return mapAgentDefineResponse(raw);
  }
}
