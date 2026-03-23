import { HttpClient } from '../http';
import { DecisionExplainParams, DecisionExplanation } from '../types';
import { mapDecisionExplanation } from '../mappers';

export class DecisionsClient {
  constructor(private http: HttpClient) {}

  async explain(params: DecisionExplainParams): Promise<DecisionExplanation> {
    const body = {
      condition_id:      params.conditionId,
      condition_version: params.conditionVersion,
      entity:            params.entity,
      timestamp:         params.timestamp,
    };
    const raw = await this.http.post('/decisions/explain', body);
    return mapDecisionExplanation(raw);
  }
}
