import { MemintelError } from '../error';
import { HttpClient } from '../http';
import { FeedbackParams, FeedbackResponse, FeedbackValue } from '../types';
import { mapFeedbackResponse } from '../mappers';

const VALID_FEEDBACK_VALUES: FeedbackValue[] = ['false_positive', 'false_negative', 'correct'];

export class FeedbackClient {
  constructor(private http: HttpClient) {}

  async submit(params: FeedbackParams): Promise<FeedbackResponse> {
    if (!VALID_FEEDBACK_VALUES.includes(params.feedbackType)) {
      throw new MemintelError(
        'parameter_error',
        `Invalid feedback value '${params.feedbackType}'. Valid values: false_positive, false_negative, correct`,
      );
    }

    const body: Record<string, unknown> = {
      condition_id:      params.conditionId,
      condition_version: params.conditionVersion,
      entity:            params.entity,
      timestamp:         params.timestamp,
      feedback:          params.feedbackType,
    };
    if (params.note !== undefined) body.note = params.note;

    const raw = await this.http.post('/feedback/decision', body);
    return mapFeedbackResponse(raw);
  }
}
