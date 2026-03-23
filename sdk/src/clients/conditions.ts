import { HttpClient } from '../http';
import {
  ApplyCalibrationParams,
  ApplyCalibrationResult,
  CalibrateParams,
  CalibrationResult,
  ConditionDefinition,
  ConditionExplanation,
} from '../types';
import {
  mapApplyCalibrationResult,
  mapCalibrationResult,
  mapConditionDefinition,
  mapConditionExplanation,
} from '../mappers';

export class ConditionsClient {
  constructor(private http: HttpClient) {}

  async get(id: string, version: string): Promise<ConditionDefinition> {
    const raw = await this.http.get(`/conditions/${id}`, { version });
    return mapConditionDefinition(raw);
  }

  async explain(params: {
    conditionId: string;
    conditionVersion: string;
    timestamp?: string;
  }): Promise<ConditionExplanation> {
    const body: Record<string, unknown> = {
      condition_id:      params.conditionId,
      condition_version: params.conditionVersion,
    };
    if (params.timestamp !== undefined) body.timestamp = params.timestamp;
    const raw = await this.http.post('/conditions/explain', body);
    return mapConditionExplanation(raw);
  }

  async calibrate(params: CalibrateParams): Promise<CalibrationResult> {
    const body: Record<string, unknown> = {
      condition_id:      params.conditionId,
      condition_version: params.conditionVersion,
    };
    if (params.feedbackType !== undefined)      body.feedback_type      = params.feedbackType;
    if (params.feedbackDirection !== undefined) body.feedback_direction = params.feedbackDirection;
    if (params.target !== undefined)            body.target             = { alerts_per_day: params.target.alertsPerDay };
    if (params.context !== undefined)           body.context            = params.context;
    const raw = await this.http.post('/conditions/calibrate', body);
    return mapCalibrationResult(raw);
  }

  async applyCalibration(params: ApplyCalibrationParams): Promise<ApplyCalibrationResult> {
    const body: Record<string, unknown> = {
      calibration_token: params.calibrationToken,
    };
    if (params.newVersion !== undefined) body.new_version = params.newVersion;
    const raw = await this.http.post('/conditions/apply-calibration', body);
    return mapApplyCalibrationResult(raw);
  }
}
