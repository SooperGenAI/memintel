import { HttpClient } from '../http';
import { FeatureSearchResult, RegisteredFeature } from '../types';
import { mapFeatureSearchResult, mapRegisteredFeature } from '../mappers';

export class FeaturesClient {
  constructor(private http: HttpClient) {}

  async search(params?: {
    query?: string;
    type?: string;
    limit?: number;
    cursor?: string;
  }): Promise<FeatureSearchResult> {
    const raw = await this.http.get('/registry/features', params as Record<string, string | number | boolean | undefined>);
    return mapFeatureSearchResult(raw);
  }

  async get(id: string): Promise<RegisteredFeature> {
    const raw = await this.http.get(`/registry/features/${id}`);
    return mapRegisteredFeature(raw);
  }
}
