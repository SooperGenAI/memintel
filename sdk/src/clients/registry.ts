import { HttpClient } from '../http';
import { SearchResult, VersionListResult } from '../types';
import { mapSearchResult, mapVersionListResult } from '../mappers';

export class RegistryClient {
  constructor(private http: HttpClient) {}

  async list(params?: {
    namespace?: string;
    type?: string;
    limit?: number;
    cursor?: string;
  }): Promise<SearchResult> {
    const raw = await this.http.get('/registry/definitions', params as Record<string, string | number | boolean | undefined>);
    return mapSearchResult(raw);
  }

  async search(params: {
    query: string;
    type?: string;
    namespace?: string;
    limit?: number;
  }): Promise<SearchResult> {
    const raw = await this.http.get('/registry/search', params as Record<string, string | number | boolean | undefined>);
    return mapSearchResult(raw);
  }

  async versions(id: string): Promise<VersionListResult> {
    const raw = await this.http.get(`/registry/definitions/${id}/versions`);
    return mapVersionListResult(raw);
  }
}
