import { ErrorType } from './types';
import { MemintelError } from './error';

export class HttpClient {
  private baseUrl: string;
  private apiKey: string;
  private timeout: number;

  constructor(baseUrl: string, apiKey: string, timeout: number) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
    this.timeout = timeout;
  }

  async post<T>(
    path: string,
    body: unknown,
    extraHeaders?: Record<string, string>,
  ): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.apiKey,
        ...extraHeaders,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeout),
    });
    return this.handleResponse<T>(res);
  }

  async get<T>(
    path: string,
    params?: Record<string, string | number | boolean | undefined>,
  ): Promise<T> {
    const url = new URL(`${this.baseUrl}${path}`);
    if (params) {
      for (const [key, value] of Object.entries(params)) {
        if (value !== undefined) {
          url.searchParams.set(key, String(value));
        }
      }
    }
    const res = await fetch(url.toString(), {
      method: 'GET',
      headers: { 'X-API-Key': this.apiKey },
      signal: AbortSignal.timeout(this.timeout),
    });
    return this.handleResponse<T>(res);
  }

  async patch<T>(
    path: string,
    body: unknown,
    extraHeaders?: Record<string, string>,
  ): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.apiKey,
        ...extraHeaders,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeout),
    });
    return this.handleResponse<T>(res);
  }

  async delete<T>(
    path: string,
    extraHeaders?: Record<string, string>,
  ): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: 'DELETE',
      headers: {
        'X-API-Key': this.apiKey,
        ...extraHeaders,
      },
      signal: AbortSignal.timeout(this.timeout),
    });
    return this.handleResponse<T>(res);
  }

  private async handleResponse<T>(res: Response): Promise<T> {
    if (res.ok) {
      return res.json() as Promise<T>;
    }

    const retryAfterSeconds = res.headers.get('Retry-After')
      ? Number(res.headers.get('Retry-After'))
      : undefined;

    type ErrPayload = { error?: { type?: string; message?: string; location?: string; suggestion?: string } };
    let errPayload: ErrPayload;
    try {
      errPayload = await res.json() as ErrPayload;
    } catch {
      errPayload = { error: { type: 'execution_error', message: res.statusText } };
    }

    const errType = (errPayload?.error?.type ?? 'execution_error') as ErrorType;
    const errMessage = errPayload?.error?.message ?? `HTTP ${res.status}`;

    throw new MemintelError(errType, errMessage, {
      location: errPayload?.error?.location,
      suggestion: errPayload?.error?.suggestion,
      retryAfterSeconds,
    });
  }
}
