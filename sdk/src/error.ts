import { ErrorType } from './types';

export class MemintelError extends Error {
  type: ErrorType;
  location?: string;
  suggestion?: string;
  retryAfterSeconds?: number;

  constructor(
    type: ErrorType,
    message: string,
    options?: {
      location?: string;
      suggestion?: string;
      retryAfterSeconds?: number;
    },
  ) {
    super(message);
    this.name = 'MemintelError';
    this.type = type;
    this.location = options?.location;
    this.suggestion = options?.suggestion;
    this.retryAfterSeconds = options?.retryAfterSeconds;

    // Restore prototype chain (required when extending built-in classes in TS)
    Object.setPrototypeOf(this, MemintelError.prototype);
  }
}
