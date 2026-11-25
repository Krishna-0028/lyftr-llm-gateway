import type { ChatResponse, HealthResponse } from './types';

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
).replace(/\/+$/, '');

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

function formatErrorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string' && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail) && detail.length > 0) {
    const messages = detail
      .map((item) => {
        if (item && typeof item === 'object' && 'msg' in item) {
          const loc = Array.isArray((item as { loc?: unknown[] }).loc)
            ? (item as { loc: unknown[] }).loc.slice(1).join('.')
            : '';
          const msg = String((item as { msg: unknown }).msg);
          return loc ? `${loc}: ${msg}` : msg;
        }
        return typeof item === 'string' ? item : JSON.stringify(item);
      })
      .filter(Boolean);
    if (messages.length > 0) {
      return messages.join('; ');
    }
  }
  return fallback;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = formatErrorDetail(body?.detail, `Health check failed (${response.status})`);
    throw new ApiError(response.status, detail);
  }
  return body as HealthResponse;
}

export interface SendChatParams {
  sessionId: string;
  message: string;
  systemPrompt?: string;
  simulateFailure?: string;
}

export async function sendChatMessage({
  sessionId,
  message,
  systemPrompt,
  simulateFailure,
}: SendChatParams): Promise<ChatResponse> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  };
  if (simulateFailure && simulateFailure.trim()) {
    headers['X-Simulate-Failure'] = simulateFailure.trim();
  }

  const payload: Record<string, string> = {
    session_id: sessionId,
    message,
  };
  if (systemPrompt && systemPrompt.trim()) {
    payload.system_prompt = systemPrompt.trim();
  }

  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
  });

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = formatErrorDetail(
      body?.detail,
      `Request failed with HTTP ${response.status}`,
    );
    throw new ApiError(response.status, detail);
  }

  return body as ChatResponse;
}
