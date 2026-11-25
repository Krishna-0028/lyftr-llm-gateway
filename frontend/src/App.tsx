import { useEffect, useId, useRef, useState } from 'react';
import { ApiError, fetchHealth, sendChatMessage } from './api';
import type { ChatEntry, HealthResponse } from './types';

function createSessionId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `s-${crypto.randomUUID().slice(0, 8)}`;
  }
  return `s-${Math.random().toString(36).slice(2, 10)}`;
}

const SAMPLE_PROMPTS = [
  'What is the capital of France?',
  'Explain key failover in 2 sentences.',
];

export default function App() {
  const [sessionId, setSessionId] = useState<string>(() => createSessionId());
  const [systemPrompt, setSystemPrompt] = useState('');
  const [simulateFailure, setSimulateFailure] = useState('');
  const [messageInput, setMessageInput] = useState('');
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [inputError, setInputError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const chatInputId = useId();
  const chatHintId = useId();
  const systemPromptId = useId();
  const simulateId = useId();
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let active = true;
    fetchHealth()
      .then((res) => {
        if (active) {
          setHealth(res);
          setHealthError(null);
        }
      })
      .catch((err: unknown) => {
        if (active) {
          setHealth(null);
          setHealthError(err instanceof Error ? err.message : 'Backend unavailable');
        }
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [entries, isSending]);

  function handleNewSession() {
    setSessionId(createSessionId());
    setEntries([]);
    setInputError(null);
    inputRef.current?.focus();
  }

  async function submitMessage(rawText: string) {
    const trimmed = rawText.trim();
    if (!trimmed) {
      setInputError('Please enter a message before sending.');
      inputRef.current?.focus();
      return;
    }

    setInputError(null);
    setMessageInput('');
    setIsSending(true);

    const userEntry: ChatEntry = {
      id: `u-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      kind: 'user',
      content: trimmed,
    };
    setEntries((prev) => [...prev, userEntry]);

    try {
      const response = await sendChatMessage({
        sessionId,
        message: trimmed,
        systemPrompt,
        simulateFailure,
      });

      const assistantEntry: ChatEntry = {
        id: `a-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        kind: 'assistant',
        content: response.reply,
        metadata: response.metadata,
      };
      setEntries((prev) => [...prev, assistantEntry]);
    } catch (err: unknown) {
      const status = err instanceof ApiError ? err.status : undefined;
      const msg =
        err instanceof Error
          ? err.message
          : 'Unexpected error while contacting the gateway.';
      const errorEntry: ChatEntry = {
        id: `e-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        kind: 'error',
        status,
        content: status ? `Error (${status}): ${msg}` : `Error: ${msg}`,
      };
      setEntries((prev) => [...prev, errorEntry]);
    } finally {
      setIsSending(false);
    }
  }

  function handleFormSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSending) {
      return;
    }
    void submitMessage(messageInput);
  }

  function handleTextareaKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (event.nativeEvent.isComposing || event.keyCode === 229) {
        return;
      }
      event.currentTarget.form?.requestSubmit();
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="header-top">
          <div className="header-title-group">
            <h1>LLM Gateway Chat</h1>
            <p className="header-subtitle">
              Session <code>{sessionId}</code> &middot; Key failover &amp; response caching
            </p>
          </div>
          <div className="header-actions">
            <span className="status-pill" aria-live="polite">
              <span
                className="status-dot"
                data-healthy={health ? 'true' : 'false'}
                aria-hidden="true"
              />
              {health
                ? `${health.keys_configured} keys · cache ${health.cache_enabled ? 'on' : 'off'}`
                : healthError || 'Checking health...'}
            </span>
            <button
              type="button"
              className="btn-secondary"
              onClick={handleNewSession}
            >
              New session
            </button>
          </div>
        </div>

        <div className="session-controls">
          <div className="control-field">
            <label htmlFor={systemPromptId}>System prompt (optional)</label>
            <input
              id={systemPromptId}
              data-testid="system-prompt-input"
              type="text"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="e.g. Answer concisely in one sentence."
            />
          </div>

          <div className="control-field">
            <label htmlFor={simulateId}>Simulate failure header (demo)</label>
            <select
              id={simulateId}
              value={simulateFailure}
              onChange={(e) => setSimulateFailure(e.target.value)}
            >
              <option value="">Off (normal routing)</option>
              <option value="key_1=429">key_1=429 (fail over to key_2)</option>
              <option value="key_1=500">key_1=500 (fail over to key_2)</option>
              <option value="*=429">*=429 (all keys fail → 503)</option>
            </select>
          </div>
        </div>
      </header>

      <main className="chat-main">
        <div className="chat-transcript" aria-live="polite">
          {entries.length === 0 ? (
            <div className="empty-state">
              <h2>Start a multi-turn conversation</h2>
              <p>
                Every assistant reply displays whether it came from{' '}
                <code>LIVE_INFERENCE</code> or <code>LLM_CACHE</code>, along with
                model, latency, and per-request token counts.
              </p>
              <div className="sample-prompts">
                {SAMPLE_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    className="sample-chip"
                    onClick={() => void submitMessage(prompt)}
                    disabled={isSending}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            entries.map((entry) => {
              if (entry.kind === 'user') {
                return (
                  <div key={entry.id} className="message-row" data-role="user">
                    <div className="message-bubble">{entry.content}</div>
                  </div>
                );
              }

              if (entry.kind === 'error') {
                return (
                  <div key={entry.id} className="error-banner" role="alert">
                    {entry.content}
                  </div>
                );
              }

              const { metadata } = entry;
              return (
                <div
                  key={entry.id}
                  className="message-row"
                  data-role="assistant"
                  data-testid="assistant-message"
                >
                  <div className="message-bubble">{entry.content}</div>
                  <div className="meta-bar">
                    <span
                      className="meta-badge"
                      data-testid="meta-source"
                      data-source={metadata.resolution_source}
                    >
                      {metadata.resolution_source}
                    </span>
                    <span className="meta-badge" data-testid="meta-model">
                      {metadata.model}
                    </span>
                    <span className="meta-badge" data-testid="meta-latency">
                      {metadata.latency_ms} ms
                    </span>
                    <span className="meta-badge">
                      In:{' '}
                      <span data-testid="meta-input-tokens">
                        {metadata.input_tokens}
                      </span>
                    </span>
                    <span className="meta-badge">
                      Out:{' '}
                      <span data-testid="meta-output-tokens">
                        {metadata.output_tokens}
                      </span>
                    </span>
                    {metadata.key_id && (
                      <span className="meta-badge">Key: {metadata.key_id}</span>
                    )}
                    {metadata.fallback_attempts > 0 && (
                      <span className="meta-badge">
                        Fallbacks: {metadata.fallback_attempts}
                      </span>
                    )}
                  </div>
                </div>
              );
            })
          )}
          {isSending && (
            <div className="message-row" data-role="assistant">
              <div className="message-bubble">Thinking…</div>
            </div>
          )}
          <div ref={transcriptEndRef} />
        </div>

        <form className="composer-form" onSubmit={handleFormSubmit} noValidate>
          <label htmlFor={chatInputId} className="visually-hidden">
            Message
          </label>
          <div className="composer-row">
            <textarea
              ref={inputRef}
              id={chatInputId}
              name="message"
              data-testid="chat-input"
              className="composer-textarea"
              rows={2}
              value={messageInput}
              onChange={(e) => {
                setMessageInput(e.target.value);
                if (inputError) {
                  setInputError(null);
                }
              }}
              onKeyDown={handleTextareaKeyDown}
              placeholder="Ask anything… (Enter to send, Shift+Enter for newline)"
              aria-invalid={inputError ? 'true' : undefined}
              aria-describedby={chatHintId}
              required
            />
            <button
              type="submit"
              data-testid="send-button"
              className="btn-primary"
              disabled={isSending}
            >
              {isSending ? 'Sending…' : 'Send'}
            </button>
          </div>
          <div className="composer-footer">
            <span
              id={chatHintId}
              role={inputError ? 'alert' : undefined}
              className={inputError ? 'validation-error' : undefined}
            >
              {inputError || 'Press Enter to send · Shift+Enter for a new line'}
            </span>
            <span>{messageInput.trim().length} / 4000</span>
          </div>
        </form>
      </main>
    </div>
  );
}
