import {ExternalLink, KeyRound, LoaderCircle, X} from 'lucide-react';
import {useState, type FormEvent} from 'react';
import {api} from './api';

type Props = {
  /** True when a key already works and the user is just replacing it. */
  replacing: boolean;
  onConnected: () => void;
  onClose?: () => void;
};

export function KeySetup({replacing, onConnected, onClose}: Props) {
  const [key, setKey] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function connect(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError('');
    try {
      await api('/settings/gemini', {method: 'POST', body: JSON.stringify({api_key: key.trim()})});
      setKey('');
      onConnected();
    } catch (err) {
      setError(`${(err as Error).message} Check that you copied the whole key, then try again.`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="key-setup" aria-labelledby="key-setup-title">
      <div className="key-setup-copy">
        <span className="key-icon" aria-hidden="true">
          <KeyRound size={22} />
        </span>
        <div>
          <h2 id="key-setup-title">{replacing ? 'Change your AI key' : 'One-time setup: connect the AI'}</h2>
          <p>The editor uses Google Gemini to watch your clips and plan the cut. You need a free key from Google:</p>
          <ol className="key-steps">
            <li>
              Open{' '}
              <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer">
                Google AI Studio <ExternalLink size={13} aria-hidden="true" />
                <span className="sr-only">(opens in a new tab)</span>
              </a>{' '}
              and sign in.
            </li>
            <li>Choose “Create API key” and copy it.</li>
            <li>Paste it here. It is saved on this computer only and never shown again.</li>
          </ol>
          <p className="fine-print">
            The free tier has daily limits, and Google may use free-tier content to improve its products. Use footage you are comfortable
            sharing.
          </p>
        </div>
      </div>
      <form onSubmit={connect} className="key-form">
        {onClose && (
          <button type="button" className="icon-button key-close" aria-label="Close AI settings" onClick={onClose}>
            <X size={18} />
          </button>
        )}
        <label htmlFor="gemini-key">Your Gemini key</label>
        <input
          id="gemini-key"
          type="password"
          autoComplete="off"
          spellCheck={false}
          value={key}
          onChange={e => setKey(e.target.value)}
          placeholder="Paste key here"
          required
          minLength={20}
          maxLength={256}
          aria-describedby={error ? 'gemini-key-error' : undefined}
        />
        {error && (
          <p id="gemini-key-error" className="field-error" role="alert">
            {error}
          </p>
        )}
        <button className="primary" disabled={busy || key.trim().length < 20}>
          {busy && <LoaderCircle className="spin" size={18} />}
          {busy ? 'Checking key…' : 'Connect'}
        </button>
      </form>
    </section>
  );
}
