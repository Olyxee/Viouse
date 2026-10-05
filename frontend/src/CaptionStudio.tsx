import {useId, type CSSProperties} from 'react';

export type CaptionSettings = {
  mode: 'off' | 'auto' | 'custom';
  style: 'pop' | 'karaoke' | 'comic' | 'behind';
  position: 'top' | 'center' | 'bottom';
  color: string;
  lines: {start: number; end: number; text: string}[];
};

export const defaultCaptions: CaptionSettings = {mode: 'auto', style: 'pop', position: 'bottom', color: '#DFFF52', lines: []};

const looks = [
  ['pop', 'Pop', 'Big words with a quick bounce'],
  ['karaoke', 'Karaoke', 'Colour sweeps as words are said'],
  ['comic', 'Comic', 'Tilted type, heavy outline'],
  ['behind', 'Behind you', 'Words sit behind the person'],
] as const;

export function captionProblem(value: CaptionSettings, duration: number): string {
  if (value.mode !== 'custom') return '';
  if (value.lines.length === 0) return 'Add at least one line of your own words.';
  const bad = value.lines.some(
    (line, i) =>
      !line.text.trim() ||
      line.end - line.start < 0.15 ||
      line.start < 0 ||
      line.end > duration + 0.035 ||
      (i > 0 && line.start < value.lines[i - 1].end),
  );
  return bad ? 'Check your lines: each needs words, lines must not overlap, and they must fit inside the video.' : '';
}

type Props = {
  value: CaptionSettings;
  onChange: (value: CaptionSettings) => void;
  duration: number;
  disabled?: boolean;
  onExport?: () => void;
  exportLabel?: string;
};

export function CaptionStudio({value, onChange, duration, disabled = false, onExport, exportLabel = 'Make a new version with these subtitles'}: Props) {
  const id = useId();
  const update = (patch: Partial<CaptionSettings>) => onChange({...value, ...patch});
  const problem = captionProblem(value, duration);
  const on = value.mode !== 'off';

  const addLine = () => {
    const start = value.lines.at(-1)?.end || 0;
    update({lines: [...value.lines, {start, end: Math.min(duration, start + 3), text: ''}]});
  };
  const setLine = (index: number, patch: Partial<CaptionSettings['lines'][number]>) =>
    update({lines: value.lines.map((line, n) => (n === index ? {...line, ...patch} : line))});

  return (
    <fieldset className="caption-studio" disabled={disabled}>
      <legend className="field-label">Subtitles</legend>

      <div className="segmented" role="radiogroup" aria-label="Subtitles on or off">
        <button type="button" role="radio" aria-checked={on} className={on ? 'selected' : ''} onClick={() => update({mode: value.mode === 'off' ? 'auto' : value.mode})}>
          On
        </button>
        <button type="button" role="radio" aria-checked={!on} className={!on ? 'selected' : ''} onClick={() => update({mode: 'off'})}>
          Off
        </button>
      </div>

      {on && (
        <>
          <div className="caption-looks" role="radiogroup" aria-label="Subtitle look">
            {looks.map(([style, title, description]) => (
              <button
                type="button"
                role="radio"
                key={style}
                aria-checked={value.style === style}
                className={value.style === style ? 'selected' : ''}
                onClick={() => update({style})}
              >
                <span className={`look-swatch ${style}`} style={{'--caption-color': value.color} as CSSProperties} aria-hidden="true">
                  Aa
                </span>
                <span>
                  <strong>{title}</strong>
                  <small>{description}</small>
                </span>
              </button>
            ))}
          </div>

          <div className={`caption-preview ${value.style}`} style={{'--caption-color': value.color} as CSSProperties} aria-hidden="true">
            <span className="preview-scene" />
            <span className={`preview-word pos-${value.style === 'behind' ? 'top' : value.position}`}>
              {value.mode === 'custom' && value.lines[0]?.text ? value.lines[0].text : 'MAKE IT POP'}
            </span>
            {value.style === 'behind' && <span className="preview-person" />}
            <small>Preview</small>
          </div>

          <p className="hint">
            {value.style === 'behind'
              ? 'Works best when one person is clearly in the shot. If nobody is found, the words stay in front.'
              : value.mode === 'auto'
                ? 'We write subtitles from what is said in your clips. No talking? Add your own words under “More subtitle options”.'
                : 'Your own words will be used instead of detected speech.'}
          </p>

          <details className="disclosure" open={value.mode === 'custom' || undefined}>
            <summary>More subtitle options</summary>
            <div className="disclosure-body">
              <div className="caption-placement">
                <label htmlFor={id + '-color'}>
                  Highlight colour
                  <input id={id + '-color'} type="color" value={value.color} onChange={e => update({color: e.target.value})} />
                </label>
                {value.style !== 'behind' && (
                  <label htmlFor={id + '-position'}>
                    Position
                    <select
                      id={id + '-position'}
                      value={value.position}
                      onChange={e => update({position: e.target.value as CaptionSettings['position']})}
                    >
                      <option value="top">Top</option>
                      <option value="center">Middle</option>
                      <option value="bottom">Bottom</option>
                    </select>
                  </label>
                )}
              </div>

              <label className="check">
                <input
                  type="checkbox"
                  checked={value.mode === 'custom'}
                  onChange={e => update({mode: e.target.checked ? 'custom' : 'auto'})}
                />
                <span>Write my own words instead of using what is said</span>
              </label>

              {value.mode === 'custom' && (
                <div className="caption-lines">
                  <p className="hint">
                    Set when each line appears, in seconds, within the finished video (about {duration.toFixed(0)}s long). Long lines are
                    shown in short bursts.
                  </p>
                  {value.lines.map((line, i) => (
                    <div className="caption-line" key={i}>
                      <label className="sr-only" htmlFor={`${id}-line-${i}`}>
                        Line {i + 1} words
                      </label>
                      <textarea
                        id={`${id}-line-${i}`}
                        value={line.text}
                        maxLength={160}
                        placeholder={`Line ${i + 1}: your words…`}
                        onChange={e => setLine(i, {text: e.target.value})}
                      />
                      <div className="caption-timing">
                        {(['start', 'end'] as const).map(field => (
                          <label key={field}>
                            {field === 'start' ? 'From (s)' : 'To (s)'}
                            <input
                              aria-label={`Line ${i + 1} ${field === 'start' ? 'starts at' : 'ends at'}, seconds`}
                              type="number"
                              min={0}
                              max={duration}
                              step={0.1}
                              value={line[field]}
                              onChange={e => setLine(i, {[field]: Number(e.target.value)})}
                            />
                          </label>
                        ))}
                        <button
                          type="button"
                          className="link-button"
                          aria-label={`Remove line ${i + 1}`}
                          onClick={() => update({lines: value.lines.filter((_, n) => n !== i)})}
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  ))}
                  <button
                    type="button"
                    className="secondary small"
                    disabled={value.lines.length >= 100 || (value.lines.at(-1)?.end || 0) >= duration}
                    onClick={addLine}
                  >
                    + Add a line
                  </button>
                  {problem && (
                    <p className="hint warn" role="status">
                      {problem}
                    </p>
                  )}
                </div>
              )}
            </div>
          </details>
        </>
      )}

      {onExport && (
        <div className="caption-export">
          <button type="button" className="secondary wide" disabled={disabled || !!problem} onClick={onExport}>
            {exportLabel}
          </button>
          <p className="hint">Your cuts and music stay the same. The current version is kept.</p>
        </div>
      )}
    </fieldset>
  );
}
