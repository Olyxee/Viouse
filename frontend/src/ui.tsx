import {AlertTriangle, Check, X} from 'lucide-react';
import type {ReactNode} from 'react';
import {projectStatus} from './api';

export const STEPS = ['Add footage', 'Choose style', 'Create', 'Review & download'] as const;

type StepperProps = {
  current: number;
  /** Steps (0-based) the user may jump back to. */
  reachable?: number[];
  onSelect?: (index: number) => void;
};

export function Stepper({current, reachable = [], onSelect}: StepperProps) {
  return (
    <nav className="stepper" aria-label="Progress">
      <ol>
        {STEPS.map((label, i) => {
          const state = i < current ? 'done' : i === current ? 'current' : 'todo';
          const canJump = onSelect && i !== current && reachable.includes(i);
          const content = (
            <>
              <span className="step-dot" aria-hidden="true">
                {state === 'done' ? <Check size={14} strokeWidth={3} /> : i + 1}
              </span>
              <span className="step-label">{label}</span>
              {state === 'done' && <span className="sr-only">(done)</span>}
            </>
          );
          return (
            <li key={label} className={state} aria-current={state === 'current' ? 'step' : undefined}>
              {canJump ? (
                <button type="button" onClick={() => onSelect(i)}>
                  {content}
                </button>
              ) : (
                <span className="step-inner">{content}</span>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

export function StatusBadge({status}: {status: string}) {
  const info = projectStatus(status);
  return <span className={`badge ${info.tone}`}>{info.label}</span>;
}

export function ErrorBanner({message, onDismiss, action}: {message: string; onDismiss?: () => void; action?: ReactNode}) {
  return (
    <div className="alert" role="alert">
      <AlertTriangle size={20} aria-hidden="true" className="alert-icon" />
      <div className="alert-body">
        <p>{message}</p>
        {action}
      </div>
      {onDismiss && (
        <button type="button" className="icon-button" aria-label="Dismiss message" onClick={onDismiss}>
          <X size={18} />
        </button>
      )}
    </div>
  );
}
