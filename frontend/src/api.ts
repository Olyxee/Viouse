import type {CaptionSettings} from './CaptionStudio';

export type Asset = {
  id: string;
  filename: string;
  kind: 'video' | 'music' | 'reference' | string;
  duration: number;
  metadata: {width: number; height: number};
};

export type Version = {
  id: string;
  duration: number;
  source_clips_used: number;
  total_clips: number;
  detected_moments: number;
  style: string;
  title: string;
  captions?: CaptionSettings;
  decisions?: {purpose: string; why: string[]}[];
};

export type Job = {
  id: string;
  kind?: string;
  status: string;
  stage: string;
  detail: string;
  error: string | null;
};

export type Project = {
  id: string;
  name: string;
  status: string;
  style: string;
  target_duration: number;
  created_at: number;
  captions?: CaptionSettings;
  assets?: Asset[];
  job?: Job | null;
  versions?: Version[];
  best_version_id?: string;
};

export type Health = {ai_configured: boolean; provider: string; media_ready?: boolean};

const OFFLINE = 'The editor could not be reached. Make sure it is running on this computer (scripts\\start.ps1), then try again.';

export async function api<T = any>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch('/api' + path, {
      ...init,
      headers: {
        ...(init.body instanceof FormData ? {} : {'Content-Type': 'application/json'}),
        ...init.headers,
      },
    });
  } catch {
    throw new Error(OFFLINE);
  }
  const data = await res.json().catch(() => ({detail: OFFLINE}));
  if (!res.ok) throw new Error(errorText(data));
  return data;
}

function errorText(data: any): string {
  if (typeof data?.detail === 'string') return data.detail;
  if (Array.isArray(data?.detail) && typeof data.detail[0]?.msg === 'string') {
    return data.detail[0].msg.replace(/^Value error, /, '');
  }
  return 'Something went wrong. Please check your choices and try again.';
}

/** Streams one file to the existing PUT endpoint, reporting progress from 0 to 1. */
export function uploadFile(projectId: string, file: File, kind: string, onProgress: (fraction: number) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', `/api/projects/${projectId}/assets?kind=${kind}&filename=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = e => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress(1);
        resolve();
        return;
      }
      let data: any = null;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON error body */
      }
      reject(new Error(data ? errorText(data) : `Upload failed (${xhr.status}).`));
    };
    xhr.onerror = () => reject(new Error(OFFLINE));
    xhr.send(file);
  });
}

export const BUSY = ['QUEUED', 'PREPROCESSING', 'ANALYZING', 'PLANNING', 'EDITING', 'RENDERING', 'REVIEWING', 'DELETING'];

export type StatusInfo = {label: string; tone: 'draft' | 'working' | 'ready' | 'attention'};

export function projectStatus(status: string): StatusInfo {
  if (status === 'COMPLETED') return {label: 'Ready', tone: 'ready'};
  if (status === 'FAILED') return {label: 'Needs attention', tone: 'attention'};
  if (BUSY.includes(status)) return {label: 'Working', tone: 'working'};
  return {label: 'Draft', tone: 'draft'};
}

export function formatDuration(seconds: number): string {
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return m ? `${m}m ${s}s` : `${s}s`;
}

export function formatDate(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleDateString(undefined, {month: 'short', day: 'numeric'});
}
