/**
 * batchApi.ts — browser-side calls for the batch routes. Every call that
 * touches a run carries the signed-in user's GUB token: for `run_as: me` runs
 * that header is how the job keeps a fresh token.
 */
'use client';

import { getValidAccessToken } from './auth/tokenStore';
import type { Arm, Question, RunAs, RunListItem, RunMeta, RunView } from './batch/types';
import type { Problem } from './batch/store';
import { ApiError } from './api';

export type { Problem };

export interface QuestionsView { text: string; questions: Question[]; problems: Problem[]; sha256: string; path: string }
export interface ArmsView { arms: Arm[]; problems: Problem[]; path: string }
export interface SubjectProbe { configured: boolean; serviceAccount: string | null; ok: boolean; email: string | null; error: string | null }

async function json<T>(res: Response): Promise<T> {
  const data = (await res.json().catch(() => ({}))) as T & { code?: string; message?: string; problems?: Problem[] };
  if (!res.ok) {
    const err = new ApiError(data.code ?? 'Error', data.message ?? res.statusText, res.status);
    (err as ApiError & { problems?: Problem[] }).problems = data.problems;
    throw err;
  }
  return data;
}

export const fetchQuestions = () => fetch('/api/batch/questions', { cache: 'no-store' }).then(json<QuestionsView>);
export const saveQuestionsText = (text: string) =>
  fetch('/api/batch/questions', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) }).then(json<QuestionsView>);
export const saveQuestions = (questions: Question[]) =>
  fetch('/api/batch/questions', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ questions }) }).then(json<QuestionsView>);

export const fetchArms = () => fetch('/api/batch/configs', { cache: 'no-store' }).then(json<ArmsView>);
export const saveArms = (arms: Arm[]) =>
  fetch('/api/batch/configs', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ arms }) }).then(json<ArmsView>);

export const fetchSubject = () => fetch('/api/batch/subject', { cache: 'no-store' }).then(json<SubjectProbe>);
export const fetchRuns = () => fetch('/api/batch/runs', { cache: 'no-store' }).then(json<{ runs: RunListItem[] }>);

export async function startRun(args: { questionIds: string[]; arms: string[]; parallel: number; timeout: number; cooldown: number; runAs: RunAs }): Promise<RunMeta> {
  const token = await getValidAccessToken();
  return fetch('/api/batch/runs', {
    method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }, body: JSON.stringify(args),
  }).then(json<RunMeta>);
}

export async function fetchRun(id: string): Promise<RunView> {
  const token = await getValidAccessToken().catch(() => null);
  return fetch(`/api/batch/runs/${encodeURIComponent(id)}`, { cache: 'no-store', headers: token ? { Authorization: `Bearer ${token}` } : {} }).then(json<RunView>);
}

export const stopRun = (id: string) => fetch(`/api/batch/runs/${encodeURIComponent(id)}/stop`, { method: 'POST' }).then(json<{ stopped: boolean }>);
export const csvUrl = (id: string) => `/api/batch/runs/${encodeURIComponent(id)}/csv`;
