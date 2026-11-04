import { randomUUID } from 'crypto';
import { mkdir, readFile, rename, writeFile } from 'fs/promises';
import path from 'path';
import { NextResponse } from 'next/server';

export const MAX_ANNOTATIONS = 1000;
export const MAX_MESSAGE_LENGTH = 1200;

const storePath = process.env.MARKSEC_ANNOTATION_STORE_PATH || path.join(/* turbopackIgnore: true */ process.cwd(), '.data', 'annotations.json');
const annotationsEnabled = process.env.NEXT_PUBLIC_MARKSEC_ENABLE_ANNOTATIONS === 'true';
let writeQueue = Promise.resolve();

export type PropertyChange = {
  property: string;
  from: string;
  to: string;
};

export type AnnotationStatus = 'open' | 'resolved';

export type Annotation = {
  id: string;
  pagePath: string;
  pageTitle: string;
  message: string;
  element: {
    tagName: string;
    name: string;
    selector: string;
    text: string;
    rect: { top: number; left: number; width: number; height: number };
  };
  createdAt: string;
  status?: AnnotationStatus;
  changes?: PropertyChange[];
};

export function isEnabled() {
  return annotationsEnabled;
}

export function notFound() {
  return NextResponse.json({ error: 'Not found' }, { status: 404 });
}

export async function readAnnotations(): Promise<Annotation[]> {
  try {
    const content = await readFile(storePath, 'utf8');
    const value: unknown = JSON.parse(content);
    return Array.isArray(value) ? value.filter(isStoredAnnotation) : [];
  } catch (error) {
    if (isNodeError(error, 'ENOENT') || error instanceof SyntaxError) return [];
    throw error;
  }
}

export async function saveAnnotations(annotations: Annotation[]) {
  await mkdir(path.dirname(storePath), { recursive: true });
  const temporaryPath = `${storePath}.${randomUUID()}.tmp`;
  await writeFile(temporaryPath, `${JSON.stringify(annotations, null, 2)}\n`, 'utf8');
  await rename(temporaryPath, storePath);
}

export function enqueueWrite(task: () => Promise<void>) {
  const next = writeQueue.then(task, task);
  writeQueue = next.catch(() => undefined);
  return next;
}

export function readString(value: unknown, maxLength: number) {
  return typeof value === 'string' ? value.trim().slice(0, maxLength) : '';
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isStoredAnnotation(value: unknown): value is Annotation {
  return isRecord(value) && typeof value.id === 'string' && typeof value.pagePath === 'string' && typeof value.message === 'string' && isRecord(value.element) && typeof value.createdAt === 'string';
}

function isNodeError(error: unknown, code: string) {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === code;
}
