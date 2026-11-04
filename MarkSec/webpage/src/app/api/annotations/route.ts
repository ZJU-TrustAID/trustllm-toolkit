import { randomUUID } from 'crypto';
import { NextRequest, NextResponse } from 'next/server';
import {
  enqueueWrite,
  isEnabled,
  isRecord,
  MAX_ANNOTATIONS,
  MAX_MESSAGE_LENGTH,
  notFound,
  readAnnotations,
  readString,
  saveAnnotations,
} from './store';
import type { Annotation, PropertyChange } from './store';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  if (!isEnabled()) return notFound();

  const pagePath = request.nextUrl.searchParams.get('page');
  const annotations = await readAnnotations();
  return NextResponse.json({
    annotations: pagePath ? annotations.filter((annotation) => annotation.pagePath === pagePath) : annotations,
  });
}

export async function POST(request: NextRequest) {
  if (!isEnabled()) return notFound();

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: '请求格式无效。' }, { status: 400 });
  }

  const annotation = parseAnnotation(body);
  if (!annotation) {
    return NextResponse.json({ error: '请提供有效的元素和评语。' }, { status: 400 });
  }

  await enqueueWrite(async () => {
    const annotations = await readAnnotations();
    annotations.unshift(annotation);
    await saveAnnotations(annotations.slice(0, MAX_ANNOTATIONS));
  });

  return NextResponse.json({ annotation }, { status: 201 });
}

function parseAnnotation(value: unknown): Annotation | null {
  if (!isRecord(value) || !isRecord(value.element)) return null;
  const { element } = value;
  const message = readString(value.message, MAX_MESSAGE_LENGTH);
  const pagePath = readString(value.pagePath, 240);
  const pageTitle = readString(value.pageTitle, 240);
  const tagName = readString(element.tagName, 80);
  const name = readString(element.name, 240);
  const selector = readString(element.selector, 1200);
  const text = readString(element.text, 400);

  if (!message || !pagePath || !tagName || !name || !selector) return null;
  if (!pagePath.startsWith('/') || pagePath.startsWith('//')) return null;

  const changes = readChanges(value.changes);

  return {
    id: randomUUID(),
    pagePath,
    pageTitle,
    message,
    element: {
      tagName,
      name,
      selector,
      text,
      rect: readRect(element.rect),
    },
    createdAt: new Date().toISOString(),
    status: 'open',
    ...(changes.length > 0 ? { changes } : {}),
  };
}

function readRect(value: unknown) {
  if (!isRecord(value)) return { top: 0, left: 0, width: 0, height: 0 };
  return {
    top: readNumber(value.top),
    left: readNumber(value.left),
    width: readNumber(value.width),
    height: readNumber(value.height),
  };
}

function readNumber(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? Math.round(value) : 0;
}

function readChanges(value: unknown): PropertyChange[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(isRecord)
    .map((item) => ({
      property: readString(item.property, 80),
      from: readString(item.from, 80),
      to: readString(item.to, 80),
    }))
    .filter((c) => c.property && c.from && c.to);
}
