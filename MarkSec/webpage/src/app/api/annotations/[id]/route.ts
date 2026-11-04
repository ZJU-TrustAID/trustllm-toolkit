import { NextRequest, NextResponse } from 'next/server';
import {
  enqueueWrite,
  isEnabled,
  isRecord,
  MAX_MESSAGE_LENGTH,
  notFound,
  readAnnotations,
  readString,
  saveAnnotations,
} from '../store';
import type { Annotation, AnnotationStatus } from '../store';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type RouteContext = { params: Promise<{ id: string }> };

export async function PATCH(request: NextRequest, context: RouteContext) {
  if (!isEnabled()) return notFound();
  const { id } = await context.params;

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: '请求格式无效。' }, { status: 400 });
  }
  if (!isRecord(body)) {
    return NextResponse.json({ error: '请提供有效字段。' }, { status: 400 });
  }

  const patch: { message?: string; status?: AnnotationStatus } = {};
  if ('message' in body) {
    const message = readString(body.message, MAX_MESSAGE_LENGTH);
    if (!message) {
      return NextResponse.json({ error: '评语不能为空。' }, { status: 400 });
    }
    patch.message = message;
  }
  if ('status' in body) {
    if (body.status !== 'open' && body.status !== 'resolved') {
      return NextResponse.json({ error: '无效的状态值。' }, { status: 400 });
    }
    patch.status = body.status;
  }
  if (patch.message === undefined && patch.status === undefined) {
    return NextResponse.json({ error: '请提供有效字段。' }, { status: 400 });
  }

  let updated: Annotation | null = null;
  await enqueueWrite(async () => {
    const annotations = await readAnnotations();
    const index = annotations.findIndex((annotation) => annotation.id === id);
    if (index === -1) return;
    annotations[index] = { ...annotations[index], ...patch };
    await saveAnnotations(annotations);
    updated = annotations[index];
  });

  if (!updated) return notFound();
  return NextResponse.json({ annotation: updated });
}

export async function DELETE(_request: NextRequest, context: RouteContext) {
  if (!isEnabled()) return notFound();
  const { id } = await context.params;

  let found = false;
  await enqueueWrite(async () => {
    const annotations = await readAnnotations();
    const remaining = annotations.filter((annotation) => annotation.id !== id);
    found = remaining.length !== annotations.length;
    if (found) await saveAnnotations(remaining);
  });

  if (!found) return notFound();
  return new NextResponse(null, { status: 204 });
}
