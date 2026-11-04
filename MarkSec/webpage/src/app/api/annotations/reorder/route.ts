import { NextRequest, NextResponse } from 'next/server';
import {
  enqueueWrite,
  isEnabled,
  isRecord,
  notFound,
  readAnnotations,
  saveAnnotations,
} from '../store';
import type { Annotation } from '../store';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  if (!isEnabled()) return notFound();

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: '请求格式无效。' }, { status: 400 });
  }
  if (!isRecord(body) || !Array.isArray(body.ids)) {
    return NextResponse.json({ error: '请提供有效的 ids。' }, { status: 400 });
  }
  const ids = body.ids.filter((id): id is string => typeof id === 'string');

  await enqueueWrite(async () => {
    const annotations = await readAnnotations();
    const byId = new Map(annotations.map((annotation) => [annotation.id, annotation]));
    const used = new Set<string>();
    const reordered: Annotation[] = [];
    for (const id of ids) {
      const annotation = byId.get(id);
      if (annotation && !used.has(id)) {
        reordered.push(annotation);
        used.add(id);
      }
    }
    for (const annotation of annotations) {
      if (!used.has(annotation.id)) reordered.push(annotation);
    }
    await saveAnnotations(reordered);
  });

  return new NextResponse(null, { status: 204 });
}
