import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const endpoints: Record<string, { method: string; path: string }> = {
  health: { method: 'GET', path: '/health' },
  capabilities: { method: 'GET', path: '/v1/capabilities' },
  run: { method: 'POST', path: '/v1/run' },
  detect: { method: 'POST', path: '/v1/detect' },
};

async function forward(request: NextRequest, context: { params: Promise<{ operation: string }> }) {
  const { operation } = await context.params;
  const endpoint = endpoints[operation];
  if (!endpoint) return NextResponse.json({ detail: 'Unknown operation.' }, { status: 404 });
  if (request.method !== endpoint.method) {
    return NextResponse.json({ detail: 'Method not allowed.' }, { status: 405 });
  }
  if (request.method === 'POST') {
    const origin = request.headers.get('origin');
    if (origin) {
      // Next.js may use its internal bind address for nextUrl. Compare with
      // the HTTP Host header, which identifies the browser-facing authority.
      let sameHost = false;
      try {
        const source = new URL(origin);
        sameHost = ['http:', 'https:'].includes(source.protocol)
          && source.host === request.headers.get('host');
      } catch { /* Reject malformed origins. */ }
      if (!sameHost) {
        return NextResponse.json({ detail: 'Cross-origin request rejected.' }, { status: 403 });
      }
    }
  }
  const backend = process.env.MARKSEC_BACKEND_URL;
  if (!backend) {
    return NextResponse.json({ detail: 'Configure MARKSEC_BACKEND_URL to connect the toolkit backend.' }, { status: 503 });
  }
  let body: string | undefined;
  if (request.method === 'POST') {
    body = await request.text();
    if (Buffer.byteLength(body) > 65536) {
      return NextResponse.json({ detail: 'Request too large.' }, { status: 413 });
    }
  }
  try {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (process.env.MARKSEC_API_TOKEN) headers.Authorization = `Bearer ${process.env.MARKSEC_API_TOKEN}`;
    const response = await fetch(`${backend.replace(/\/$/, '')}${endpoint.path}`, {
      method: request.method, headers, body, cache: 'no-store',
      signal: AbortSignal.timeout(300000),
    });
    return new NextResponse(await response.text(), {
      status: response.status,
      headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store',
        ...(response.headers.has('retry-after') ? { 'Retry-After': response.headers.get('retry-after')! } : {}) },
    });
  } catch {
    return NextResponse.json({ detail: 'Toolkit backend is unavailable or timed out.' }, { status: 503 });
  }
}

export const GET = forward;
export const POST = forward;
