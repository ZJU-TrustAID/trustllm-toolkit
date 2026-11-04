import { NextResponse } from "next/server";

const COUNTER_ENDPOINT = "https://cdn.busuanzi.cc/api.php";
// Keep the existing counter identity and historical totals across the domain move.
const PUBLIC_SITE_ORIGIN = "https://marksec.trustaid.net";
// Busuanzi stat 103266 at cutover (2026-08-22): 10 PV and 4 UV.
const LEGACY_VISIT_OFFSET = 10;

type CounterResponse = {
  busuanzi_site_pv?: number;
};

type VisitRequest = {
  pathname?: string;
  referrer?: string;
};

function safeReferrer(value: unknown) {
  if (typeof value !== "string" || value.length > 2048) {
    return "";
  }

  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : "";
  } catch {
    return "";
  }
}

export async function POST(request: Request) {
  let body: VisitRequest;

  try {
    body = (await request.json()) as VisitRequest;
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  if (typeof body.pathname !== "string" || !body.pathname.startsWith("/")) {
    return NextResponse.json({ error: "Invalid pathname" }, { status: 400 });
  }

  try {
    const response = await fetch(COUNTER_ENDPOINT, {
      method: "POST",
      body: JSON.stringify({
        url: new URL(body.pathname, PUBLIC_SITE_ORIGIN).href,
        referrer: safeReferrer(body.referrer),
      }),
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      throw new Error(`Visit counter returned ${response.status}`);
    }

    const data = (await response.json()) as CounterResponse;
    if (typeof data.busuanzi_site_pv !== "number") {
      throw new Error("Visit counter returned an invalid payload");
    }

    return NextResponse.json(
      { visits: data.busuanzi_site_pv + LEGACY_VISIT_OFFSET },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch {
    return NextResponse.json({ error: "Visit counter unavailable" }, { status: 502 });
  }
}
