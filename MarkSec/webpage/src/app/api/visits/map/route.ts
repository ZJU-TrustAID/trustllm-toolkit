import { readdir, readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import path from "node:path";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const LOG_DIRECTORY = "/var/log/nginx";
const LOG_PREFIX = "marksec-visits.jsonl";
const CACHE_TTL_MS = 30_000;
const THIRTY_DAYS_MS = 30 * 24 * 60 * 60 * 1000;

type RangeKey = "all" | "30d";

type VisitLogEntry = {
  timestamp?: string;
  ip?: string;
  country?: string;
  status?: number;
};

type CachedStats = {
  expiresAt: number;
  payload: VisitorStats;
};

type VisitorStats = {
  range: RangeKey;
  totalRequests: number;
  uniqueIps: number;
  countryCount: number;
  countries: Array<{
    code: string;
    requests: number;
    uniqueIps: number;
  }>;
  generatedAt: string;
  windowStart: string | null;
};

const cache = new Map<RangeKey, CachedStats>();

async function getLogFiles() {
  const entries = await readdir(LOG_DIRECTORY);
  return entries
    .filter((entry) => entry === LOG_PREFIX || entry.startsWith(`${LOG_PREFIX}.`))
    .map((entry) => path.join(LOG_DIRECTORY, entry))
    .sort();
}

async function readLogFile(filePath: string) {
  const contents = await readFile(filePath);
  return filePath.endsWith(".gz")
    ? gunzipSync(contents).toString("utf8")
    : contents.toString("utf8");
}

function normalizeCountry(country: unknown) {
  return typeof country === "string" && /^[A-Z]{2}$/.test(country) ? country : "ZZ";
}

async function buildStats(range: RangeKey): Promise<VisitorStats> {
  const now = Date.now();
  const cutoff = range === "30d" ? now - THIRTY_DAYS_MS : 0;
  const allIps = new Set<string>();
  const countryStats = new Map<string, { requests: number; ips: Set<string> }>();
  let totalRequests = 0;

  for (const filePath of await getLogFiles()) {
    let contents: string;

    try {
      contents = await readLogFile(filePath);
    } catch {
      continue;
    }

    for (const line of contents.split("\n")) {
      if (!line.trim()) {
        continue;
      }

      let entry: VisitLogEntry;
      try {
        entry = JSON.parse(line) as VisitLogEntry;
      } catch {
        continue;
      }

      if (typeof entry.ip !== "string" || !entry.ip) {
        continue;
      }

      if (typeof entry.status === "number" && (entry.status < 200 || entry.status >= 400)) {
        continue;
      }

      const timestamp = entry.timestamp ? Date.parse(entry.timestamp) : Number.NaN;
      if (cutoff && (!Number.isFinite(timestamp) || timestamp < cutoff)) {
        continue;
      }

      const code = normalizeCountry(entry.country);
      const current = countryStats.get(code) ?? { requests: 0, ips: new Set<string>() };
      current.requests += 1;
      current.ips.add(entry.ip);
      countryStats.set(code, current);
      allIps.add(entry.ip);
      totalRequests += 1;
    }
  }

  const countries = [...countryStats.entries()]
    .map(([code, stats]) => ({
      code,
      requests: stats.requests,
      uniqueIps: stats.ips.size,
    }))
    .sort((left, right) => right.requests - left.requests || right.uniqueIps - left.uniqueIps);

  return {
    range,
    totalRequests,
    uniqueIps: allIps.size,
    countryCount: countries.filter(({ code }) => code !== "ZZ").length,
    countries,
    generatedAt: new Date(now).toISOString(),
    windowStart: cutoff ? new Date(cutoff).toISOString() : null,
  };
}

export async function GET(request: Request) {
  const requestedRange = new URL(request.url).searchParams.get("range");
  const range: RangeKey = requestedRange === "30d" ? "30d" : "all";
  const now = Date.now();
  const cached = cache.get(range);

  if (cached && cached.expiresAt > now) {
    return NextResponse.json(cached.payload, {
      headers: { "Cache-Control": "no-store" },
    });
  }

  try {
    const payload = await buildStats(range);
    cache.set(range, { expiresAt: now + CACHE_TTL_MS, payload });
    return NextResponse.json(payload, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch {
    return NextResponse.json({ error: "Visitor map unavailable" }, { status: 503 });
  }
}
