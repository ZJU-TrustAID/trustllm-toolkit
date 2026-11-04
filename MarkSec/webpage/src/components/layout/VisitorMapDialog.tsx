"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";
import { Activity, Globe2, RefreshCw, X } from "lucide-react";

type VisitorCountry = {
  code: string;
  requests: number;
  uniqueIps: number;
};

type VisitorStats = {
  range: "all" | "30d";
  totalRequests: number;
  uniqueIps: number;
  countryCount: number;
  countries: VisitorCountry[];
  generatedAt: string;
  windowStart: string | null;
};

const COUNTRY_COORDINATES: Record<string, [number, number]> = {
  AR: [-64, -34],
  AU: [134, -25],
  BR: [-51, -10],
  CA: [-106, 57],
  CN: [104, 35],
  DE: [10, 51],
  ES: [-4, 40],
  FR: [2, 46],
  GB: [-3, 55],
  HK: [114, 22],
  IN: [79, 22],
  IT: [12, 42],
  JP: [138, 36],
  KR: [128, 36],
  NL: [5, 52],
  PL: [19, 52],
  RU: [90, 61],
  SG: [104, 1],
  TH: [101, 15],
  TW: [121, 24],
  UA: [32, 49],
  US: [-100, 38],
  VN: [108, 16],
  ZA: [24, -29],
};

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function countryName(code: string) {
  if (code === "ZZ") {
    return "Unresolved";
  }

  try {
    return new Intl.DisplayNames(["en"], { type: "region" }).of(code) ?? code;
  } catch {
    return code;
  }
}

function formatUpdatedAt(value: string | undefined) {
  if (!value) {
    return "—";
  }

  return new Intl.DateTimeFormat("en", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function mapPosition([longitude, latitude]: [number, number]) {
  return {
    left: `${((longitude + 180) / 360) * 100}%`,
    top: `${((90 - latitude) / 180) * 100}%`,
  };
}

export default function VisitorMapDialog() {
  const [open, setOpen] = useState(false);
  const [stats, setStats] = useState<VisitorStats>();
  const [loading, setLoading] = useState(false);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  async function loadStats() {
    setLoading(true);

    try {
      const response = await fetch("/marksec/api/visits/map?range=all", {
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(`Visitor map returned ${response.status}`);
      }

      setStats((await response.json()) as VisitorStats);
    } catch {
      // Keep the last successful snapshot visible when a refresh is unavailable.
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!open) {
      return;
    }

    const initialRefresh = window.setTimeout(() => void loadStats(), 0);
    const interval = window.setInterval(() => void loadStats(), 30_000);

    return () => {
      window.clearTimeout(initialRefresh);
      window.clearInterval(interval);
    };
  }, [open]);

  useEffect(() => {
    if (!open) {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  const visibleCountries = useMemo(
    () => (stats?.countries ?? []).filter(({ code }) => code !== "ZZ").slice(0, 12),
    [stats],
  );
  const largestCountryRequests = visibleCountries[0]?.requests ?? 1;

  return (
    <>
      <button
        type="button"
        className="site-footer-map-trigger"
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen(true)}
      >
        <Globe2 className="h-4 w-4" aria-hidden="true" />
        <span>Visitor map</span>
        <span className="site-footer-map-live" aria-hidden="true">
          Live
        </span>
      </button>

      {open ? (
        <div
          className="visitor-dialog-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.currentTarget === event.target) {
              setOpen(false);
            }
          }}
        >
          <section
            className="visitor-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="visitor-map-title"
          >
            <header className="visitor-dialog-header">
              <div>
                <p className="visitor-dialog-kicker">
                  <Activity className="h-3.5 w-3.5" aria-hidden="true" />
                  Live aggregate
                </p>
                <h2 id="visitor-map-title">Visitor map</h2>
              </div>
              <div className="visitor-dialog-actions">
                <span className="visitor-dialog-updated">
                  {loading ? "Refreshing" : `Updated ${formatUpdatedAt(stats?.generatedAt)}`}
                </span>
                <button
                  type="button"
                  className="visitor-dialog-icon-button"
                  onClick={() => void loadStats()}
                  aria-label="Refresh visitor data"
                  title="Refresh visitor data"
                >
                  <RefreshCw className={`h-4 w-4 ${loading ? "is-spinning" : ""}`} aria-hidden="true" />
                </button>
                <button
                  ref={closeButtonRef}
                  type="button"
                  className="visitor-dialog-icon-button"
                  onClick={() => setOpen(false)}
                  aria-label="Close visitor map"
                  title="Close visitor map"
                >
                  <X className="h-5 w-5" aria-hidden="true" />
                </button>
              </div>
            </header>

            <div className="visitor-dialog-summary" aria-live="polite">
              <div className="visitor-dialog-stat">
                <span>Requests</span>
                <strong>{formatNumber(stats?.totalRequests ?? 0)}</strong>
              </div>
              <div className="visitor-dialog-stat">
                <span>Unique IPs</span>
                <strong>{formatNumber(stats?.uniqueIps ?? 0)}</strong>
              </div>
              <div className="visitor-dialog-stat">
                <span>Countries</span>
                <strong>{formatNumber(stats?.countryCount ?? 0)}</strong>
              </div>
            </div>

            <div className="visitor-dialog-content">
              <div className="visitor-map-stage">
                <div className="visitor-map-canvas" role="img" aria-label="Visitor distribution by country">
                  <Image
                    className="visitor-map-image"
                    src="/marksec/maps/world-map.svg"
                    alt=""
                    fill
                    sizes="(max-width: 639px) 100vw, 60vw"
                  />
                  {visibleCountries.map((country) => {
                    const coordinates = COUNTRY_COORDINATES[country.code];
                    if (!coordinates) {
                      return null;
                    }

                    const radius = 0.55 + (country.requests / largestCountryRequests) * 0.75;
                    return (
                      <span
                        className="visitor-map-marker"
                        key={country.code}
                        style={mapPosition(coordinates)}
                        title={`${countryName(country.code)}: ${formatNumber(country.requests)} requests`}
                      >
                        <span className="visitor-map-marker-dot" style={{ width: `${radius}rem`, height: `${radius}rem` }} />
                        <span className="visitor-map-marker-label">{country.code}</span>
                      </span>
                    );
                  })}
                </div>
                {!stats?.countries.length && !loading ? (
                  <p className="visitor-map-empty">No visits recorded yet.</p>
                ) : null}
              </div>

              <div className="visitor-country-list">
                <div className="visitor-country-list-header">
                  <span>Top locations</span>
                  <span>Requests</span>
                </div>
                {visibleCountries.length ? (
                  visibleCountries.map((country) => (
                    <div className="visitor-country-row" key={country.code}>
                      <div>
                        <strong>{countryName(country.code)}</strong>
                        <span>{country.uniqueIps} unique IPs</span>
                      </div>
                      <b>{formatNumber(country.requests)}</b>
                      <span
                        className="visitor-country-bar"
                        style={{ width: `${Math.max(8, (country.requests / largestCountryRequests) * 100)}%` }}
                      />
                    </div>
                  ))
                ) : (
                  <p className="visitor-country-empty">Waiting for the first aggregate.</p>
                )}
              </div>
            </div>
          </section>
        </div>
      ) : null}
    </>
  );
}
