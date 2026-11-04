"use client";

import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

type CounterResponse = {
  visits?: number;
};

function isLocalHost(hostname: string) {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
}

export default function SiteVisitCounter() {
  const pathname = usePathname();
  const [visits, setVisits] = useState<string>();

  useEffect(() => {
    if (isLocalHost(window.location.hostname)) {
      return;
    }

    let active = true;

    void fetch("/marksec/api/visits", {
      method: "POST",
      body: JSON.stringify({
        pathname,
        referrer: document.referrer,
      }),
      cache: "no-store",
      credentials: "omit",
      keepalive: true,
      headers: { "Content-Type": "application/json" },
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Visit counter returned ${response.status}`);
        }

        return response.json() as Promise<CounterResponse>;
      })
      .then((data) => {
        if (active && typeof data.visits === "number") {
          setVisits(new Intl.NumberFormat("en-US").format(data.visits));
        }
      })
      .catch(() => {
        // The counter is non-essential; keep the quiet fallback when unavailable.
      });

    return () => {
      active = false;
    };
  }, [pathname]);

  return (
    <p className="site-footer-visits" aria-label="Total site visits" aria-live="polite">
      Visits <span>{visits ?? "—"}</span>
    </p>
  );
}
