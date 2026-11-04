'use client';

import { useEffect, useState } from 'react';
import { PatchMark } from 'patch-mark/react';
import type { AnnotationStore } from 'patch-mark';

export default function PatchMarkClient() {
  const [store, setStore] = useState<AnnotationStore | undefined>(undefined);

  useEffect(() => {
    if (process.env.NEXT_PUBLIC_MARKSEC_ANNOTATION_STORE !== 'fetch') return;
    let cancelled = false;
    // Dynamic import keeps the custom-element module (which extends
    // HTMLElement) out of the SSR bundle; createFetchStore is obtained lazily.
    import('patch-mark').then(({ createFetchStore }) => {
      if (!cancelled) {
        setStore(createFetchStore({ endpoint: '/marksec/api/annotations' }));
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <PatchMark
      store={store}
      onError={(error) => console.error('[patch-mark]', error)}
    />
  );
}
