'use client';

import { useEffect, useState } from 'react';
import { Eye, EyeOff } from 'lucide-react';
import DetectionCard from './DetectionCard';
import TokenText from './TokenText';
import type { DemoScenario } from '@/lib/demo-types';

type Capabilities = { model: string; watermarks: string[]; attacks: string[] };
const api = '/marksec/api/toolkit';

async function readResponse(response: Response) {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === 'string' ? data.detail : 'The request could not be completed.');
  }
  return data;
}

export default function WatermarkWorkbench({ mode }: { mode: 'live' | 'preset' }) {
  const [presets, setPresets] = useState<DemoScenario[]>([]);
  const [presetId, setPresetId] = useState('kgw-rw');
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [prompt, setPrompt] = useState('Explain what a zero-knowledge proof is and give a simple analogy.');
  const [watermark, setWatermark] = useState('KGW');
  const [attack, setAttack] = useState('LLMP');
  const [result, setResult] = useState<DemoScenario | null>(null);
  const [initializing, setInitializing] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [coloring, setColoring] = useState(false);
  const live = mode === 'live';

  useEffect(() => {
    const controller = new AbortController();
    async function initialize() {
      try {
        if (mode === 'live') {
          const response = await fetch(`${api}/capabilities`, { signal: controller.signal, cache: 'no-store' });
          setCapabilities(await readResponse(response));
        } else {
          const data = await Promise.all(['kgw-rw', 'kgw-llmp'].map(async (id) => {
            const response = await fetch(`/marksec/demo-data/${id}.json`, { signal: controller.signal });
            return readResponse(response);
          }));
          setPresets(data);
        }
      } catch (e) {
        if (!controller.signal.aborted) setError(e instanceof Error ? e.message : 'Could not load this mode.');
      } finally {
        if (!controller.signal.aborted) setInitializing(false);
      }
    }
    initialize();
    return () => controller.abort();
  }, [mode]);

  const selectedPreset = presets.find((item) => item.id === presetId);

  async function run() {
    setResult(null);
    setError(null);
    if (!live) {
      if (selectedPreset) setResult(selectedPreset);
      return;
    }
    setRunning(true);
    try {
      const response = await fetch(`${api}/run`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, watermark, attack, max_new_tokens: 128, seed: 42 }),
        signal: AbortSignal.timeout(300000),
      });
      setResult(await readResponse(response));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Toolkit request failed.');
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="demo-workbench mt-10" aria-busy={initializing || running}>
      <div className="demo-empty-state" role="status">
        <p>{live ? 'Live toolkit' : 'Preset demo'}</p>
        <span>{live
          ? (capabilities ? `Model: ${capabilities.model}. Each run generates, rewrites, and detects fresh text.` : error ? 'Toolkit is unavailable. Start the backend and reload this page.' : 'Connecting to the toolkit backend…')
          : 'Fixed illustrative answers and token colors. This mode makes no model calls.'}</span>
      </div>
      {error ? <p role="alert" className="rounded-lg border border-red-200 bg-red-50 p-5 text-red-700">{error}</p> : null}
      <div className="demo-control-panel">
        <section className="demo-prompt-panel">
          <label htmlFor={`prompt-${mode}`} className="demo-panel-heading">Prompt</label>
          <textarea id={`prompt-${mode}`} value={live ? prompt : selectedPreset?.prompt ?? ''}
            readOnly={!live} disabled={running || initializing} maxLength={4096}
            className={!live ? 'demo-prompt-readonly' : undefined}
            onChange={(event) => { setPrompt(event.target.value); setResult(null); }} />
          <button type="button" className="demo-generate-button" onClick={run}
            disabled={running || initializing || (live ? !capabilities || !prompt.trim() : !selectedPreset)}>
            {running ? 'Running…' : live ? 'Run toolkit' : 'Show preset answer'}
          </button>
          {running ? <p role="status">Generating and rewriting text. The first run also loads the model.</p> : null}
        </section>
        <section className="demo-options-panel">
          {live ? <>
            <label className="demo-option-group">Watermark
              <select aria-label="Watermark" value={watermark} disabled={running || !capabilities}
                onChange={(e) => { setWatermark(e.target.value); setResult(null); }}>
                {(capabilities?.watermarks ?? ['KGW']).map((name) => <option key={name}>{name}</option>)}
              </select>
            </label>
            <label className="demo-option-group">Attack method
              <select aria-label="Attack method" value={attack} disabled={running || !capabilities}
                onChange={(e) => { setAttack(e.target.value); setResult(null); }}>
                {(capabilities?.attacks ?? ['LLMP']).map((name) => <option key={name}>{name}</option>)}
              </select>
            </label>
          </> : <label className="demo-option-group">Preset scenario
            <select aria-label="Preset scenario" value={presetId} disabled={initializing}
              onChange={(e) => { setPresetId(e.target.value); setResult(null); }}>
              {presets.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}
            </select>
          </label>}
        </section>
      </div>
      {result ? <div className="demo-result-panel">
        <div className="demo-result-header">
          <div><p className="demo-result-kicker">{live ? 'Computed result' : 'Preset illustration'}</p><h3>{result.title}</h3></div>
          <button type="button" className="demo-color-toggle" aria-pressed={coloring} onClick={() => setColoring(!coloring)}>
            {coloring ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}{coloring ? 'Coloring on' : 'Coloring off'}
          </button>
        </div>
        <div className="demo-prompt-readout"><span>Prompt</span><p>{result.prompt}</p></div>
        <div className="demo-detection-grid">{result.variants.map((variant) =>
          <DetectionCard key={variant.label} title={variant.label} detection={variant.detection} />)}</div>
        <div className="demo-text-grid">{result.variants.map((variant) =>
          <section className="demo-text-panel" key={variant.label}>
            <div className="demo-text-panel-header"><span>{variant.label}</span><p>{variant.description}</p></div>
            <TokenText tokens={variant.tokens} coloring={coloring} verbatim={variant.token_spacing === 'verbatim'} />
          </section>)}</div>
        <p className="demo-score-note">{live
          ? `Model: ${result.model}. Detector scores are computed from the returned text; rewriting may leave the signal intact.`
          : 'Illustrative fixture values and simulated coloring; these are not measured benchmark results.'}</p>
      </div> : <div className="demo-empty-state"><p>{live
        ? 'Enter a prompt and run the toolkit to inspect generation and rewriting.'
        : 'Select a scenario and show its preset answer.'}</p></div>}
    </div>
  );
}
