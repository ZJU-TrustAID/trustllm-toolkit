export type TokenLabel = 'green' | 'red' | 'neutral';

export interface TokenAnnotation {
  text: string;
  label: TokenLabel;
  token_id?: number;
  position?: number;
}

export interface DetectionMetrics {
  z_score: number | null;
  green_fraction: number;
  num_scored_tokens: number;
  is_watermarked: boolean;
  threshold?: number;
  reason?: string | null;
}

export interface DemoVariant {
  label: string;
  description: string;
  text: string;
  tokens: TokenAnnotation[];
  detection: DetectionMetrics;
  attack_method?: string;
  token_spacing?: 'verbatim';
}

export interface DemoScenario {
  mode?: 'live' | 'preset';
  id: string;
  title: string;
  description: string;
  prompt: string;
  model: string;
  watermark: {
    type: string;
    gamma: number;
    delta: number;
    prefix_length: number;
  };
  variants: DemoVariant[];
}
