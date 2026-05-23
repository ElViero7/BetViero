-- Creación de la tabla de mapeo de jugadores para resolver nombres de la API a IDs únicos de Sackmann
CREATE TABLE IF NOT EXISTS tennis.player_mappings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  api_name TEXT NOT NULL,
  sackmann_player_id INTEGER NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT player_mappings_api_name_unique UNIQUE (api_name)
);

CREATE INDEX IF NOT EXISTS idx_player_mappings_api_name ON tennis.player_mappings (api_name);
GRANT ALL ON TABLE tennis.player_mappings TO anon, authenticated, service_role;
