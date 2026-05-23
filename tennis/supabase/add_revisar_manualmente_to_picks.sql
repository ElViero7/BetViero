-- Migración para agregar la columna revisar_manualmente a la tabla tennis.tennis_picks
ALTER TABLE tennis.tennis_picks 
ADD COLUMN IF NOT EXISTS revisar_manualmente BOOLEAN NOT NULL DEFAULT FALSE;
