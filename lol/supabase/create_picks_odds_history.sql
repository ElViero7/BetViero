CREATE TABLE IF NOT EXISTS picks_odds_history (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    pick_id uuid REFERENCES picks(id) ON DELETE CASCADE,
    my_odd numeric(8,4) NOT NULL,
    fair_odd numeric(8,4),
    ev numeric(8,6),
    recorded_at timestamptz DEFAULT now()
);
