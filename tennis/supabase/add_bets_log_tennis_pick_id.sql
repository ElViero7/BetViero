alter table tennis.bets_log
  add column if not exists tennis_pick_id uuid references tennis.tennis_picks (id) on delete set null;

create index if not exists idx_tennis_bets_log_tennis_pick
  on tennis.bets_log (tennis_pick_id);
