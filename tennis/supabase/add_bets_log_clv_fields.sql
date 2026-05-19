alter table tennis.bets_log
  add column if not exists clv_pinnacle_odd numeric(10,4),
  add column if not exists clv_captured_at timestamptz,
  add column if not exists clv_edge numeric(10,8),
  add column if not exists clv_value numeric(10,8),
  add column if not exists clv_source_event_id text,
  add column if not exists clv_source text;

create index if not exists idx_tennis_bets_log_clv_pending
  on tennis.bets_log (source, status, placed_at desc)
  where clv_pinnacle_odd is null;
