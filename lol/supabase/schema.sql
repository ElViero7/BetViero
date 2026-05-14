create extension if not exists pgcrypto;

create schema if not exists betting;

create type betting.match_status as enum (
  'scheduled',
  'live',
  'completed',
  'cancelled'
);

create type betting.signal_status as enum (
  'open',
  'waiting_result',
  'won',
  'lost',
  'void'
);

create type betting.bet_side as enum (
  'blue',
  'red'
);

create or replace function betting.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = timezone('utc', now());
  return new;
end;
$$;

create table if not exists betting.teams (
  id uuid primary key default gen_random_uuid(),
  team_slug text not null unique,
  canonical_name text not null unique,
  league_hint text,
  region_hint text,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists betting.team_name_mappings (
  id uuid primary key default gen_random_uuid(),
  provider text not null,
  provider_team_name text not null,
  team_id uuid not null references betting.teams(id) on delete cascade,
  created_at timestamptz not null default timezone('utc', now()),
  unique (provider, provider_team_name)
);

create table if not exists betting.matches (
  id uuid primary key default gen_random_uuid(),
  external_match_id text,
  oracle_game_id text unique,
  odds_event_id text unique,
  league text not null,
  tournament_name text,
  patch text,
  best_of integer,
  map_number integer not null default 1,
  scheduled_at timestamptz,
  blue_team_id uuid not null references betting.teams(id),
  red_team_id uuid not null references betting.teams(id),
  match_status betting.match_status not null default 'scheduled',
  winner_team_id uuid references betting.teams(id),
  blue_firstblood boolean,
  red_firstblood boolean generated always as (
    case
      when blue_firstblood is null then null
      else not blue_firstblood
    end
  ) stored,
  raw_source jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint matches_map_one_only check (map_number = 1),
  constraint matches_different_teams check (blue_team_id <> red_team_id),
  constraint matches_winner_is_participant check (
    winner_team_id is null
    or winner_team_id = blue_team_id
    or winner_team_id = red_team_id
  )
);

create table if not exists betting.model_runs (
  id uuid primary key default gen_random_uuid(),
  model_name text not null,
  model_version text not null,
  market_key text not null default 'first_blood_map1',
  artifact_path text not null,
  feature_columns jsonb not null default '[]'::jsonb,
  metrics jsonb not null default '{}'::jsonb,
  train_rows integer,
  test_rows integer,
  trained_at timestamptz not null default timezone('utc', now()),
  created_at timestamptz not null default timezone('utc', now()),
  unique (model_name, model_version, market_key)
);

create table if not exists betting.predictions (
  id uuid primary key default gen_random_uuid(),
  match_id uuid not null references betting.matches(id) on delete cascade,
  model_run_id uuid not null references betting.model_runs(id) on delete restrict,
  market_key text not null default 'first_blood_map1',
  predicted_at timestamptz not null default timezone('utc', now()),
  blue_team_prob numeric(8,6) not null,
  red_team_prob numeric(8,6) not null,
  feature_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now()),
  constraint predictions_prob_bounds check (
    blue_team_prob between 0 and 1
    and red_team_prob between 0 and 1
  )
);

create table if not exists betting.bets (
  id uuid primary key default gen_random_uuid(),
  match_name text not null,
  my_odd numeric(10,4) not null,
  market text not null default 'First Blood',
  status text not null default 'pending',
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint bets_my_odd_positive check (my_odd > 1.0)
);

create table if not exists betting.odds_snapshots (
  id uuid primary key default gen_random_uuid(),
  match_id uuid not null references betting.matches(id) on delete cascade,
  bookmaker_key text not null,
  market_key text not null default 'first_blood_map1',
  side betting.bet_side not null,
  outcome_name text not null,
  decimal_odds numeric(10,4) not null,
  implied_prob numeric(10,6) generated always as (
    case
      when decimal_odds > 0 then round((1 / decimal_odds)::numeric, 6)
      else null
    end
  ) stored,
  collected_at timestamptz not null default timezone('utc', now()),
  raw_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now()),
  constraint odds_snapshots_decimal_odds_positive check (decimal_odds > 1.0)
);

create table if not exists betting.bet_signals (
  id uuid primary key default gen_random_uuid(),
  match_id uuid not null references betting.matches(id) on delete cascade,
  prediction_id uuid not null references betting.predictions(id) on delete cascade,
  odds_snapshot_id uuid not null references betting.odds_snapshots(id) on delete cascade,
  bookmaker_key text not null,
  market_key text not null default 'first_blood_map1',
  side betting.bet_side not null,
  fair_prob numeric(8,6) not null,
  implied_prob numeric(8,6) not null,
  edge_pct numeric(10,6) not null,
  ev_pct numeric(10,6) not null,
  kelly_fraction numeric(10,6),
  stake_units numeric(10,4),
  status betting.signal_status not null default 'open',
  alert_sent_at timestamptz,
  opened_at timestamptz not null default timezone('utc', now()),
  closed_at timestamptz,
  closing_decimal_odds numeric(10,4),
  closing_implied_prob numeric(10,6),
  closing_clv_pct numeric(10,6),
  result_side betting.bet_side,
  pnl_units numeric(10,4),
  notes text,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint bet_signals_prob_bounds check (
    fair_prob between 0 and 1
    and implied_prob between 0 and 1
  ),
  constraint bet_signals_result_consistency check (
    (status in ('open', 'waiting_result') and pnl_units is null)
    or status in ('won', 'lost', 'void')
  )
);

create unique index if not exists idx_team_name_mappings_provider_name
  on betting.team_name_mappings (provider, provider_team_name);

create index if not exists idx_matches_lookup
  on betting.matches (league, scheduled_at desc);

create index if not exists idx_matches_odds_event_id
  on betting.matches (odds_event_id);

create index if not exists idx_predictions_match_predicted_at
  on betting.predictions (match_id, predicted_at desc);

create index if not exists idx_odds_snapshots_match_collected
  on betting.odds_snapshots (match_id, collected_at desc);

create index if not exists idx_odds_snapshots_market_bookmaker
  on betting.odds_snapshots (market_key, bookmaker_key, collected_at desc);

create index if not exists idx_bet_signals_status_opened
  on betting.bet_signals (status, opened_at desc);

create index if not exists idx_bet_signals_match
  on betting.bet_signals (match_id, opened_at desc);

drop trigger if exists trg_teams_updated_at on betting.teams;
create trigger trg_teams_updated_at
before update on betting.teams
for each row
execute function betting.set_updated_at();

drop trigger if exists trg_matches_updated_at on betting.matches;
create trigger trg_matches_updated_at
before update on betting.matches
for each row
execute function betting.set_updated_at();

drop trigger if exists trg_bet_signals_updated_at on betting.bet_signals;
create trigger trg_bet_signals_updated_at
before update on betting.bet_signals
for each row
execute function betting.set_updated_at();

drop trigger if exists trg_bets_updated_at on betting.bets;
create trigger trg_bets_updated_at
before update on betting.bets
for each row
execute function betting.set_updated_at();

create or replace view betting.v_signal_dashboard as
select
  bs.id as signal_id,
  bs.status,
  bs.market_key,
  bs.bookmaker_key,
  bs.side,
  m.league,
  m.tournament_name,
  m.patch,
  m.scheduled_at,
  blue.canonical_name as blue_team,
  red.canonical_name as red_team,
  p.blue_team_prob,
  p.red_team_prob,
  os.decimal_odds as taken_decimal_odds,
  bs.fair_prob,
  bs.implied_prob as taken_implied_prob,
  bs.edge_pct,
  bs.ev_pct,
  bs.kelly_fraction,
  bs.stake_units,
  bs.closing_decimal_odds,
  bs.closing_implied_prob,
  bs.closing_clv_pct,
  bs.result_side,
  bs.pnl_units,
  bs.alert_sent_at,
  bs.opened_at,
  bs.closed_at
from betting.bet_signals bs
join betting.matches m on m.id = bs.match_id
join betting.predictions p on p.id = bs.prediction_id
join betting.odds_snapshots os on os.id = bs.odds_snapshot_id
join betting.teams blue on blue.id = m.blue_team_id
join betting.teams red on red.id = m.red_team_id
order by bs.opened_at desc;
