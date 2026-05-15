create schema if not exists tennis;

create extension if not exists pgcrypto;

create or replace function tennis.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table if not exists tennis.matches_history (
  id uuid primary key default gen_random_uuid(),
  source text not null default 'manual',
  external_match_id text,
  tour text,
  tournament text,
  round text,
  surface text,
  indoor boolean,
  best_of smallint,
  match_date timestamptz not null,
  player_1 text not null,
  player_2 text not null,
  player_1_id text,
  player_2_id text,
  player_1_rank integer,
  player_2_rank integer,
  player_1_seed integer,
  player_2_seed integer,
  player_1_age numeric(5,2),
  player_2_age numeric(5,2),
  player_1_hand text,
  player_2_hand text,
  winner text not null,
  loser text,
  score text,
  player_1_sets_won smallint,
  player_2_sets_won smallint,
  player_1_games_won smallint,
  player_2_games_won smallint,
  player_1_aces smallint,
  player_2_aces smallint,
  player_1_double_faults smallint,
  player_2_double_faults smallint,
  player_1_first_serve_pct numeric(6,5),
  player_2_first_serve_pct numeric(6,5),
  player_1_break_points_saved smallint,
  player_2_break_points_saved smallint,
  player_1_break_points_faced smallint,
  player_2_break_points_faced smallint,
  closing_odds_player_1 numeric(10,4),
  closing_odds_player_2 numeric(10,4),
  raw_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint matches_history_best_of_chk check (best_of is null or best_of in (3, 5)),
  constraint matches_history_surface_chk check (
    surface is null or surface in ('hard', 'clay', 'grass', 'carpet', 'unknown')
  ),
  constraint matches_history_winner_chk check (winner in (player_1, player_2)),
  constraint matches_history_unique_external unique (source, external_match_id)
);

create index if not exists idx_tennis_matches_history_date
  on tennis.matches_history (match_date desc);

create index if not exists idx_tennis_matches_history_players
  on tennis.matches_history (player_1, player_2, match_date desc);

create index if not exists idx_tennis_matches_history_surface
  on tennis.matches_history (surface, match_date desc);

drop trigger if exists trg_matches_history_updated_at on tennis.matches_history;
create trigger trg_matches_history_updated_at
before update on tennis.matches_history
for each row
execute function tennis.set_updated_at();

create table if not exists tennis.upcoming_odds (
  id uuid primary key default gen_random_uuid(),
  quote_key text,
  source text not null default 'the_odds_api',
  external_event_id text not null,
  sport_key text,
  commence_time timestamptz not null,
  tour text,
  tournament text,
  player_1 text not null,
  player_2 text not null,
  bookmaker_key text not null,
  bookmaker_title text,
  market_key text not null,
  selection text not null,
  line numeric(10,4),
  price numeric(10,4) not null,
  implied_probability numeric(10,8),
  collected_at timestamptz not null default now(),
  raw_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint upcoming_odds_price_chk check (price > 1.0),
  constraint upcoming_odds_probability_chk check (
    implied_probability is null or (implied_probability >= 0 and implied_probability <= 1)
  ),
  constraint upcoming_odds_unique_quote unique (
    source,
    external_event_id,
    bookmaker_key,
    market_key,
    selection,
    line
  )
);

create index if not exists idx_tennis_upcoming_odds_commence
  on tennis.upcoming_odds (commence_time);

create index if not exists idx_tennis_upcoming_odds_event_market
  on tennis.upcoming_odds (external_event_id, market_key, bookmaker_key);

create index if not exists idx_tennis_upcoming_odds_collected
  on tennis.upcoming_odds (collected_at desc);

create unique index if not exists idx_tennis_upcoming_odds_quote_key
  on tennis.upcoming_odds (quote_key);

create unique index if not exists idx_tennis_upcoming_odds_unique_quote
  on tennis.upcoming_odds (
    source,
    external_event_id,
    bookmaker_key,
    market_key,
    selection,
    coalesce(line, -999999)
  );

drop trigger if exists trg_upcoming_odds_updated_at on tennis.upcoming_odds;
create trigger trg_upcoming_odds_updated_at
before update on tennis.upcoming_odds
for each row
execute function tennis.set_updated_at();

create table if not exists tennis.bets_log (
  id uuid primary key default gen_random_uuid(),
  tennis_pick_id uuid references tennis.tennis_picks (id) on delete set null,
  upcoming_odds_id uuid references tennis.upcoming_odds (id) on delete set null,
  match_history_id uuid references tennis.matches_history (id) on delete set null,
  placed_at timestamptz not null default now(),
  event_time timestamptz,
  source text not null default 'manual',
  bet_type text not null default 'real',
  tour text,
  tournament text,
  player_1 text not null,
  player_2 text not null,
  market_key text not null,
  selection text not null,
  line numeric(10,4),
  bookmaker_key text,
  my_odd numeric(10,4) not null,
  model_probability numeric(10,8),
  model_fair_odd numeric(10,4),
  edge numeric(10,8),
  ev numeric(10,8),
  stake numeric(12,4) not null,
  kelly_fraction numeric(10,8),
  status text not null default 'pending',
  result_selection text,
  pnl numeric(12,4),
  settled_at timestamptz,
  notes text,
  raw_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint bets_log_bet_type_chk check (bet_type in ('real', 'paper')),
  constraint bets_log_status_chk check (status in ('pending', 'won', 'lost', 'void')),
  constraint bets_log_my_odd_chk check (my_odd > 1.0),
  constraint bets_log_stake_chk check (stake >= 0),
  constraint bets_log_probability_chk check (
    model_probability is null or (model_probability >= 0 and model_probability <= 1)
  )
);

create index if not exists idx_tennis_bets_log_placed
  on tennis.bets_log (placed_at desc);

create index if not exists idx_tennis_bets_log_status
  on tennis.bets_log (status, placed_at desc);

create index if not exists idx_tennis_bets_log_event
  on tennis.bets_log (event_time, player_1, player_2);

create index if not exists idx_tennis_bets_log_tennis_pick
  on tennis.bets_log (tennis_pick_id);

drop trigger if exists trg_bets_log_updated_at on tennis.bets_log;
create trigger trg_bets_log_updated_at
before update on tennis.bets_log
for each row
execute function tennis.set_updated_at();

create table if not exists tennis.tennis_picks (
  id uuid primary key default gen_random_uuid(),
  partido text not null,
  mercado text not null,
  linea double precision not null,
  casa_apuestas text not null,
  cuota_inicial double precision not null,
  nuestra_prob double precision not null,
  ev_inicial double precision not null,
  fecha_registro timestamptz not null default now(),
  cuota_cierre double precision,
  resultado text,
  constraint tennis_picks_cuota_inicial_chk check (cuota_inicial > 1.0),
  constraint tennis_picks_prob_chk check (nuestra_prob >= 0 and nuestra_prob <= 1)
);

create index if not exists idx_tennis_picks_lookup
  on tennis.tennis_picks (partido, mercado, linea, fecha_registro desc);

create index if not exists idx_tennis_picks_fecha
  on tennis.tennis_picks (fecha_registro desc);

grant usage on schema tennis to anon, authenticated, service_role;
grant all on all tables in schema tennis to anon, authenticated, service_role;
grant all on all sequences in schema tennis to anon, authenticated, service_role;
