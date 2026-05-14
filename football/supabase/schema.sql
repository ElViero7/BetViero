begin;

create extension if not exists pgcrypto;

create table if not exists public.teams_mapping (
    id uuid primary key default gen_random_uuid(),
    canonical_team_name text not null,
    canonical_team_slug text not null,
    fifa_code char(3),
    api_football_team_id bigint,
    api_football_team_name text,
    odds_api_team_name text,
    odds_api_team_key text,
    country_code char(2),
    is_active boolean not null default true,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint teams_mapping_canonical_team_slug_key unique (canonical_team_slug),
    constraint teams_mapping_api_football_team_id_key unique (api_football_team_id),
    constraint teams_mapping_odds_api_team_key_key unique (odds_api_team_key)
);

create index if not exists idx_teams_mapping_fifa_code
    on public.teams_mapping (fifa_code);

create table if not exists public.matches (
    id uuid primary key default gen_random_uuid(),
    api_football_fixture_id bigint not null,
    competition_name text not null,
    competition_stage text,
    season integer not null,
    match_date timestamptz not null,
    home_team_id uuid not null references public.teams_mapping (id),
    away_team_id uuid not null references public.teams_mapping (id),
    venue_name text,
    status text not null default 'scheduled',
    home_score integer,
    away_score integer,
    stats_snapshot jsonb not null default '{}'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint matches_api_football_fixture_id_key unique (api_football_fixture_id),
    constraint matches_distinct_teams_chk check (home_team_id <> away_team_id)
);

create index if not exists idx_matches_match_date
    on public.matches (match_date);

create index if not exists idx_matches_status_match_date
    on public.matches (status, match_date);

create table if not exists public.predictions (
    id uuid primary key default gen_random_uuid(),
    match_id uuid not null references public.matches (id) on delete cascade,
    market_type text not null,
    selection text not null,
    line_value numeric(10,4),
    model_version text not null,
    predicted_probability numeric(8,6) not null,
    fair_odds numeric(10,4) generated always as (
        case
            when predicted_probability > 0 then round((1 / predicted_probability)::numeric, 4)
            else null
        end
    ) stored,
    prediction_source text not null default 'daily_batch',
    feature_snapshot jsonb not null default '{}'::jsonb,
    model_inputs_hash text,
    predicted_at timestamptz not null default timezone('utc', now()),
    created_at timestamptz not null default timezone('utc', now()),
    constraint predictions_probability_chk
        check (predicted_probability > 0 and predicted_probability < 1),
    constraint predictions_unique_key
        unique (match_id, market_type, selection, line_value, model_version)
);

create index if not exists idx_predictions_match_market
    on public.predictions (match_id, market_type, selection);

create index if not exists idx_predictions_predicted_at
    on public.predictions (predicted_at desc);

create table if not exists public.odds_history (
    id uuid primary key default gen_random_uuid(),
    match_id uuid not null references public.matches (id) on delete cascade,
    sportsbook_key text not null,
    market_type text not null,
    selection text not null,
    line_value numeric(10,4),
    price_decimal numeric(10,4) not null,
    implied_probability numeric(8,6) generated always as (
        round((1 / price_decimal)::numeric, 6)
    ) stored,
    bookmaker_last_update timestamptz,
    collected_at timestamptz not null default timezone('utc', now()),
    source_event_id text,
    raw_payload jsonb not null default '{}'::jsonb,
    constraint odds_history_price_decimal_chk check (price_decimal > 1.0000)
);

create index if not exists idx_odds_history_collected_at
    on public.odds_history (collected_at desc);

create index if not exists idx_odds_history_match_market_time
    on public.odds_history (match_id, market_type, selection, sportsbook_key, collected_at desc);

create index if not exists idx_odds_history_delta_lookup
    on public.odds_history (match_id, sportsbook_key, market_type, selection, line_value, collected_at desc);

create table if not exists public.value_bets_tracked (
    id uuid primary key default gen_random_uuid(),
    prediction_id uuid not null references public.predictions (id) on delete cascade,
    odds_history_id uuid not null references public.odds_history (id) on delete cascade,
    match_id uuid not null references public.matches (id) on delete cascade,
    sportsbook_key text not null,
    market_type text not null,
    selection text not null,
    line_value numeric(10,4),
    model_version text not null,
    predicted_probability numeric(8,6) not null,
    fair_odds numeric(10,4) not null,
    offered_odds numeric(10,4) not null,
    implied_probability_after_margin numeric(8,6) not null,
    edge numeric(10,6) not null,
    expected_value numeric(10,6) not null,
    stake_plan jsonb not null default '{}'::jsonb,
    alert_status text not null default 'pending',
    alert_sent_at timestamptz,
    tracked_at timestamptz not null default timezone('utc', now()),
    closing_odds numeric(10,4),
    closing_line_value numeric(10,6),
    bet_outcome text,
    realized_pnl numeric(12,4),
    settlement_at timestamptz,
    metadata jsonb not null default '{}'::jsonb,
    constraint value_bets_tracked_predicted_probability_chk
        check (predicted_probability > 0 and predicted_probability < 1),
    constraint value_bets_tracked_offered_odds_chk
        check (offered_odds > 1.0000),
    constraint value_bets_tracked_expected_value_chk
        check (expected_value > 0),
    constraint value_bets_tracked_unique_alert
        unique (prediction_id, odds_history_id)
);

create index if not exists idx_value_bets_tracked_match
    on public.value_bets_tracked (match_id, tracked_at desc);

create index if not exists idx_value_bets_tracked_alert_status
    on public.value_bets_tracked (alert_status, tracked_at desc);

create index if not exists idx_value_bets_tracked_clv_pending
    on public.value_bets_tracked (closing_odds, settlement_at);

commit;
