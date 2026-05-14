create table if not exists public.picks (
  id uuid primary key default gen_random_uuid(),
  match_name text not null,
  league text,
  market text not null,
  selection text not null,
  line numeric(10,4),
  my_odd numeric(10,4) not null,
  model_probability numeric(10,6) not null,
  model_fair_odd numeric(10,4),
  predicted_value numeric(10,4),
  implied_probability numeric(10,6) not null,
  edge numeric(10,6) not null,
  ev numeric(10,6) not null,
  full_kelly numeric(10,6) not null,
  kelly_fraction numeric(10,6) not null,
  recommended_stake numeric(10,4) not null,
  bankroll_snapshot numeric(10,4) not null,
  bet_type text not null default 'real',
  status text not null default 'pending',
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint picks_my_odd_positive check (my_odd > 1.0),
  constraint picks_market_check check (market in ('winner', 'firstblood', 'firsttower', 'kills')),
  constraint picks_bet_type_check check (bet_type in ('real', 'paper')),
  constraint picks_status_check check (status in ('pending', 'won', 'lost', 'void'))
);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = timezone('utc', now());
  return new;
end;
$$;

drop trigger if exists trg_picks_updated_at on public.picks;
create trigger trg_picks_updated_at
before update on public.picks
for each row
execute function public.set_updated_at();

grant usage on schema public to anon, authenticated, service_role;
grant all on table public.picks to anon, authenticated, service_role;
grant all on all sequences in schema public to anon, authenticated, service_role;
