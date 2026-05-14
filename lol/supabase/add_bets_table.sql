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

drop trigger if exists trg_bets_updated_at on betting.bets;
create trigger trg_bets_updated_at
before update on betting.bets
for each row
execute function betting.set_updated_at();
