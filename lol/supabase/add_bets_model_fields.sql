alter table public.bets
add column if not exists model_fair_odd numeric(10,4),
add column if not exists model_selected_team text,
add column if not exists picked_team text;
