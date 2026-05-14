create schema if not exists tennis;

create extension if not exists pgcrypto;

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
grant all on table tennis.tennis_picks to anon, authenticated, service_role;
