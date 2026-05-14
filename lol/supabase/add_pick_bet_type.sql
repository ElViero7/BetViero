alter table public.picks
add column if not exists bet_type text not null default 'real';

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'picks_bet_type_check'
      and conrelid = 'public.picks'::regclass
  ) then
    alter table public.picks
    add constraint picks_bet_type_check check (bet_type in ('real', 'paper'));
  end if;
end $$;
