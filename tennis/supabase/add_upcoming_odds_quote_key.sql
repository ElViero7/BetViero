alter table tennis.upcoming_odds
  add column if not exists quote_key text;

update tennis.upcoming_odds
set quote_key = concat_ws(
  '|',
  source,
  external_event_id,
  bookmaker_key,
  market_key,
  selection,
  coalesce(to_char(line, 'FM999999990.0000'), 'null')
)
where quote_key is null;

create unique index if not exists idx_tennis_upcoming_odds_quote_key
  on tennis.upcoming_odds (quote_key);
