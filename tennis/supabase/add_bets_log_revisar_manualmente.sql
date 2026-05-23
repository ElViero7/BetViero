-- SQL Migration: Add revisar_manualmente column to tennis.bets_log
alter table tennis.bets_log
add column if not exists revisar_manualmente boolean default false;
