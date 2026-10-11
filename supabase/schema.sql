-- Run this in the Supabase Dashboard: SQL Editor -> New query -> paste -> Run
-- Access is service-role only: RLS is enabled with no public policies,
-- so only the server-side API (using the service_role key) can read/write.

create table if not exists public.holidays (
  id bigint generated always as identity primary key,
  date date not null,
  name text not null,
  category text not null check (category in ('national', 'regional', 'industrial')),
  impact text not null check (impact in ('high', 'medium', 'low')),
  notes text not null default '',
  unique (date, name)
);

create table if not exists public.forecast_logs (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  generated_at_ist text not null,
  n_blocks int not null,
  forecast_start_ist text not null,
  forecast_end_ist text not null,
  model text not null,
  holdout_mape numeric,
  payload jsonb not null
);

alter table public.holidays enable row level security;
alter table public.forecast_logs enable row level security;
