# OpenWeights 0.22.1

The cluster manager notices within a minute that the RunPod account is out of
credits, starts no pods while that lasts, and starts them again within one cycle
(15 s) of a top-up.

## What was wrong

RunPod refuses to start a pod when the account balance does not cover one hour of
it. The manager counted each refusal as a failure of the GPU type it had asked for.
Three failures put a GPU type on a cooldown that grows from 1 minute to 6 hours, so
an account without credits drove every GPU type to the top of that ladder. The
cooldowns live in the manager's memory and outlasted the top-up: on 2026-11-26 one
organization had 57 pending jobs, a funded account and `1x A100` / `1x A100S` locked
out for another hour and a half, until the manager was restarted. It happened again
the next day.

## What changes

`openweights/cluster/runpod_credits.py`: `RunpodCreditMonitor` reads the balance of
the organization's RunPod account (`myself { clientBalance }`, with the key the
manager already uses).

- Before starting pods, at most once a minute: below `OW_RUNPOD_MIN_BALANCE_USD`
  (default 5) provisioning is paused. No worker row, no pod start, jobs stay
  pending.
- When a pod start fails the balance is read at once. If it does not cover one hour
  of that hardware, or RunPod's message says the balance is too low, the failure
  pauses provisioning instead of counting against the GPU type, and the other GPU
  types are not tried.
- While paused the balance is read every cycle. A top-up ends the pause on the next
  cycle. If the balance looks sufficient but RunPod refused for lack of credits, or
  the balance cannot be read, one pod start is tried every 5 minutes
  (`OW_RUNPOD_CREDIT_PROBE_INTERVAL_SECONDS`); a start that works ends the pause.
- The pause and its end are logged ("RunPod account is out of credits ...",
  "RunPod credits are back ..."), with the balance, and show in the dashboard's
  Cluster tab.

A RunPod key that may not read the balance changes nothing compared to before,
except that refusals whose message names the balance are recognised.

Not changed: workers that RunPod stops when the balance reaches zero are cleaned up
as before (no ping for two minutes), and their jobs go back to pending. Nobody is
notified outside the log.

No migration, no new images; `IMAGE_VERSION` stays v0.16.0. Only the cluster manager
needs the upgrade.

## Verification

- `tests/test_runpod_credits.py`: pause on a low balance before any pod start;
  resume on the cycle after a top-up; a refusal with an empty account stops the cycle
  after one attempt and blames no GPU type; a refusal with a funded account is left
  to the per-GPU cooldowns; unreadable balance.
- The balance query was run against a real RunPod account with a manager's key.
- Not verified: RunPod's exact wording of the refusal (the logs of the two incidents
  were not at hand), which is why the decision rests on the balance and the message
  patterns are only a second line. Not run against an account that is really empty.
