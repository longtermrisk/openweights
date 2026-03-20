import openweights
import time
import sys
import os

# Add path for slack MCP - we'll use subprocess to call slack instead
# since MCP tools aren't available in subprocess. We'll write results to a file
# and the main process will send the Slack message.

JOB_ID = 'mftjob-52e5f3a367f9-bma-7b-sdft-v4'
SLACK_CHANNEL = 'C0ALAAJTM8U'
MAX_PENDING_POLLS = 30  # 30 minutes

def format_completions(completions_event):
    samples = completions_event.get('data', {}).get('samples', [])
    lines = [f"*Job `{JOB_ID}` — completions found* (step {completions_event.get('data', {}).get('step', '?')})"]
    lines.append("")
    for i, s in enumerate(samples, 1):
        prompt_tail = s.get('prompt_tail', '')
        response = s.get('response', '')
        n_tokens = s.get('response_tokens', '?')
        # Truncate long responses
        if len(response) > 300:
            response = response[:300] + '…'
        lines.append(f"*Sample {i}* ({n_tokens} tokens)")
        lines.append(f"_Prompt tail:_ `{prompt_tail[:200]}`")
        lines.append(f"_Response:_ {response}")
        lines.append("")
    return "\n".join(lines)

def main():
    client = openweights.OpenWeights()
    poll_count = 0
    pending_warned = False

    while True:
        poll_count += 1
        j = client.jobs.retrieve(JOB_ID)
        status = j.status
        print(f"[Poll {poll_count}] status={status}, runs={len(j.runs) if j.runs else 0}", flush=True)

        # Check for terminal states
        if status in ('failed', 'canceled', 'cancelled'):
            msg = f":x: Job `{JOB_ID}` reached terminal state: *{status}*. No completions were found before it ended."
            with open('/tmp/poll_result.txt', 'w') as f:
                f.write(f"TERMINAL\n{msg}")
            print(f"Job terminal: {status}", flush=True)
            return

        # Check for completions in run events
        if j.runs:
            events = j.runs[0].events or []
            completions = [e for e in events if e.get('data', {}).get('tag') == 'completions']
            if completions:
                # Use the first completions event
                msg = format_completions(completions[0])
                with open('/tmp/poll_result.txt', 'w') as f:
                    f.write(f"COMPLETIONS\n{msg}")
                print("Found completions!", flush=True)
                return

        # Check if job completed without completions
        if status == 'completed':
            msg = f":white_check_mark: Job `{JOB_ID}` completed but no completions events were found in its run events."
            with open('/tmp/poll_result.txt', 'w') as f:
                f.write(f"TERMINAL\n{msg}")
            print("Job completed, no completions found.", flush=True)
            return

        # Warn if still pending after 30 polls
        if poll_count >= MAX_PENDING_POLLS and not pending_warned and status == 'pending':
            msg = f":hourglass: Job `{JOB_ID}` has been *pending for 30+ minutes* and hasn't started yet. Please check the job queue."
            with open('/tmp/poll_result_pending.txt', 'w') as f:
                f.write(f"PENDING_WARN\n{msg}")
            print("Pending warning written.", flush=True)
            pending_warned = True
            # Continue polling

        time.sleep(60)

if __name__ == '__main__':
    main()
