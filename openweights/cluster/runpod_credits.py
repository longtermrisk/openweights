"""Notice quickly when the RunPod account is out of credits, and when it is topped up.

RunPod refuses to start a pod when the account balance does not cover one hour of
it. Without this module the manager counted every such refusal as a failure of the
GPU type, so within minutes all GPU types were on an escalating provisioning
cooldown of up to six hours. The cooldowns outlived the top-up: workers only
started again after a restart of the manager.

The monitor reads the account balance from RunPod and pauses provisioning as a
whole while the account cannot pay. A refusal for lack of credits is never counted
against a GPU type. While paused the balance is read every cycle, so provisioning
resumes within one poll interval of a top-up.
"""

import logging
import os
import time
from typing import Callable, Optional

from openweights.cluster.start_runpod import GPU_COST_PER_HOUR, parse_hardware_config

logger = logging.getLogger(__name__)

# Below this balance (USD) no new pods are started, whatever they cost.
MIN_BALANCE_USD = float(os.getenv("OW_RUNPOD_MIN_BALANCE_USD", "5"))
# How often the balance is read while provisioning works.
BALANCE_CHECK_INTERVAL_SECONDS = int(
    os.getenv("OW_RUNPOD_BALANCE_CHECK_INTERVAL_SECONDS", "60")
)
# How often it is read while provisioning is paused (0: every cycle).
PAUSED_BALANCE_CHECK_INTERVAL_SECONDS = int(
    os.getenv("OW_RUNPOD_PAUSED_BALANCE_CHECK_INTERVAL_SECONDS", "0")
)
# While paused with a balance that looks sufficient (or cannot be read), one pod
# start is tried this often to find out whether RunPod accepts again.
CREDIT_PROBE_INTERVAL_SECONDS = int(
    os.getenv("OW_RUNPOD_CREDIT_PROBE_INTERVAL_SECONDS", "300")
)
# A top-up smaller than this is not taken for one (refunds of a few cents).
TOP_UP_EPSILON_USD = 0.5

# RunPod's wording for "the balance does not cover this pod". Only used when the
# balance itself does not settle the question (it cannot be read, or RunPod wants
# more than the price table here says).
OUT_OF_CREDITS_ERROR_PATTERNS = [
    "insufficient balance",
    "insufficient funds",
    "insufficient credit",
    "not enough balance",
    "not enough funds",
    "not enough credit",
    "balance is too low",
    "balance too low",
    "low balance",
    "add funds",
    "add credits",
    "out of credits",
    "out of funds",
]

BALANCE_QUERY = "query { myself { clientBalance } }"


def is_out_of_credits_error(error: Exception | str) -> bool:
    """Check whether an error message says the RunPod balance is too low."""
    msg = str(error).lower()
    return any(pattern in msg for pattern in OUT_OF_CREDITS_ERROR_PATTERNS)


def fetch_runpod_balance(runpod_client) -> float:
    """Return the balance (USD) of the account the client's API key belongs to."""
    from runpod.api.graphql import run_graphql_query

    response = run_graphql_query(BALANCE_QUERY, api_key=runpod_client.api_key)
    return float(response["data"]["myself"]["clientBalance"])


def hourly_cost_usd(hardware_type: str) -> float:
    """Approximate price of one hour of a hardware type; 0 if the GPU is not listed."""
    try:
        count, gpu = parse_hardware_config(hardware_type)
    except ValueError:
        return 0.0
    return count * GPU_COST_PER_HOUR.get(gpu, 0.0)


class RunpodCreditMonitor:
    """Pause provisioning while the RunPod account is out of credits.

    Usage, once per manager cycle and per pod start:

    * ``can_provision()`` before starting pods. False while paused.
    * ``record_failure(hardware_type, error)`` when a pod start failed. True if
      the cause is the balance: the caller then stops starting pods and does not
      count the failure against the GPU type.
    * ``record_success()`` when a pod started.
    """

    def __init__(
        self,
        fetch_balance: Callable[[], float],
        min_balance_usd: float = MIN_BALANCE_USD,
        check_interval_seconds: int = BALANCE_CHECK_INTERVAL_SECONDS,
        paused_check_interval_seconds: int = PAUSED_BALANCE_CHECK_INTERVAL_SECONDS,
        probe_interval_seconds: int = CREDIT_PROBE_INTERVAL_SECONDS,
        now_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.fetch_balance = fetch_balance
        self.min_balance_usd = min_balance_usd
        self.check_interval_seconds = check_interval_seconds
        self.paused_check_interval_seconds = paused_check_interval_seconds
        self.probe_interval_seconds = probe_interval_seconds
        self.now_fn = now_fn or time.time
        #: Last balance read from RunPod; None if it could not be read.
        self.balance: Optional[float] = None
        self.paused = False
        self.paused_since: Optional[float] = None
        self.pause_reason: Optional[str] = None
        self._balance_at_pause: Optional[float] = None
        self._last_check_at: Optional[float] = None
        self._next_probe_at = 0.0
        self._balance_unreadable_logged = False

    def refresh(self, force: bool = False) -> Optional[float]:
        """Read the balance if it is due. Returns it, or None if RunPod did not say."""
        now = self.now_fn()
        interval = (
            self.paused_check_interval_seconds
            if self.paused
            else self.check_interval_seconds
        )
        if (
            not force
            and self._last_check_at is not None
            and now - self._last_check_at < interval
        ):
            return self.balance
        self._last_check_at = now
        try:
            self.balance = float(self.fetch_balance())
            self._balance_unreadable_logged = False
        except Exception as e:
            # A stale balance would be worse than none: it could keep the pause
            # after a top-up, or hide that the credits ran out.
            self.balance = None
            if not self._balance_unreadable_logged:
                logger.warning(
                    "Could not read the RunPod balance (%s). Out-of-credits is "
                    "then only recognised by RunPod's error message.",
                    e,
                )
                self._balance_unreadable_logged = True
        return self.balance

    def _pause(self, reason: str) -> None:
        now = self.now_fn()
        self.paused = True
        self.paused_since = now
        self.pause_reason = reason
        self._balance_at_pause = self.balance
        self._next_probe_at = now + self.probe_interval_seconds
        logger.error(
            "RunPod account is out of credits (%s). Not starting workers; jobs stay "
            "pending. The balance is checked every cycle and workers start again "
            "once the account is topped up.",
            reason,
        )

    def _resume(self, reason: str) -> None:
        paused_for = (
            int(self.now_fn() - self.paused_since)
            if self.paused_since is not None
            else 0
        )
        self.paused = False
        self.paused_since = None
        self.pause_reason = None
        self._balance_at_pause = None
        logger.info(
            "RunPod credits are back (%s) after %d s. Starting workers again.",
            reason,
            paused_for,
        )

    def _balance_text(self) -> str:
        return "unknown" if self.balance is None else f"${self.balance:.2f}"

    def can_provision(self) -> bool:
        """Whether pods may be started in this cycle."""
        balance = self.refresh()
        if not self.paused:
            if balance is not None and balance < self.min_balance_usd:
                self._pause(
                    f"balance ${balance:.2f} is below ${self.min_balance_usd:.2f}"
                )
                return False
            return True

        if balance is not None:
            if balance < self.min_balance_usd:
                return False
            if (
                self._balance_at_pause is not None
                and balance >= self._balance_at_pause + TOP_UP_EPSILON_USD
            ):
                self._resume(
                    f"balance ${balance:.2f}, was ${self._balance_at_pause:.2f}"
                )
                return True

        # The balance cannot be read, or it looks sufficient but RunPod refused:
        # try one pod start now and then. record_success() ends the pause.
        now = self.now_fn()
        if now >= self._next_probe_at:
            self._next_probe_at = now + self.probe_interval_seconds
            logger.info(
                "Trying one worker start to see whether RunPod accepts again "
                "(balance %s).",
                self._balance_text(),
            )
            return True
        return False

    def record_failure(self, hardware_type: str, error: Exception | str) -> bool:
        """Classify a failed pod start. True if the account's balance is the cause."""
        balance = self.refresh(force=True)
        needed = max(self.min_balance_usd, hourly_cost_usd(hardware_type))
        if balance is not None and balance < needed:
            reason = (
                f"balance ${balance:.2f} does not cover one hour of {hardware_type} "
                f"(about ${needed:.2f}); RunPod said: {error}"
            )
        elif is_out_of_credits_error(error):
            reason = f"balance {self._balance_text()}; RunPod said: {error}"
        else:
            return False
        if self.paused:
            # A probe that failed: stay paused, wait for the next one or a top-up.
            self._balance_at_pause = balance
        else:
            self._pause(reason)
        return True

    def record_success(self) -> None:
        """A pod started: the account can pay."""
        if self.paused:
            self._resume(f"a worker started, balance {self._balance_text()}")

    def status_line(self) -> str:
        """One line for the log while paused."""
        return (
            f"Provisioning paused: RunPod account is out of credits "
            f"(balance {self._balance_text()}; {self.pause_reason}). Jobs stay pending."
        )
