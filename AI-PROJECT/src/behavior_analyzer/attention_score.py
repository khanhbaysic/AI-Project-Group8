import logging

from src.states import DISTRACTED


LOGGER = logging.getLogger(__name__)


class AttentionScorer:
    """Continuous attention score with raw integration and EMA smoothing.

    Each state has a per-second rate in ``CONFIG["attention_rates"]``. Every
    update changes the raw score by ``rate * dt`` and clamps it to [0, 100].
    For example, two seconds of ``DISTRACTED`` at -2.0 points/second lowers
    the raw score by about 4 points.

    The dashboard value is smoothed with exponential moving average:

        display = alpha * raw + (1 - alpha) * previous_display

    This keeps the live score readable instead of jumping on every frame. If a
    state is missing from the rate config, the old fallback behavior is kept:
    use the ``DISTRACTED`` rate, but log a warning once so the config can be
    corrected.
    """

    def __init__(self, rates, alpha=0.15):
        self.rates = rates
        self.alpha = alpha
        self.raw_score = 100.0
        self.display_score = 100.0
        self._warned_missing_states = set()

    def update(self, state, dt):
        if state in self.rates:
            rate = self.rates[state]
        else:
            rate = self.rates.get(DISTRACTED, -2.0)
            if state not in self._warned_missing_states:
                LOGGER.warning(
                    "Missing attention rate for state %r; using %r rate %.2f",
                    state,
                    DISTRACTED,
                    rate,
                )
                self._warned_missing_states.add(state)
        self.raw_score = max(0.0, min(100.0, self.raw_score + rate * dt))
        self.display_score = self.alpha * self.raw_score + (1.0 - self.alpha) * self.display_score
        return self.raw_score, self.display_score
