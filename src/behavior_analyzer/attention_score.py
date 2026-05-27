class AttentionScorer:
    def __init__(self, rates, alpha=0.15):
        self.rates = rates
        self.alpha = alpha
        self.raw_score = 100.0
        self.display_score = 100.0

    def update(self, state, dt):
        rate = self.rates.get(state, self.rates.get("DISTRACTED", -2.0))
        self.raw_score = max(0.0, min(100.0, self.raw_score + rate * dt))
        self.display_score = self.alpha * self.raw_score + (1.0 - self.alpha) * self.display_score
        return self.raw_score, self.display_score
