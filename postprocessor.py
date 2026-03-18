import torch

class FairPostProcessor:
    def __init__(self, model, objectives, selector, lr=1e-2, epochs=100):
        self.model = model
        self.objectives = objectives
        self.selector = selector
        self.lr = lr
        self.epochs = epochs

        self.pareto_points_ = []
        self.threshold_history_ = []
        self.best_thresholds_ = None

    def fit(self, probs, y_true, sensitive_attr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        probs = torch.tensor(probs, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.long)
        sensitive_attr = torch.tensor(sensitive_attr, dtype=torch.long)

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            scores = self.model(probs)

            losses = []
            for obj in self.objectives:
                loss = obj(scores, y_true, sensitive_attr)
                losses.append(loss)

            total_loss = sum(losses)  # só para começar
            total_loss.backward()
            optimizer.step()

            point = [loss.item() for loss in losses]
            self.pareto_points_.append(point)
            self.threshold_history_.append(self.model.thresholds.detach().clone())

        idx = self.selector.select(self.pareto_points_)
        self.best_thresholds_ = self.threshold_history_[idx]
        return self

    def predict(self, probs):
        probs = torch.tensor(probs, dtype=torch.float32)
        with torch.no_grad():
            self.model.thresholds.copy_(self.best_thresholds_)
            scores = self.model(probs)
            return torch.argmax(scores, dim=1).cpu().numpy()

    def get_thresholds(self):
        return self.best_thresholds_