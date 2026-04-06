import torch

class GradientDiagnostics:
    def __init__(self, objectives):
        self.objectives = objectives

    def _compute_cosine_similarity(self, grads_per_obj):
        if len(self.objectives) != 2:
            return {}

        name1 = self.objectives[0].name
        name2 = self.objectives[1].name

        g1 = grads_per_obj.get(name1)
        g2 = grads_per_obj.get(name2)

        if g1 is None or g2 is None:
            return {f"{name1}__{name2}": None}

        cos = torch.dot(g1, g2) / (torch.norm(g1) * torch.norm(g2) + 1e-12)
        return {f"{name1}__{name2}": float(cos.detach().cpu())}

    def collect(self, losses, params):
        grad_norms = {}
        grads_per_obj = {}

        for obj, loss in zip(self.objectives, losses):
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                allow_unused=True
            )

            flat = [g.reshape(-1) for g in grads if g is not None]

            if flat:
                gvec = torch.cat(flat)
                grad_norms[obj.name] = float(torch.norm(gvec).detach().cpu())
                grads_per_obj[obj.name] = gvec.detach()
            else:
                grad_norms[obj.name] = 0.0
                grads_per_obj[obj.name] = None

        cosine_dict = self._compute_cosine_similarity(grads_per_obj)

        return grad_norms, cosine_dict