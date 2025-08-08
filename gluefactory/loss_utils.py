import torch 
import torch.nn.functional as F





# Fonction de normalisation inchangée
def normalize_descriptors(descriptors):
    return F.normalize(descriptors, p=2, dim=-1)



# Ajouter les fonctions de perte personnalisée
def compute_consistency_loss(matches):
    """Maximise la consistance (et donc les matching scores) en la formulant comme une vraie perte"""
    matches0 = matches['matches0']
    matches1 = matches['matches1']
    scores0 = matches['matching_scores0']
    scores1 = matches['matching_scores1']

    valid_matches = (matches0 != -1) & (matches1 != -1)
    if valid_matches.sum() == 0:
        return torch.tensor(0.0, device=matches0.device)

    # Consistance forward-backward
    forward_matches = matches1.gather(1, matches0.clamp(min=0))
    backward_matches = matches0.gather(1, matches1.clamp(min=0))

    expected0 = torch.arange(matches0.size(1), device=matches0.device)[None]
    expected1 = torch.arange(matches1.size(1), device=matches1.device)[None]

    forward_consistency = (forward_matches == expected0).float()
    backward_consistency = (backward_matches == expected1).float()

    # Moyenne pondérée
    consistency = (forward_consistency * scores0 + backward_consistency * scores1) / 2

    # En pratique, on peut faire :
    loss = -torch.log(consistency[valid_matches] + 1e-6).mean()
    return loss



def custom_loss(matches, data, matcher):
    """Fonction de perte personnalisée"""
    # Perte de base
    losses, metrics = matcher.loss(matches, data, use_gt=False)
    base_loss = losses['total']
   
    # Perte de matching
    matching_scores = matches['matching_scores0']
    eps = 1e-6
    normalized_score = (matching_scores.mean() - 0.4) / (1.0 - 0.4 + eps)  # centré sur 0.4
    normalized_score = torch.clamp(normalized_score, min=eps, max=1.0)
    matching_loss = -torch.log(normalized_score)
   
    # Perte de consistance
    consistency_loss = compute_consistency_loss(matches)
    if torch.isnan(consistency_loss):
        consistency_loss = torch(0.0, device=matching_scores.device)
   
    # Perte de diversité des descripteurs
    desc0 = data['descriptors0']
    desc1 = data['descriptors1']
    std0 = torch.clamp(torch.std(desc0,dim=1), min=1e-6)
    std1 = torch.clamp(torch.std(desc1,dim=1), min=1e-6)
    diversity_loss = -(std0.mean() + std1.mean())
   
    # Combiner les pertes
    total_loss = (base_loss +
                  0.1 * matching_loss +
                  1.0 * consistency_loss +
                  0.01 * diversity_loss)
    if torch.isnan(total_loss):
        print("Warning:NaN detected in loss calculation")
        tota_loss = base_loss
   
    losses['total'] = total_loss
    losses['matching'] = matching_loss.detach()  # Pour le monitoring
    losses['consistency'] = consistency_loss.detach()
    losses['diversity'] = diversity_loss.detach()
   
    return losses, metrics


def check_gradients(model, name):
    """Vérifie les gradients pour détecter les anomalies"""
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                return False
    return True