def thermal_causal_effect_loss(
        factual_logits,
        counterfactual_logits,
        labels,
        margin="adaptive",
        ignore_index=None):
    factual_prob = F.softmax(factual_logits.float(), dim=1)
    counterfactual_prob = F.softmax(counterfactual_logits.float(), dim=1)

    gather_labels = labels
    if ignore_index is not None:
        gather_labels = gather_labels.masked_fill(gather_labels == ignore_index, 0)
    gather_labels = gather_labels.clamp(min=0, max=factual_logits.shape[1] - 1).unsqueeze(1)

    factual_gt_prob = factual_prob.gather(1, gather_labels).squeeze(1)
    counterfactual_gt_prob = counterfactual_prob.gather(1, gather_labels).squeeze(1)
    thermal_effect = factual_gt_prob - counterfactual_gt_prob
    # counterfactual_uncertainty = 1.0 - counterfactual_gt_prob.detach()

    valid_mask = torch.ones_like(labels, dtype=torch.bool)
    if ignore_index is not None:
        valid_mask = labels != ignore_index

    if margin is None or (isinstance(margin, str) and margin.lower() == "adaptive"):
        target_margin = torch.zeros_like(thermal_effect)
        with torch.no_grad():
            for cls_idx in labels[valid_mask].unique():
                cls_mask = valid_mask & (labels == cls_idx)
                cls_effect = thermal_effect.detach()[cls_mask].clamp_min(0.0)
                if cls_effect.numel() > 0:
                    target_margin[cls_mask] = cls_effect.mean()
    else:
        target_margin = float(margin)

    loss = F.relu(target_margin+0.05 - thermal_effect)
    # loss = F.relu(target_margin+0.05 - thermal_effect)
    # loss = loss * counterfactual_uncertainty

    if ignore_index is not None:
        loss = loss[valid_mask]

    if loss.numel() == 0:
        return factual_logits.new_tensor(0.0)
    return loss.mean()





