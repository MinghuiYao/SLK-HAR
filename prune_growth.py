import torch
import math
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def momentum_redistribution(masking, name, weight, mask):
    grad = masking.get_momentum_for_weight(weight)
    mean_magnitude = torch.abs(grad[mask.bool()]).mean().item()
    return mean_magnitude

def magnitude_redistribution(masking, name, weight, mask):
    mean_magnitude = torch.abs(weight)[mask.bool()].mean().item()
    return mean_magnitude

def nonzero_redistribution(masking, name, weight, mask):
    nonzero = (weight !=0.0).sum().item()
    return nonzero

def no_redistribution(masking, name, weight, mask):
    num_params = masking.baseline_nonzero
    n = weight.numel()
    return n/float(num_params)

'''
                PRUNE
'''
def magnitude_prune(masking, mask, weight, name):
    num_remove = math.ceil(masking.prune_rate*masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)
    if num_remove == 0.0: return weight.data != 0.0

    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
    mask.data.view(-1)[idx[:k]] = 0.0
    return mask

def global_magnitude_prune(masking):
    prune_rate = 0.0
    for name in masking.name2prune_rate:
        if name in masking.masks:
            prune_rate = masking.name2prune_rate[name]
    tokill = math.ceil(prune_rate*masking.baseline_nonzero)
    total_removed = 0
    prev_removed = 0
    while total_removed < tokill*(1.0-masking.tolerance) or (total_removed > tokill*(1.0+masking.tolerance)):
        total_removed = 0
        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks: continue
                remain = (torch.abs(weight.data) > masking.prune_threshold).sum().item()
                total_removed += masking.name2nonzeros[name] - remain

        if prev_removed == total_removed: break
        prev_removed = total_removed
        if total_removed > tokill*(1.0+masking.tolerance):
            masking.prune_threshold *= 1.0-masking.increment
            masking.increment *= 0.99
        elif total_removed < tokill*(1.0-masking.tolerance):
            masking.prune_threshold *= 1.0+masking.increment
            masking.increment *= 0.99

    for module in masking.modules:
        for name, weight in module.named_parameters():
            if name not in masking.masks: continue
            masking.masks[name][:] = torch.abs(weight.data) > masking.prune_threshold

    return int(total_removed)


def magnitude_and_negativity_prune(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name]*masking.name2nonzeros[name])
    if num_remove == 0.0: return weight.data != 0.0

    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + (num_remove/2.0))

    # remove all weights which absolute value is smaller than threshold
    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
    mask.data.view(-1)[idx[:k]] = 0.0

    # remove the most negative weights
    x, idx = torch.sort(weight.data.view(-1))
    mask.data.view(-1)[idx[:math.ceil(num_remove/2.0)]] = 0.0

    return mask

'''
                GROWTH
'''

def random_growth(masking, name, new_mask, total_regrowth, weight):
    n = (new_mask==0).sum().item()
    if n == 0: return new_mask
    expeced_growth_probability = (total_regrowth/n)
    new_weights = torch.rand(new_mask.shape).to(device) < expeced_growth_probability
    return new_mask.bool() | new_weights

def random_unfired_growth(masking, name, new_mask, total_regrowth, weight):
    n = (new_mask == 0).sum().item()
    if n == 0: return new_mask
    num_nonfired_weights = (masking.fired_masks[name] == 0).sum().item()

    if total_regrowth <= num_nonfired_weights:
        idx = (masking.fired_masks[name].flatten() == 0).nonzero()
        indices = torch.randperm(len(idx))[:total_regrowth]
        new_mask.data.view(-1)[idx[indices]] = 1.0
    else:
        new_mask[masking.fired_masks[name] == 0] = 1.0
        n = (new_mask == 0).sum().item()
        expeced_growth_probability = ((total_regrowth - num_nonfired_weights) / n)
        new_weights = torch.rand(new_mask.shape).to(device) < expeced_growth_probability
        new_mask = new_mask.byte() | new_weights
    return new_mask

def gradient_growth(masking, name, new_mask, total_regrowth, weight):
    grad = masking.get_gradient_for_weights(weight)
    if grad.dtype == torch.float16:
        grad = grad * (new_mask == 0).half()
    else:
        grad = grad * (new_mask == 0).float()
    y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
    new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

    return new_mask

def mix_growth(masking, name, new_mask, total_regrowth, weight):
    gradient_grow = int(total_regrowth * masking.mix)
    random_grow = total_regrowth - gradient_grow
    grad = masking.get_gradient_for_weights(weight)

    if grad.dtype == torch.float16:
        grad = grad * (new_mask == 0).half()
    else:
        grad = grad * (new_mask == 0).float()
    y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
    new_mask.data.view(-1)[idx[:gradient_grow]] = 1.0

    n = (new_mask == 0).sum().item()
    expeced_growth_probability = (random_grow / n)
    new_weights = torch.rand(new_mask.shape).to(device) < expeced_growth_probability
    new_mask = new_mask.bool() | new_weights

    return new_mask


def momentum_growth(masking, name, new_mask, total_regrowth, weight):
    grad = masking.get_momentum_for_weight(weight)
    if grad.dtype == torch.float16:
        grad = grad*(new_mask==0).half()
    else:
        grad = grad*(new_mask==0).float()
    y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
    new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

    return new_mask

def momentum_neuron_growth(masking, name, new_mask, total_regrowth, weight):
    grad = masking.get_momentum_for_weight(weight)

    M = torch.abs(grad)
    if len(M.shape) == 2: sum_dim = [1]
    elif len(M.shape) == 4: sum_dim = [1, 2, 3]

    v = M.mean(sum_dim).data
    v /= v.sum()

    slots_per_neuron = (new_mask==0).sum(sum_dim)

    M = M*(new_mask==0).float()
    for i, fraction  in enumerate(v):
        neuron_regrowth = math.floor(fraction.item()*total_regrowth)
        available = slots_per_neuron[i].item()

        y, idx = torch.sort(M[i].flatten())
        if neuron_regrowth > available:
            neuron_regrowth = available
        threshold = y[-(neuron_regrowth)].item()
        if threshold == 0.0: continue
        if neuron_regrowth < 10: continue
        new_mask[i] = new_mask[i] | (M[i] > threshold)

    return new_mask


def global_momentum_growth(masking, total_regrowth):
    togrow = total_regrowth
    total_grown = 0
    last_grown = 0
    while total_grown < togrow*(1.0-masking.tolerance) or (total_grown > togrow*(1.0+masking.tolerance)):
        total_grown = 0
        total_possible = 0
        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks: continue

                new_mask = masking.masks[name]
                grad = masking.get_momentum_for_weight(weight)
                grad = grad*(new_mask==0).float()
                possible = (grad !=0.0).sum().item()
                total_possible += possible
                grown = (torch.abs(grad.data) > masking.growth_threshold).sum().item()
                total_grown += grown
        if total_grown == last_grown: break
        last_grown = total_grown


        if total_grown > togrow*(1.0+masking.tolerance):
            masking.growth_threshold *= 1.02
        elif total_grown < togrow*(1.0-masking.tolerance):
            masking.growth_threshold *= 0.98

    total_new_nonzeros = 0
    for module in masking.modules:
        for name, weight in module.named_parameters():
            if name not in masking.masks: continue

            new_mask = masking.masks[name]
            grad = masking.get_momentum_for_weight(weight)
            grad = grad*(new_mask==0).float()
            masking.masks[name][:] = (new_mask.bool() | (torch.abs(grad.data) > masking.growth_threshold)).float()
            total_new_nonzeros += new_mask.sum().item()
    return total_new_nonzeros




prune_funcs = {}
prune_funcs['magnitude'] = magnitude_prune
prune_funcs['SET'] = magnitude_and_negativity_prune
prune_funcs['global_magnitude'] = global_magnitude_prune

growth_funcs = {}
growth_funcs['random'] = random_growth
growth_funcs['random_unfired'] = random_unfired_growth
growth_funcs['momentum'] = momentum_growth
growth_funcs['gradient'] = gradient_growth
growth_funcs['mix'] = mix_growth
growth_funcs['momentum_neuron'] = momentum_neuron_growth
growth_funcs['global_momentum_growth'] = global_momentum_growth

redistribution_funcs = {}
redistribution_funcs['momentum'] = momentum_redistribution
redistribution_funcs['nonzero'] = nonzero_redistribution
redistribution_funcs['magnitude'] = magnitude_redistribution
redistribution_funcs['none'] = no_redistribution