#modeling.py
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config
from typing import Optional
from collections import Counter
from torch.distributions.normal import Normal
import os
softplus = nn.Softplus()

def _rank0():
    return os.environ.get("RANK", "0") == "0"

class HyperMoEShared(nn.Module):
    """Cross-layer shared hypernetwork bundle for HyperMoE."""
    def __init__(self, cfg, d_model, n_experts):
        super().__init__()
        self.n_experts_embedding = nn.Embedding(n_experts, cfg["experts_embedding_dim"])
        self.embedding_process = nn.Sequential(
            nn.Linear(cfg["experts_embedding_dim"], cfg["process_dim"]),
            nn.ReLU(),
            nn.Linear(cfg["process_dim"], cfg["hypernet_input"]),
        )
        self.param_gen = ParameterGenerator(cfg, d_model, d_model)
        self.adapter_layer = AdapterLayer(d_model, d_model, cfg["adapter_dim"])

class Expert(nn.Module):
    def __init__(self, d_model, d_ff, initializer_range=0.02, use_gelu=False):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.GELU() if use_gelu else nn.ReLU()
        self._reset_parameters(initializer_range)
    def forward(self, x):
        return self.w2(self.act(self.w1(x)))
    def _reset_parameters(self, initializer_range):
        nn.init.normal_(self.w1.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.w2.weight, mean=0.0, std=initializer_range)

def hyperfanin_init_weight(linear_layer, hypernet_in, mainnet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in * mainnet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    if linear_layer.bias is not None:
        nn.init.constant_(linear_layer.bias, 0.0)

def hyperfanin_init_bias(linear_layer, hypernet_in):
    bound = 1e-3 * math.sqrt(3.0 / hypernet_in)
    if linear_layer.bias is not None:
        nn.init.uniform_(linear_layer.bias, -bound, bound)

class SimpleGenerator(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super().__init__()
        self.hidden_dim = cfg["hypernetwork_bottleneck"]
        self.adapter_dim = cfg["adapter_dim"]
        self.linear1 = nn.Linear(cfg["hypernet_input"] + cfg["layer_emb_dim"], self.hidden_dim)
        self.activation_fn = nn.ReLU()
        self.weight_up = nn.Linear(self.hidden_dim, output_dim * self.adapter_dim)
        self.weight_down = nn.Linear(self.hidden_dim, input_dim * self.adapter_dim)
        self.bias_up = nn.Linear(self.hidden_dim, output_dim)
        self.bias_down = nn.Linear(self.hidden_dim, self.adapter_dim)
        hyperfanin_init_weight(self.weight_up, self.hidden_dim, self.adapter_dim)
        hyperfanin_init_weight(self.weight_down, self.hidden_dim, input_dim)
        hyperfanin_init_bias(self.bias_up, self.hidden_dim)
        hyperfanin_init_bias(self.bias_down, self.hidden_dim)

    def forward(self, x):
        x = self.activation_fn(self.linear1(x))
        return (self.weight_up(x), self.weight_down(x), self.bias_up(x), self.bias_down(x))

class ParameterGenerator(nn.Module):
    def __init__(self, cfg, input_size, output_size):
        super().__init__()
        self.cfg = cfg
        self.layer_embed = nn.Embedding(cfg["num_hidden_layers"], cfg["layer_emb_dim"])
        self.decoder = SimpleGenerator(cfg, input_size, output_size)

    def forward(self, hidden_inputs, layer_idx):
        layer_idx = torch.ones(hidden_inputs.size(0), hidden_inputs.size(1),
                               dtype=torch.long, device=hidden_inputs.device) * layer_idx
        layer_inputs = self.layer_embed(layer_idx)
        hidden_inputs = torch.cat([hidden_inputs, layer_inputs], dim=-1)
        return self.decoder(hidden_inputs)

class AdapterLayer(nn.Module):
    def __init__(self, input_size, output_size, adapter_dim):
        super().__init__()
        self.adapter_dim = adapter_dim
        self.input_dim = input_size
        self.output_dim = output_size
        self.adapter_down_weight = None
        self.adapter_down_bias = None
        self.adapter_up_weight = None
        self.adapter_up_bias = None
        self.hidden_act = nn.ReLU()
        self.adapter_down_manual = nn.Linear(self.input_dim, self.adapter_dim)
        self.adapter_up_manual = nn.Linear(self.adapter_dim, self.output_dim)
        nn.init.xavier_uniform_(self.adapter_up_manual.weight, gain=1e-4)
        nn.init.xavier_uniform_(self.adapter_down_manual.weight, gain=1e-4)
        nn.init.constant_(self.adapter_up_manual.bias, 0.0)
        nn.init.constant_(self.adapter_down_manual.bias, 0.0)

    def clear_adapter(self):
        self.adapter_down_weight = None
        self.adapter_down_bias = None
        self.adapter_up_weight = None
        self.adapter_up_bias = None

    def apply_adapter_params(self, bsz, lg, uw, dw, ub, db):
        self.adapter_down_weight = dw.view(bsz, lg, self.input_dim, self.adapter_dim)
        self.adapter_down_bias = db.view(bsz, lg, self.adapter_dim)
        self.adapter_up_weight = uw.view(bsz, lg, self.adapter_dim, self.output_dim)
        self.adapter_up_bias = ub.view(bsz, lg, self.output_dim)

    def forward(self, x):
        if self.adapter_down_weight is not None:
            wd = self.adapter_down_weight
            bu = self.adapter_up_bias
            wu = self.adapter_up_weight
            bd = self.adapter_down_bias
            x = x.to(wd.dtype)
            x = torch.einsum('bij,bijk->bik', x, wd) + bd
            x = self.hidden_act(x)
            x = torch.einsum('bik,bikj->bij', x, wu) + bu
        else:
            x = self.adapter_down_manual(x)
            x = self.hidden_act(x)
            x = self.adapter_up_manual(x)
        return x

class SparseDispatcher:
    def __init__(self, n_experts, gates):
        self._gates = gates
        self._n_experts = n_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates, as_tuple=False).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates, as_tuple=False)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            g = self._nonzero_gates
            if g.dim() == 1:
                g = g.unsqueeze(-1)
            while g.dim() < stitched.dim():
                g = g.unsqueeze(-1)
            g = g.to(device=stitched.device, dtype=stitched.dtype)
            stitched = stitched * g

        if stitched.dim() == 2:
            zeros = torch.zeros(self._gates.size(0), stitched.size(1),
                                device=stitched.device, dtype=stitched.dtype)
        else:
            zeros = torch.zeros(self._gates.size(0), stitched.size(-2), stitched.size(-1),
                                device=stitched.device, dtype=stitched.dtype)
        return zeros.index_add(0, self._batch_index, stitched)

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

def _gates_to_load(gates): return (gates > 0).sum(0)

def cv_squared(x):
    eps = 1e-10
    if x.shape[0] == 1: return torch.tensor([0.], device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean()**2 + eps)

def _prob_in_top_k(layer, clean_values, noisy_values, noise_stddev, noisy_top_values):
    batch = clean_values.size(0)
    m = noisy_top_values.size(1)
    top_values_flat = noisy_top_values.flatten()
    normal = Normal(layer.mean, layer.std)
    threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + layer.k
    threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
    is_in = torch.gt(noisy_values, threshold_if_in)
    threshold_positions_if_out = threshold_positions_if_in - 1
    threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
    prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
    prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
    return torch.where(is_in, prob_if_in, prob_if_out)

def noisy_top_k_gating_mixing(layer, x, train, noise_epsilon=1e-2):
    clean_logits = x @ layer.w_gate
    if layer.noisy_gating and train:
        raw_noise_stddev = x @ layer.w_noise
        noise_stddev = softplus(raw_noise_stddev) + noise_epsilon
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits
    else:
        logits = clean_logits
    logits = torch.softmax(logits, dim=-1)
    top_logits, top_indices = logits.topk(min(layer.k+1, layer.n_experts), dim=-1)
    top_k_logits = top_logits[:, :layer.k]
    top_k_indices = top_indices[:, :layer.k]
    top_k_gates = top_k_logits / (top_k_logits.sum(dim=-1, keepdim=True) + 1e-9)

    zeros = torch.zeros_like(logits, dtype=top_k_gates.dtype, device=x.device)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)

    ones = torch.ones_like(top_k_logits, dtype=top_k_gates.dtype, device=x.device)
    zeros_mask = torch.zeros_like(logits, dtype=top_k_gates.dtype, device=x.device)
    unselected_mask = torch.ones_like(logits, dtype=top_k_gates.dtype, device=x.device) - \
                      zeros_mask.scatter(1, top_k_indices, ones)
    gates_unselected = unselected_mask / (unselected_mask.sum(dim=-1, keepdim=True) + 1e-9)

    expert_mask = torch.nn.functional.one_hot(top_k_indices, num_classes=layer.n_experts).permute(2, 1, 0)

    if layer.noisy_gating and layer.k < layer.n_experts and train:
        load = (_prob_in_top_k(layer, clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
    else:
        load = _gates_to_load(gates)
    return gates, load, gates_unselected, expert_mask

class RecurrentRouter(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.gru = nn.GRUCell(d_model, hidden_dim)
        nn.init.xavier_uniform_(self.gru.weight_ih)
        nn.init.orthogonal_(self.gru.weight_hh)
        nn.init.zeros_(self.gru.bias_ih)
        nn.init.zeros_(self.gru.bias_hh)
    def forward(self, x, h_prev=None):
        B, T, H = x.shape
        x_flat = x.view(-1, H)
        x_in = x_flat.to(self.gru.weight_ih.dtype)
        if h_prev is None:
            h_prev = x_in.new_zeros(x_in.size(0), self.gru.hidden_size)
        h_new = self.gru(x_in, h_prev)
        return h_new

class XMoEThresholdRouter(nn.Module):
    def __init__(self, d_model, num_experts, threshold=0.90, capacity_factor=1.0, alpha=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.threshold = float(threshold)
        self.capacity_factor = float(capacity_factor)
        self.alpha = float(alpha)
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        self.last_scores = None
    def forward(self, x):
        B, T, H = x.shape
        N = B * T
        x_flat = x.view(N, H)
        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)
        self.last_scores = probs.detach()
        p_sorted, idx_sorted = torch.sort(probs, dim=-1, descending=True)
        csum = torch.cumsum(p_sorted, dim=-1)
        reached = (csum >= self.threshold)
        reached[:, -1] = True
        k_idx = reached.float().argmax(dim=-1)
        arangeE = torch.arange(probs.size(1), device=probs.device).view(1, -1)
        select_prefix = arangeE <= k_idx.view(-1, 1)
        assign_sorted = select_prefix
        assign_mask = torch.zeros_like(assign_sorted, dtype=torch.bool)
        assign_mask.scatter_(dim=1, index=idx_sorted, src=assign_sorted)
        rank_sorted = torch.arange(1, self.num_experts + 1, device=probs.device).view(1, -1).expand_as(p_sorted)
        R_sorted = p_sorted - rank_sorted.float()
        R_sorted = torch.where(assign_sorted, R_sorted, torch.full_like(R_sorted, -1e9))
        R = torch.empty_like(R_sorted)
        R.scatter_(dim=1, index=idx_sorted, src=R_sorted)
        capacity = int(math.ceil(N / self.num_experts) * self.capacity_factor)
        if not self.training:
            capacity = N
        capacity = max(1, min(capacity, N))
        R_col_sorted_vals, R_col_sorted_tok = torch.sort(R, dim=0, descending=True)
        keep_flags = torch.zeros_like(R, dtype=torch.bool)
        C = min(capacity, N)
        if C > 0:
            topC_tok = R_col_sorted_tok[:C, :]
            row_idx = topC_tok
            col_idx = torch.arange(self.num_experts, device=probs.device).view(1, -1).expand_as(topC_tok)
            keep_flags[row_idx, col_idx] = True
        final_mask = assign_mask & keep_flags
        scores_sel = torch.where(final_mask, probs, torch.zeros_like(probs))
        top1_scores, top1_idx = probs.max(dim=-1)
        one_hot = F.one_hot(top1_idx, num_classes=self.num_experts).float()
        fi = one_hot.mean(dim=0)
        pi = torch.zeros(self.num_experts, device=probs.device, dtype=probs.dtype)
        pi.index_add_(0, top1_idx, top1_scores)
        pi = pi / float(N)
        aux_loss = self.num_experts * (fi * pi).sum() * self.alpha
        return final_mask, scores_sel, aux_loss, top1_idx

class Router(nn.Module):
    def __init__(self, d_model, num_experts, top_k=1, alpha=0.01, seq_aux=False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha
        self.seq_aux = seq_aux
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.last_scores = None
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
    def forward(self, x):
        bsz, seq, h = x.shape
        logits = self.gate(x.view(-1, h))
        scores = F.softmax(logits, dim=-1)
        self.last_scores = scores.detach()
        topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        if self.top_k > 1:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-9)
        aux_loss = None
        if self.training and self.alpha > 0.0:
            one_hot = F.one_hot(topk_idx.view(-1), num_classes=self.num_experts)
            freq = one_hot.float().mean(0)
            Pi = scores.mean(0)
            fi = freq * self.num_experts
            aux_loss = (Pi * fi).sum() * self.alpha
        return topk_weight, topk_idx, aux_loss

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts,
                mode="switch", shared_expert=None, global_experts=None,
                alpha=0.01, capacity_factor=1.25, freq_dict=None,
                shared_router: RecurrentRouter=None,
                xmoe_threshold: float = 0.90,
                xmoe_capacity_factor: float = 1.0,
                xmoe_expert_mult: float = 0.25):
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.num_experts = num_experts
        self.shared_expert = shared_expert
        self.global_experts = global_experts
        self.capacity_factor = capacity_factor
        self.last_scores = None
        self.aux_alpha = alpha
        self.last_aux = {"balance": None, "distill": None, "overflow_rate": None}

        if mode == "xmoe":
            d_ff_expert = max(64, int(d_model * 4 * xmoe_expert_mult))
            self.experts = nn.ModuleList([Expert(d_model, d_ff_expert, use_gelu=True) for _ in range(num_experts)])
            self.xmoe_router = XMoEThresholdRouter(
                d_model=d_model,
                num_experts=num_experts,
                threshold=xmoe_threshold,
                capacity_factor=xmoe_capacity_factor,
                alpha=alpha
            )
        elif mode == "ours_com":
            self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(1)])
            self.num_experts = len(global_experts)
            assert shared_router is not None, "ours_com requires a shared_router"
            self.shared_router = shared_router
            self.h_ln = nn.LayerNorm(self.shared_router.gru.hidden_size)
            self.gate_head = nn.Linear(self.shared_router.gru.hidden_size, self.num_experts, bias=False)
            self.cond_dim = self.shared_router.gru.hidden_size
            self.score_dim = self.num_experts
            self.film_hidden = max(128, self.cond_dim // 4)
            self.score_proj = nn.Linear(self.score_dim, self.cond_dim, bias=False)
            self.score_ln = nn.LayerNorm(self.score_dim)
            self.cond_ln  = nn.LayerNorm(self.cond_dim)
            self.film_mlp = nn.Sequential(
                nn.Linear(self.cond_dim * 2, self.film_hidden, bias=True),
                nn.ReLU(),
                nn.Linear(self.film_hidden, 2 * self.cond_dim, bias=True)
            )
            nn.init.zeros_(self.film_mlp[-1].weight)
            nn.init.zeros_(self.film_mlp[-1].bias)
            self.proj_gamma_beta = None
            if self.cond_dim != d_model:
                self.proj_gamma_beta = nn.Linear(self.cond_dim, d_model, bias=False)

        elif mode == "ours_refine":
            self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(1)])
            self.num_experts = len(global_experts)
            assert shared_router is not None, "ours_refine requires a shared_router"
            self.shared_router = shared_router

            self.cond_dim = self.shared_router.gru.hidden_size
            self.score_dim = self.num_experts
            self.h_ln = nn.LayerNorm(self.cond_dim)

            self.prev_score_ln   = nn.LayerNorm(self.score_dim)
            self.prev_score_proj = nn.Linear(self.score_dim, self.cond_dim, bias=False)

            self.gate_head = nn.Linear(self.cond_dim * 2, self.num_experts, bias=False)

            self.last_scores = None

        elif mode == "hash":
            self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
            self.hash_router = HashRouter(
                vocab_size=50257,
                num_experts=num_experts,
                method="balanced",
                freq_dict=freq_dict
            )
        elif mode in {"switch", "gshard"}:
            self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
            self.router = Router(d_model, num_experts, top_k=(1 if mode=="switch" else 2), alpha=alpha)
        
        elif mode == "stablemoe":
            self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
            self.capacity_factor = capacity_factor
            self.expert_centroids = nn.Parameter(torch.empty(num_experts, d_model))
            nn.init.orthogonal_(self.expert_centroids, gain=0.1)

            self.stable_routing_dim = getattr(self, "stable_routing_dim", 50)
            self.vocab_size = getattr(self, "vocab_size", 50257)
            
            self.stable_stage1_steps = None
            self.stable_balance_alpha = getattr(self, "stable_balance_alpha", 0.3)

            self.register_buffer("_num_updates_buf", torch.zeros((), dtype=torch.long))
            self._stage2_frozen = False
            
            self._stable_root_ref = None

        elif global_experts is None and mode not in {"hypermoe"}:
            self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def _ddp_probe_loss_hyper(self, dtype, device, alpha=1e-8):
        if not hasattr(self, "experts_hypermoe"):
            return None
        d = self.input_size
        probe = torch.randn(8, d, device=device, dtype=dtype) * 1e-3
        loss = None
        for e in self.experts_hypermoe:
            y = e(probe)
            l = (y**2).mean()
            loss = l if loss is None else (loss + l)
        return alpha * loss if loss is not None else None

    def _ddp_probe_loss(self, dtype, device, alpha=1e-8, n_proj=1):
        if not hasattr(self, "experts"):
            return None
        d = self.d_model
        loss = None
        for _ in range(n_proj):
            probe = torch.randn(8, d, device=device, dtype=dtype) * 1e-3
            for e in self.experts:
                y = e(probe)
                l = (y**2).mean()
                loss = l if loss is None else (loss + l)
        if loss is None:
            return None
        return alpha * loss

    def _init_hypermoe(self, cfg, h, shared: Optional[nn.Module]=None):
        self.n_experts = self.num_experts
        self.k = int(cfg.get("k", 1))
        self.noisy_gating = bool(cfg.get("noisy_gating", True))
        self.input_size = h
        self.output_size = h

        self.w_gate  = nn.Parameter(torch.zeros(h, self.n_experts))
        self.w_noise = nn.Parameter(torch.zeros(h, self.n_experts))
        nn.init.normal_(self.w_gate,  mean=0.0, std=0.02)
        nn.init.normal_(self.w_noise, mean=0.0, std=0.02)
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std  = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

        self.experts_hypermoe = nn.ModuleList([Expert(h, h*4, use_gelu=True) for _ in range(self.n_experts)])

        use_hn = cfg.get("use_hypernet", True)
        if not use_hn:
            self.adapter_layer = None
            return

        if shared is not None:
            self.adapter_layer        = shared.adapter_layer
            self.n_experts_embedding  = shared.n_experts_embedding
            self.embedding_process    = shared.embedding_process
            self.param_gen            = shared.param_gen

            for p in self.adapter_layer.adapter_down_manual.parameters():
                p.requires_grad_(False)
            for p in self.adapter_layer.adapter_up_manual.parameters():
                p.requires_grad_(False)
        else:
            self.adapter_layer = AdapterLayer(self.input_size, self.output_size, cfg["adapter_dim"])
            self.n_experts_embedding = nn.Embedding(self.n_experts, cfg["experts_embedding_dim"])
            self.embedding_process = nn.Sequential(
                nn.Linear(cfg["experts_embedding_dim"], cfg["process_dim"]),
                nn.ReLU(),
                nn.Linear(cfg["process_dim"], cfg["hypernet_input"]),
            )
            self.param_gen = ParameterGenerator(cfg, self.input_size, self.output_size)

    @staticmethod
    def _make_finite(x: torch.Tensor) -> torch.Tensor:
        ok = torch.isfinite(x)
        if not ok.all():
            minv = x[ok].min() if ok.any() else torch.tensor(0., device=x.device, dtype=x.dtype)
            x = x.clone()
            x[~ok] = minv
        return x

    def _maybe_freeze_stage2(self):
        if self.mode != "stablemoe" or self._stage2_frozen or (not self.training):
            return
        root = self._stable_root_ref() if hasattr(self, "_stable_root_ref") else None
        assert root is not None, "StableMoE root ref missing"
        root.stablemoe_routing_weight.requires_grad_(False)
        root.stablemoe_distill_E.requires_grad_(False)
        self._stage2_frozen = True
        if self.training:
            print("🔒 StableMoE: Froze stablemoe_routing_weight & stablemoe_distill_E for Stage-2")

    def forward(self, x, input_ids=None, routing_state=None, global_step: Optional[int]=None):
        bsz, seq, h = x.shape
        routed_out = torch.zeros_like(x)
        balance_loss = None
        updated_routing_state = routing_state
        if self.global_experts is not None:
            experts = self.global_experts
        elif hasattr(self, "experts"):
            experts = self.experts
        else:
            experts = None

        if self.mode == "dense":
            routed_out = experts[0](x)
            return routed_out, None, updated_routing_state

        elif self.mode == "switch":
            top_scores, top_idx, balance_loss = self.router(x)
            top_idx = top_idx.view(-1)
            top_scores = top_scores.view(-1, 1)
            x_flat = x.view(-1, h)
            out_flat = torch.zeros_like(x_flat)
            capacity = int(self.capacity_factor * math.ceil(x_flat.size(0) / self.num_experts))
            if not self.training:
                capacity = x_flat.size(0)
            for eid in range(self.num_experts):
                idx = (top_idx == eid).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    if idx.numel() > capacity:
                        keep = idx[:capacity]
                        dropped = idx[capacity:]
                        out_flat[dropped] = x_flat[dropped]
                    else:
                        keep = idx
                    expert_out = experts[eid](x_flat[keep]) * top_scores[keep]
                    out_flat[keep] = expert_out.to(out_flat.dtype)
            routed_out = out_flat.view(bsz, seq, h)

            probe = self._ddp_probe_loss(dtype=x.dtype, device=x.device, alpha=getattr(self, "ddp_probe_alpha", 1e-8))
            balance_loss = probe

            return routed_out, balance_loss, updated_routing_state

        elif self.mode == "gshard":
            top_scores, top_idx, balance_loss = self.router(x)
            x_flat = x.view(-1, h)
            out_flat = torch.zeros_like(x_flat)
            capacity = int(self.capacity_factor * math.ceil(x_flat.size(0) * 2 / self.num_experts))
            if not self.training:
                capacity = x_flat.size(0)
            for eid in range(self.num_experts):
                for k in range(2):
                    idx = (top_idx[:, k] == eid).nonzero(as_tuple=True)[0]
                    if idx.numel() > 0:
                        score = top_scores[idx, k]
                        if self.training and k == 1:
                            randmask = torch.rand_like(score)
                            mask = score * 2 > randmask
                            idx = idx[mask]
                            score = score[mask]
                        if idx.numel() > 0:
                            score = score.unsqueeze(-1)
                            if idx.numel() > capacity:
                                keep = idx[:capacity]
                                dropped = idx[capacity:]
                                out_flat[dropped] += x_flat[dropped]
                            else:
                                keep = idx
                            expert_out = experts[eid](x_flat[keep]) * score[:keep.size(0)]
                            out_flat[keep] = (out_flat[keep] + expert_out.to(out_flat.dtype))
            routed_out = out_flat.view(bsz, seq, h)

            probe = self._ddp_probe_loss(dtype=x.dtype, device=x.device, alpha=getattr(self, "ddp_probe_alpha", 1e-8))
            if probe is not None:
                balance_loss = (balance_loss if balance_loss is not None else 0.0) + probe

            return routed_out, balance_loss, updated_routing_state

        elif self.mode == "ours_com":
            h_new = self.shared_router(x, h_prev=routing_state)
            logits = self.gate_head(self.h_ln(h_new))
            scores = F.softmax(logits, dim=-1)
            self.last_scores = scores.detach()

            scores_norm = self.score_ln(scores)
            s_feat = self.score_proj(scores_norm)
            h_cond = self.cond_ln(h_new)
            cond = torch.cat([h_cond, s_feat], dim=-1)
            film = self.film_mlp(cond)
            gamma_hat, beta = film.chunk(2, dim=-1)
            gamma = 1.0 + 0.1 * torch.tanh(gamma_hat)
            if self.proj_gamma_beta is not None:
                gamma = self.proj_gamma_beta(gamma)
                beta  = self.proj_gamma_beta(beta)

            B, T, H = x.shape
            x_f = x.view(-1, H)
            x_f = gamma.to(x_f.dtype) * x_f + beta.to(x_f.dtype)
            x_mod = x_f.view(B, T, H)

            local_out = self.experts[0](x_mod)

            top_scores, top_idx = torch.topk(scores, k=1, dim=-1)
            x_flat = x_mod.view(-1, H)
            out_flat = torch.zeros_like(x_flat)
            top_idx_flat = top_idx.view(-1)
            top_scores_flat = top_scores.view(-1, 1)
            for eid in range(len(self.global_experts)):
                idx = (top_idx_flat == eid).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    expert_out = self.global_experts[eid](x_flat[idx]) * top_scores_flat[idx]
                    out_flat[idx] = expert_out.to(out_flat.dtype)
            global_out = out_flat.view_as(x_mod)

            aux_loss = None
            if self.training and getattr(self, "aux_alpha", 0.01) > 0.0:
                one_hot = torch.nn.functional.one_hot(top_idx_flat, num_classes=self.num_experts).float()
                freq = one_hot.mean(0)
                Pi   = scores.mean(0)
                fi   = freq * self.num_experts
                aux_loss = (Pi * fi).sum() * self.aux_alpha

            routed_out = local_out + global_out

            alpha = getattr(self, "ddp_probe_alpha", 1e-8)
            probe = self._ddp_probe_loss(dtype=x.dtype, device=x.device, alpha=alpha)
            if hasattr(self, "global_experts") and self.global_experts is not None:
                d = self.d_model
                p = torch.randn(8, d, device=x.device, dtype=x.dtype) * 1e-3
                g_loss = None
                for e in self.global_experts:
                    y = e(p); l = (y**2).mean()
                    g_loss = l if g_loss is None else (g_loss + l)
                if g_loss is not None:
                    probe = (probe if probe is not None else 0.0) + alpha * g_loss

            if probe is not None:
                aux_loss = (aux_loss if aux_loss is not None else 0.0) + probe

            return routed_out, aux_loss, h_new

        elif self.mode == "ours_refine":
            h_new = self.shared_router(x, h_prev=(routing_state["h"] if isinstance(routing_state, dict) else routing_state))
            h_feat = self.h_ln(h_new)

            N = h_feat.size(0)
            if isinstance(routing_state, dict) and ("logits" in routing_state) and (routing_state["logits"] is not None):
                prev_logits = routing_state["logits"]
                prev_logits = prev_logits.to(h_feat.dtype)
            else:
                prev_logits = h_feat.new_zeros((N, self.score_dim))

            prev_feat = self.prev_score_proj(self.prev_score_ln(prev_logits))

            gate_in = torch.cat([h_feat, prev_feat], dim=-1)
            logits = self.gate_head(gate_in)
            scores = F.softmax(logits, dim=-1)
            self.last_scores = scores.detach()

            B, T, H = x.shape
            local_out = self.experts[0](x)
    
            top_scores, top_idx = torch.topk(scores, k=1, dim=-1)
            x_flat = x.view(-1, H)
            out_flat = torch.zeros_like(x_flat)
            top_idx_flat = top_idx.view(-1)
            top_scores_flat = top_scores.view(-1, 1)
            for eid in range(len(self.global_experts)):
                idx = (top_idx_flat == eid).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    expert_out = self.global_experts[eid](x_flat[idx]) * top_scores_flat[idx]
                    out_flat[idx] = expert_out.to(out_flat.dtype)
            global_out = out_flat.view_as(x)

            aux_loss = None
            if self.training and getattr(self, "aux_alpha", 0.01) > 0.0:
                one_hot = torch.nn.functional.one_hot(top_idx_flat, num_classes=self.num_experts).float()
                freq = one_hot.mean(0)
                Pi   = scores.mean(0)
                fi   = freq * self.num_experts
                aux_loss = (Pi * fi).sum() * self.aux_alpha

            routed_out = local_out + global_out

            alpha = getattr(self, "ddp_probe_alpha", 1e-8)
            probe = self._ddp_probe_loss(dtype=x.dtype, device=x.device, alpha=alpha)
            if hasattr(self, "global_experts") and self.global_experts is not None:
                d = self.d_model
                p = torch.randn(8, d, device=x.device, dtype=x.dtype) * 1e-3
                g_loss = None
                for e in self.global_experts:
                    y = e(p); l = (y**2).mean()
                    g_loss = l if g_loss is None else (g_loss + l)
                if g_loss is not None:
                    probe = (probe if probe is not None else 0.0) + alpha * g_loss
            if probe is not None:
                aux_loss = (aux_loss if aux_loss is not None else 0.0) + probe

            updated_routing_state = {"h": h_new, "logits": logits.detach()}

            return routed_out, aux_loss, updated_routing_state

        elif self.mode == "hash":
            if input_ids is None:
                try:
                    from utils import get_current_input_ids
                    input_ids = get_current_input_ids()
                except Exception:
                    pass
            assert input_ids is not None, "hash mode requires input_ids"
            x_flat = x.view(-1, h)
            token_eids = self.hash_router.route(input_ids).view(-1)
            out_flat = torch.zeros_like(x_flat)
            for eid in range(self.num_experts):
                idx = (token_eids == eid).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    expert_out = self.experts[eid](x_flat[idx])
                    out_flat[idx] = expert_out.to(out_flat.dtype)
            routed_out = out_flat.view(bsz, seq, h)

            probe = self._ddp_probe_loss(dtype=x.dtype, device=x.device, alpha=getattr(self, "ddp_probe_alpha", 1e-8))
            balance_loss = probe

            return routed_out, None, updated_routing_state

        elif self.mode == "xmoe":
            final_mask, scores_sel, aux_loss, top1_idx = self.xmoe_router(x)
            N = bsz * seq
            H = h
            x_flat = x.view(N, H)
            out_flat = x_flat.new_zeros(N, H)
            for eid in range(self.num_experts):
                tok_idx = final_mask[:, eid].nonzero(as_tuple=True)[0]
                if tok_idx.numel() == 0:
                    continue
                xin = x_flat[tok_idx]
                y = experts[eid](xin).to(out_flat.dtype)
                w = scores_sel[tok_idx, eid].unsqueeze(-1)
                out_flat.index_add_(0, tok_idx, y * w)
            routed_out = out_flat.view(bsz, seq, H)
            self.last_scores = self.xmoe_router.last_scores
            balance_loss = aux_loss

            probe = self._ddp_probe_loss(dtype=x.dtype, device=x.device, alpha=getattr(self, "ddp_probe_alpha", 1e-8))
            if probe is not None:
                balance_loss = (balance_loss if balance_loss is not None else 0.0) + probe

            return routed_out, balance_loss, updated_routing_state

        elif self.mode == "stablemoe":
            _stage1_steps = self.stable_stage1_steps if self.stable_stage1_steps is not None else (1 << 62)
            if global_step is not None:
                is_stage2 = (global_step >= _stage1_steps)
            else:
                if self.training:
                    self._num_updates_buf += 1
                is_stage2 = (int(self._num_updates_buf.item()) >= _stage1_steps)
                
            if is_stage2:
                self._maybe_freeze_stage2()

            N = bsz * seq
            x_flat = x.view(N, h)

            if is_stage2:
                if input_ids is None:
                    try:
                        from utils import get_current_input_ids
                        input_ids = get_current_input_ids()
                    except Exception:
                        pass
                if input_ids is None:
                    raise ValueError("stablemoe mode requires input_ids for routing.")
                with torch.no_grad():
                    root = self._stable_root_ref() if hasattr(self, "_stable_root_ref") else None
                    assert root is not None, "StableMoE root ref missing"
                    rfeat = F.embedding(input_ids.view(-1), root.stablemoe_routing_weight)
                    E = root.stablemoe_distill_E
                    affinities = rfeat @ E.t()
                distill_loss = None
            else:
                affinities = x_flat @ self.expert_centroids.t()
                affinities = self._make_finite(affinities)
                if input_ids is None:
                    try:
                        from utils import get_current_input_ids
                        input_ids = get_current_input_ids()
                    except Exception:
                        pass
                if input_ids is None:
                    raise ValueError("stablemoe mode requires input_ids during Stage-1.")

                with torch.no_grad():
                    target = affinities.argmax(dim=1)

                is_primary = (getattr(self, "_layer_idx", 0) == 0)
                if is_primary:
                    root = self._stable_root_ref() if hasattr(self, "_stable_root_ref") else None
                    assert root is not None, "StableMoE root ref missing"
                    rfeat = F.embedding(input_ids.view(-1), root.stablemoe_routing_weight)
                    E = root.stablemoe_distill_E
                    logits_d = rfeat @ E.t()
                    distill_loss = F.cross_entropy(logits_d, target, reduction="mean")
                else:
                    distill_loss = None

            top1_idx = affinities.argmax(dim=1)
            cap = int(math.ceil(N / self.num_experts) * self.capacity_factor)
            if not self.training:
                cap = N
            cap = max(1, min(cap, N))

            out_flat = x_flat.new_zeros(N, h)

            s_full = self._make_finite(x_flat @ self.expert_centroids.t())
            s_full = s_full.to(x_flat.dtype)
            s_top1 = s_full.gather(1, top1_idx.view(-1,1))
            gate_sigmoid = torch.sigmoid(s_top1)

            if (not is_stage2) and self.training and self.stable_balance_alpha > 0:
                n = float(N) / self.num_experts
                balance = x_flat.new_zeros(())
                for eid in range(self.num_experts):
                    idx = (top1_idx == eid).nonzero(as_tuple=True)[0]
                    if idx.numel() == 0:
                        continue
                    sigma_sum = torch.sigmoid(s_full[idx, eid]).mean()
                    balance = balance + ((idx.numel() - n) / n) * sigma_sum
                balance_loss = self.stable_balance_alpha * balance
            else:
                balance_loss = None

            for eid in range(self.num_experts):
                idx = (top1_idx == eid).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                keep = idx[:cap]
                drop = idx[cap:]
                if idx.numel() > 0 and cap > 0:
                    self.last_aux["overflow_rate"] = float(max(idx.numel()-cap, 0)) / float(idx.numel())

                if keep.numel() > 0:
                    y = experts[eid](x_flat[keep]).to(out_flat.dtype)
                    y = y * gate_sigmoid[keep]
                    out_flat[keep] = y
                if drop.numel() > 0:
                    out_flat[drop] = x_flat[drop]

            routed = out_flat.view(bsz, seq, h)
            if distill_loss is not None:
                balance_loss = (balance_loss if balance_loss is not None else 0.0) + distill_loss

            self.last_aux["balance"] = (None if is_stage2 else (balance_loss - (distill_loss or 0.0)))
            self.last_aux["distill"] = distill_loss

            probe = self._ddp_probe_loss(dtype=x.dtype, device=x.device, alpha=getattr(self, "ddp_probe_alpha", 1e-8))
            if probe is not None:
                balance_loss = (balance_loss if balance_loss is not None else 0.0) + probe

            return routed, balance_loss, updated_routing_state

        elif self.mode == "hypermoe":
            cfg = getattr(self, "_hypermoe_cfg", None)
            assert cfg is not None, "hypermoe config가 설정되지 않았습니다."

            if hasattr(self, "adapter_layer") and self.adapter_layer is not None:
                self.adapter_layer.clear_adapter()

            res = x
            x_flat = x.reshape(-1, self.input_size)
            x_flat = x_flat.to(self.w_gate.dtype)
            gates, load, gates_out, expert_mask = noisy_top_k_gating_mixing(self, x_flat, self.training)
            self.last_scores = gates.detach()
            
            importance = gates.sum(0)
            loss_coef = float(cfg.get("loss_coef", 1e-2))
            balance_loss = (cv_squared(importance) + cv_squared(load)) * loss_coef

            dispatcher = SparseDispatcher(self.n_experts, gates)
            expert_inputs = dispatcher.dispatch(x_flat)
            expert_outputs = [self.experts_hypermoe[i](expert_inputs[i]) if expert_inputs[i].numel() > 0
                              else x_flat.new_zeros((0, self.output_size)) for i in range(self.n_experts)]
            y = dispatcher.combine(expert_outputs).reshape(bsz, seq, self.output_size)

            if cfg.get("use_hypernet", True) and hasattr(self, "adapter_layer") and self.adapter_layer is not None:
                index_out = torch.nonzero(gates_out)[:, -1].contiguous().flatten()
                emb = self.n_experts_embedding(index_out)
                emb = emb.view(res.size(0), res.size(1), self.n_experts - self.k, -1).contiguous()
                embedding_input = torch.sum(emb, dim=-2)
                hyp_in = self.embedding_process(embedding_input)
                uw, dw, ub, db = self.param_gen(hyp_in, cfg["layer_idx"])
                self.adapter_layer.apply_adapter_params(res.size(0), res.size(1), uw, dw, ub, db)
                y = self.adapter_layer(res) + y

            probe = self._ddp_probe_loss_hyper(dtype=x.dtype, device=x.device, alpha=getattr(self, "ddp_probe_alpha", 1e-8))
            if probe is not None:
                balance_loss = (balance_loss if balance_loss is not None else 0.0) + probe

            return y, balance_loss, updated_routing_state
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return routed_out, balance_loss, updated_routing_state

class GPT2LayerMoE(nn.Module):
    def __init__(self, config: GPT2Config, mode="switch",
                 num_experts=8, shared_expert=None, global_experts=None,
                 alpha=0.01, capacity_factor=1.25, layer_idx=None, num_layers=None, freq_dict=None,
                 shared_router: RecurrentRouter=None,
                 xmoe_threshold: float = 0.90,
                 xmoe_capacity_factor: float = 1.0,
                 xmoe_expert_mult: float = 0.25,
                 hypermoe_kwargs: dict | None = None):
        super().__init__()
        self.mode = mode
        self.moe = MoELayer(
            d_model=config.n_embd,
            d_ff=config.n_embd * 4,
            num_experts=num_experts,
            mode=mode,
            shared_expert=shared_expert,
            global_experts=global_experts,
            alpha=alpha,
            capacity_factor=capacity_factor,
            freq_dict=freq_dict,
            shared_router=shared_router,
            xmoe_threshold=xmoe_threshold,
            xmoe_capacity_factor=xmoe_capacity_factor,
            xmoe_expert_mult=xmoe_expert_mult,
        )
        self.layer_idx = 0 if layer_idx is None else int(layer_idx)
        setattr(self.moe, "_layer_idx", self.layer_idx)
        if mode == "hypermoe":
            d_model = config.n_embd
            defaults = dict(
                k=1,
                noisy_gating=True,
                use_hypernet=True,
                adapter_dim=max(16, d_model // 32),
                hypernet_input=d_model,
                hypernetwork_bottleneck=max(64, d_model // 8),
                layer_emb_dim=8,
                experts_embedding_dim=32,
                process_dim=max(64, d_model // 8),
                num_hidden_layers=getattr(config, "n_layer", getattr(config, "num_hidden_layers", 12)),
                loss_coef=1e-2,
                layer_idx=(layer_idx if layer_idx is not None else 0),
            )
            cfg = {**defaults, **(hypermoe_kwargs or {})}
            setattr(self.moe, "_hypermoe_cfg", cfg)
            shared_pack = (hypermoe_kwargs or {}).get("_shared_pack", None)
            self.moe._init_hypermoe(cfg, config.n_embd, shared=shared_pack)

        self.last_balance_loss = None
    def forward(self, hidden_states, input_ids=None, routing_state=None, global_step=None, **kwargs):
        out, balance_loss, updated_routing_state = self.moe(
            hidden_states, input_ids=input_ids, routing_state=routing_state, global_step=global_step
        )
        self.last_balance_loss = balance_loss
        return out, balance_loss, updated_routing_state

def balanced_assignment(freq_dict, num_experts, vocab_size):
    buckets = [[] for _ in range(num_experts)]
    bucket_loads = [0] * num_experts
    for token, freq in sorted(freq_dict.items(), key=lambda x: -x[1]):
        eid = min(range(num_experts), key=lambda i: bucket_loads[i])
        buckets[eid].append(token)
        bucket_loads[eid] += freq
    table = [0] * vocab_size
    for eid, tokens in enumerate(buckets):
        for t in tokens:
            table[t] = eid
    return table

class HashRouter:
    def __init__(self, vocab_size, num_experts, method="balanced", seed=0,
                 freq_dict=None, device=None, saved_table_path=None):
        self.num_experts = num_experts
        random.seed(seed)
        self.method = method
        table_loaded_from_file = False
        if freq_dict and isinstance(freq_dict, dict) and '__load_from_file__' in freq_dict:
            table_path = freq_dict['__load_from_file__']
            import os, torch
            if os.path.exists(table_path):
                print(f"🔹 Loading HashRouter table directly from {table_path}")
                obj = torch.load(table_path, map_location='cpu')
                self.table_tensor = obj.to(torch.long)
                self.table = self.table_tensor.tolist()
                table_loaded_from_file = True
            else:
                raise FileNotFoundError(f"Hash router table specified in freq_dict not found at: {table_path}")
        if not table_loaded_from_file:
            import torch
            if saved_table_path and os.path.exists(saved_table_path):
                print(f"🔹 Loading HashRouter table from {saved_table_path}")
                obj = torch.load(saved_table_path, map_location='cpu')
                self.table_tensor = obj.to(torch.long)
                self.table = self.table_tensor.tolist()
            elif method == "random":
                self.table = [random.randint(0, num_experts-1) for _ in range(vocab_size)]
            elif method == "balanced":
                assert freq_dict is not None, "freq_dict must be provided for 'balanced' method"
                self.table = balanced_assignment(freq_dict, num_experts, vocab_size)
            else:
                raise ValueError(f"Unknown method: {method}")
            if not (saved_table_path and os.path.exists(saved_table_path)):
                self.table_tensor = torch.tensor(self.table, dtype=torch.long)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.table_tensor = self.table_tensor.to(device, dtype=torch.long)
    def route(self, token_ids):
        return self.table_tensor[token_ids]

def convert_gpt2_to_moe(
    model,
    config,
    mode: str = "switch",
    num_experts: int = 8,
    alpha: float = 0.01,
    capacity_factor: float = 1.25,
    freq_dict=None,
    xmoe_threshold: float = 0.90,
    xmoe_capacity_factor: float | None = None,
    xmoe_expert_mult: float = 0.25,
    stable_routing_dim: int = 50,
    stable_balance_alpha: float = 0.3,
):
    if mode == "dense":
        if _rank0():
            print("🔹 Mode is 'dense', returning original GPT-2 model without MoE conversion.")
        return model

    shared_expert = None
    global_experts = None
    shared_router = None
    layer_experts = num_experts

    if mode == "stablemoe":
        vocab_size = getattr(config, "vocab_size", 50257)
        eff_num_experts = num_experts
        
        if not hasattr(model, "stablemoe_routing_weight"):
            model.register_parameter(
                "stablemoe_routing_weight",
                nn.Parameter(torch.empty(vocab_size, stable_routing_dim))
            )
            nn.init.normal_(model.stablemoe_routing_weight, mean=0.0, std=0.02)
            if _rank0():
                print(f"🔹 StableMoE: Registered shared routing weight ({vocab_size}, {stable_routing_dim})")

        if not hasattr(model, "stablemoe_distill_E"):
            model.register_parameter(
                "stablemoe_distill_E",
                nn.Parameter(torch.empty(eff_num_experts, stable_routing_dim))
            )
            nn.init.orthogonal_(model.stablemoe_distill_E, gain=0.1)
            if _rank0():
                print(f"🔹 StableMoE: Registered shared distilled centroids ({eff_num_experts}, {stable_routing_dim})")

    if mode == "ours_com":
        assert num_experts >= 2, "ours_com requires at least 2 experts (1 local + N global)."
        global_experts = nn.ModuleList([
            Expert(config.n_embd, config.n_embd * 4, initializer_range=config.initializer_range)
            for _ in range(num_experts - 1)
        ])
        shared_router = RecurrentRouter(d_model=config.n_embd, hidden_dim=config.n_embd)
        layer_experts = 1

    if mode == "ours_refine":
        assert num_experts >= 2, "ours_refine requires at least 2 experts (1 local + N global)."
        global_experts = nn.ModuleList([
            Expert(config.n_embd, config.n_embd * 4, initializer_range=config.initializer_range)
            for _ in range(num_experts - 1)
        ])
        shared_router = RecurrentRouter(d_model=config.n_embd, hidden_dim=config.n_embd)
        layer_experts = 1

    if mode == "xmoe":
        if xmoe_capacity_factor is None:
            scale = 1.0 / max(1e-6, float(xmoe_expert_mult))
            xmoe_capacity_factor = float(capacity_factor) * scale
        xmoe_capacity_factor = float(max(0.5, min(xmoe_capacity_factor, 8.0)))
        if _rank0():
            print(f"🧮 XMoE γ auto-scale: base_cf={capacity_factor:.2f}, mult={xmoe_expert_mult:.2f} ⇒ γ={xmoe_capacity_factor:.2f}")

    hypermoe_shared_pack = None
    if mode == "hypermoe":
        d_model = config.n_embd
        defaults = dict(
            k=1,
            noisy_gating=True,
            use_hypernet=True,
            adapter_dim=max(16, d_model // 32),
            hypernet_input=d_model,
            hypernetwork_bottleneck=max(64, d_model // 8),
            layer_emb_dim=8,
            experts_embedding_dim=32,
            process_dim=max(64, d_model // 8),
            num_hidden_layers=getattr(config, "n_layer", getattr(config, "num_hidden_layers", 12)),
            loss_coef=1e-2,
            layer_idx=0,
        )
        hypermoe_shared_pack = HyperMoEShared(defaults, d_model, num_experts)
        model.hypermoe_shared = hypermoe_shared_pack

    for i, block in enumerate(model.transformer.h):
        if mode == "ours_com":
            block.mlp = GPT2LayerMoE(
                config, mode, layer_experts,
                shared_expert=None,
                global_experts=global_experts,
                alpha=alpha,
                capacity_factor=capacity_factor,
                freq_dict=None,
                shared_router=shared_router,
                layer_idx=i,
            )
        elif mode == "ours_refine":
            block.mlp = GPT2LayerMoE(
                config, mode, layer_experts,
                shared_expert=None,
                global_experts=global_experts,
                alpha=alpha,
                capacity_factor=capacity_factor,
                freq_dict=None,
                shared_router=shared_router,
                layer_idx=i,
            )
        elif mode == "hypermoe":
            hypermoe_defaults = dict(
                k=1,
                noisy_gating=True,
                use_hypernet=True,
                adapter_dim=max(16, config.n_embd // 32),
                hypernet_input=config.n_embd,
                hypernetwork_bottleneck=max(64, config.n_embd // 8),
                layer_emb_dim=8,
                experts_embedding_dim=32,
                process_dim=max(64, config.n_embd // 8),
                num_hidden_layers=getattr(config, "n_layer", getattr(config, "num_hidden_layers", 12)),
                loss_coef=1e-2,
                layer_idx=i,
                _shared_pack=hypermoe_shared_pack,
            )
            layer = GPT2LayerMoE(
                config, mode, num_experts,
                shared_expert=None,
                global_experts=None,
                alpha=alpha,
                capacity_factor=capacity_factor,
                freq_dict=None,
                hypermoe_kwargs=hypermoe_defaults,
                layer_idx=i,
            )
            block.mlp = layer
        else:
            layer = GPT2LayerMoE(
                config, mode, layer_experts,
                shared_expert,
                global_experts,
                alpha,
                capacity_factor,
                freq_dict=freq_dict if mode == "hash" else None,
                xmoe_threshold=xmoe_threshold,
                xmoe_capacity_factor=xmoe_capacity_factor,
                xmoe_expert_mult=xmoe_expert_mult,
                layer_idx=i,
            )
            if mode == "stablemoe":
                layer.moe.vocab_size = getattr(config, "vocab_size", 50257)
                layer.moe.stable_routing_dim = stable_routing_dim
                layer.moe.stable_balance_alpha = stable_balance_alpha
                import weakref
                object.__setattr__(layer.moe, "_stable_root_ref", weakref.ref(model))
            block.mlp = layer
    return model
