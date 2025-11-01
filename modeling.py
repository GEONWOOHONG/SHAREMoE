# modeling.py — Expert/Router/MoE/GPT2LayerMoE/HashRouter/convert 함수
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config
from typing import Optional
from collections import Counter

# ===== Expert =====
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

# ===== Schedulers/routers =====
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

# ===== MoE Layer =====
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
        elif mode == "hash":
            self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
            self.hash_router = HashRouter(
                vocab_size=50257,
                num_experts=num_experts,
                method="balanced",
                freq_dict=freq_dict
            )
        elif mode == "expert_choice":
            self.experts = nn.ModuleList([Expert(d_model, d_ff, use_gelu=True) for _ in range(num_experts)])
            self.router = nn.Linear(d_model, num_experts, bias=False)
            self.capacity_factor = capacity_factor
            nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        elif mode in {"switch", "gshard"}:
            self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
            self.router = Router(d_model, num_experts, top_k=(1 if mode=="switch" else 2), alpha=alpha)
        
        elif mode == "stablemoe":
            # === StableMoE (single-device, no all_to_all) ===
            self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
            self.capacity_factor = capacity_factor
            # 라우팅용 센트로이드 (Stage-1: d_model, Stage-2: routing_dim)
            self.expert_centroids = nn.Parameter(torch.empty(num_experts, d_model))
            nn.init.orthogonal_(self.expert_centroids, gain=0.1)

            # 경량 라우터 임베딩 (word-embedding 대용; default 50d)
            self.stable_routing_dim = getattr(self, "stable_routing_dim", 50)
            self.vocab_size = getattr(self, "vocab_size", 50257)
            self.routing_emb = nn.Embedding(self.vocab_size, self.stable_routing_dim)

            # distilled centroids (Stage-1의 타깃 & Stage-2 라우팅에 사용)
            self.distill_expert_centroids = nn.Parameter(
                torch.empty(num_experts, self.stable_routing_dim)
            )
            nn.init.orthogonal_(self.distill_expert_centroids, gain=0.1)

            # 스케줄/로스 계수 (train.py에서 주입; 여기선 기본값 미설정)
            self.stable_stage1_steps = None
            self.stable_balance_alpha = getattr(self, "stable_balance_alpha", 0.3)

            # 내부 업데이트 카운터(옵션)
            self.register_buffer("_num_updates_buf", torch.zeros((), dtype=torch.long))
            self._stage2_frozen = False

        elif global_experts is None:
            self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    @staticmethod
    def _make_finite(x: torch.Tensor) -> torch.Tensor:
        ok = torch.isfinite(x)
        if not ok.all():
            minv = x[ok].min() if ok.any() else torch.tensor(0., device=x.device, dtype=x.dtype)
            x = x.clone()
            x[~ok] = minv
        return x

    def _maybe_freeze_stage2(self):
        if self.mode != "stablemoe" or self._stage2_frozen:
            return
        for p in self.routing_emb.parameters():
            p.requires_grad_(False)
        self.distill_expert_centroids.requires_grad_(False)
        self._stage2_frozen = True
        if self.training:
            print("🔒 StableMoE: Froze routing_emb & distill_E for Stage-2")

    def forward(self, x, input_ids=None, routing_state=None, global_step: Optional[int]=None):
        bsz, seq, h = x.shape
        routed_out = torch.zeros_like(x)
        balance_loss = None
        updated_routing_state = routing_state
        experts = self.global_experts if self.global_experts is not None else self.experts

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
            return routed_out, balance_loss, updated_routing_state

        elif self.mode == "gshard":
            top_scores, top_idx, balance_loss = self.router(x)
            x_flat = x.view(-1, h)
            out_flat = torch.zeros_like(x_flat)
            capacity = int(self.capacity_factor * math.ceil(x_flat.size(0) * 2 / self.num_experts))
            for eid in range(self.num_experts):
                for k in range(2):
                    idx = (top_idx[:, k] == eid).nonzero(as_tuple=True)[0]
                    if idx.numel() > 0:
                        score = top_scores[idx, k]
                        if k == 1:
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
            return routed_out, aux_loss, h_new

        elif self.mode == "hash":
            if input_ids is None:
                try:
                    from utils import get_current_input_ids  # 수정: utils에서 import
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
            return routed_out, None, updated_routing_state

        elif self.mode == "expert_choice":
            num_tokens = bsz * seq
            x_flat = x.view(-1, h)
            router_logits = self.router(x_flat)
            balance_loss = None
            top_k = int(self.capacity_factor * num_tokens / self.num_experts)
            top_k = max(1, min(top_k, num_tokens))
            affinity_scores = F.softmax(router_logits, dim=-1)
            self.last_scores = affinity_scores.detach()
            weights, selected_tokens = torch.topk(affinity_scores.T, top_k, dim=-1)
            out_flat = x_flat.new_zeros(x_flat.size())
            for eid in range(self.num_experts):
                token_indices = selected_tokens[eid]
                expert_input = x_flat[token_indices]
                if expert_input.numel() > 0:
                    expert_output = self.experts[eid](expert_input).to(out_flat.dtype)
                    score = weights[eid].unsqueeze(-1)
                    out_flat.index_add_(0, token_indices, expert_output * score)
            routed_out = out_flat.view(bsz, seq, h)
            return routed_out, balance_loss, updated_routing_state

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
            return routed_out, balance_loss, updated_routing_state

        elif self.mode == "stablemoe":
            """
            Stage-1: 대용량 표현(h) 기반 라우팅, distill CE + balance loss 사용
            Stage-2: 경량 라우터(embedding) 기반 라우팅 고정, 보조 손실 없음
            단일 GPU 버전: expert별 토큰 슬라이스 → 처리 → 원위치 scatter
            """
            # --- stage 판정 ---
            # 안전 가드: 아직 train.py가 값을 주입하지 않으면 매우 큰 값으로 간주(즉, stage-1 유지)
            _stage1_steps = self.stable_stage1_steps if self.stable_stage1_steps is not None else (1 << 62)
            if global_step is not None:
                is_stage2 = (global_step >= _stage1_steps)
            else:
                # 학습 중이면 내부 카운터 업데이트(옵션)
                if self.training:
                    self._num_updates_buf += 1
                is_stage2 = (int(self._num_updates_buf.item()) >= _stage1_steps)
                
            if is_stage2:
                self._maybe_freeze_stage2()

            N = bsz * seq
            x_flat = x.view(N, h)

            # --- affinity 계산 ---
            if is_stage2:
                # 경량 라우터 경로(동결 가정)로 라우팅
                if input_ids is None:
                    try:
                        from utils import get_current_input_ids
                        input_ids = get_current_input_ids()
                    except Exception:
                        pass
                if input_ids is None:
                    raise ValueError("stablemoe mode requires input_ids for routing.")
                with torch.no_grad():  # stage2 라우팅 고정
                    rfeat = self.routing_emb(input_ids.view(-1))          # [N, rdim]
                affinities = rfeat @ self.distill_expert_centroids.t()    # [N, E]
                distill_loss = None
            else:
                # Stage-1: full feat로 라우팅 + distill target/CE
                affinities = x_flat @ self.expert_centroids.t()           # [N, E]
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
                    target = affinities.argmax(dim=1)                    # [N]
                rfeat = self.routing_emb(input_ids.view(-1))              # [N, rdim]
                logits_d = rfeat @ self.distill_expert_centroids.t()      # [N, E]
                distill_loss = F.cross_entropy(logits_d, target, reduction="mean")

            # --- greedy top-1 assignment + capacity 컷 ---
            top1_idx = affinities.argmax(dim=1)                           # [N]
            # capacity per expert
            cap = int(math.ceil(N / self.num_experts) * self.capacity_factor)
            cap = max(1, min(cap, N))

            out_flat = x_flat.new_zeros(N, h)

            # 공통 full-feature affinity와 sigmoid 게이트(두 스테이지 동일 정의)
            s_full = self._make_finite(x_flat @ self.expert_centroids.t())        # [N, E]
            s_full = s_full.to(x_flat.dtype)
            s_top1 = s_full.gather(1, top1_idx.view(-1,1))                        # [N,1]
            gate_sigmoid = torch.sigmoid(s_top1)                                  # [N,1]

            # balance loss (Stage-1에서만)
            if (not is_stage2) and self.training and self.stable_balance_alpha > 0:
                # 논문식: 각 expert에 실제로 라우팅된 토큰들의 σ(s_full) 합을 이용
                n = float(N) / self.num_experts
                balance = x_flat.new_zeros(())
                for eid in range(self.num_experts):
                    idx = (top1_idx == eid).nonzero(as_tuple=True)[0]
                    if idx.numel() == 0:
                        continue
                    sigma_sum = torch.sigmoid(s_full[idx, eid]).sum()
                    balance = balance + ((idx.numel() - n) / n) * sigma_sum
                balance_loss = self.stable_balance_alpha * balance
            else:
                balance_loss = None

            # expert별 토큰 처리 (용량 초과면 나머지는 패스-스루)
            for eid in range(self.num_experts):
                idx = (top1_idx == eid).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                keep = idx[:cap]
                drop = idx[cap:]
                # overflow rate 로깅용
                if idx.numel() > 0 and cap > 0:
                    self.last_aux["overflow_rate"] = float(max(idx.numel()-cap, 0)) / float(idx.numel())

                if keep.numel() > 0:
                    y = experts[eid](x_flat[keep]).to(out_flat.dtype)
                    y = y * gate_sigmoid[keep]
                    out_flat[keep] = y
                if drop.numel() > 0:
                    # 용량 초과분은 residual 통과
                    out_flat[drop] = x_flat[drop]

            routed = out_flat.view(bsz, seq, h)
            # distill_loss를 balance_loss에 합산해 기존 인터페이스 유지
            if distill_loss is not None:
                balance_loss = (balance_loss if balance_loss is not None else 0.0) + distill_loss
            # 분리 로깅
            self.last_aux["balance"] = (None if is_stage2 else (balance_loss - (distill_loss or 0.0)))
            self.last_aux["distill"] = distill_loss

            return routed, balance_loss, updated_routing_state
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return routed_out, balance_loss, updated_routing_state

# ===== GPT2LayerMoE wrapper =====
class GPT2LayerMoE(nn.Module):
    def __init__(self, config: GPT2Config, mode="switch",
                 num_experts=8, shared_expert=None, global_experts=None,
                 alpha=0.01, capacity_factor=1.25, layer_idx=None, num_layers=None, freq_dict=None,
                 shared_router: RecurrentRouter=None,
                 xmoe_threshold: float = 0.90,
                 xmoe_capacity_factor: float = 1.0,
                 xmoe_expert_mult: float = 0.25):
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
        self.last_balance_loss = None
    def forward(self, hidden_states, input_ids=None, routing_state=None, global_step=None, **kwargs):
        out, balance_loss, updated_routing_state = self.moe(
            hidden_states, input_ids=input_ids, routing_state=routing_state, global_step=global_step
        )
        self.last_balance_loss = balance_loss
        return out, balance_loss, updated_routing_state

# ===== Hash helpers (shared with tools_hash) =====
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

# ===== convert utility =====
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
        print("🔹 Mode is 'dense', returning original GPT-2 model without MoE conversion.")
        return model

    shared_expert = None
    global_experts = None
    shared_router = None
    layer_experts = num_experts

    # --- StableMoE 전용: 모든 레이어가 공유할 라우터 weight와 distilled centroids ---
    shared_routing_weight = None
    shared_distill_E = None

    if mode == "ours_com":
        assert num_experts >= 2, "ours_com requires at least 2 experts (1 local + N global)."
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
        print(f"🧮 XMoE γ auto-scale: base_cf={capacity_factor:.2f}, mult={xmoe_expert_mult:.2f} ⇒ γ={xmoe_capacity_factor:.2f}")

    for block in model.transformer.h:
        if mode == "ours_com":
            block.mlp = GPT2LayerMoE(
                config, mode, layer_experts,
                shared_expert=None,
                global_experts=global_experts,
                alpha=alpha,
                capacity_factor=capacity_factor,
                freq_dict=None,
                shared_router=shared_router,
            )
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
            )
            if mode == "stablemoe":
                layer.moe.vocab_size = getattr(config, "vocab_size", 50257)
                layer.moe.stable_routing_dim = stable_routing_dim
                layer.moe.stable_balance_alpha = stable_balance_alpha
                
                # 1) shared weight / shared distill_E 최초 1회 생성
                if shared_routing_weight is None:
                    shared_routing_weight = nn.Parameter(
                        torch.empty(layer.moe.vocab_size, layer.moe.stable_routing_dim)
                    )
                    nn.init.normal_(shared_routing_weight, mean=0.0, std=0.02)  # 임의 초기화
                    print(f"🔹 StableMoE: Created shared routing weight ({layer.moe.vocab_size}, {layer.moe.stable_routing_dim})")
                if shared_distill_E is None:
                    shared_distill_E = nn.Parameter(
                        torch.empty(layer.moe.num_experts, layer.moe.stable_routing_dim)
                    )
                    nn.init.orthogonal_(shared_distill_E, gain=0.1)
                    print(f"🔹 StableMoE: Created shared distilled centroids ({layer.moe.num_experts}, {layer.moe.stable_routing_dim})")

                # 2) 각 레이어는 "자기 임베딩 모듈"을 갖되, weight는 모두 같은 파라미터를 바라보게 tying
                layer.moe.routing_emb = nn.Embedding(layer.moe.vocab_size, layer.moe.stable_routing_dim)
                layer.moe.routing_emb.weight = shared_routing_weight  # <-- 핵심: 파라미터 공유

                # 3) distill_E도 동일 파라미터 객체를 공유
                layer.moe.distill_expert_centroids = shared_distill_E
            block.mlp = layer
    return model
