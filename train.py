# train.py — 학습/평가/통계 (원본 로직 그대로)
import os, math, torch, tiktoken, shutil, contextlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import load_model
import torch.nn.functional as F
from config import RUNS_DIR, HASH_TABLE_PATH
from data import load_or_prepare_pile, worker_init_fn, get_dataloader_generator
from modeling import convert_gpt2_to_moe, GPT2LayerMoE
from patches import (
    patch_model_basic,
    patch_model_for_hash_moe,
    patch_model_for_ours_com,
    patch_model_for_stablemoe,
)
from utils import (set_seed, get_default_optimizer, get_default_scheduler,
                   print_model_info, save_checkpoint, ensure_flash_attn)

from torch.utils.tensorboard import SummaryWriter

ensure_flash_attn()  # 원본과 동일하게 모듈 레벨에서 즉시 호출

enc = tiktoken.get_encoding("gpt2")

def chunked_cross_entropy(logits, labels, ignore_index=-100, chunk_tokens=8192):
    """
    GPT-2 계열은 HF가 내부에서 CE 계산 시 shift를 합니다.
    여기서도 동일하게 직접 shift 후, BT 축을 chunk로 나눠 CE를 계산해 피크 메모리 절감.
    logits: [B, T, V], labels: [B, T]
    """
    # 1) GPT2-style shift (next-token prediction)
    shift_logits = logits[:, :-1, :].contiguous()    # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()        # [B, T-1]

    B, Tm1, V = shift_logits.shape
    bt = B * Tm1

    # 2) BT x V 로 펴서 chunk-by-chunk로 CE 계산
    flat_logits = shift_logits.view(bt, V)
    flat_labels = shift_labels.view(bt)

    total = 0.0
    valid = 0
    for start in range(0, bt, chunk_tokens):
        end = min(start + chunk_tokens, bt)
        loss = F.cross_entropy(
            flat_logits[start:end],
            flat_labels[start:end],
            ignore_index=ignore_index,
            reduction="sum",
        )
        total += loss
        # 유효 토큰 수 (ignore_index 제외)
        if ignore_index is not None:
            valid += (flat_labels[start:end] != ignore_index).sum().item()
        else:
            valid += (end - start)

    denom = max(valid, 1)
    return total / denom

def init_distributed():
    """torchrun 환경이면 DDP 초기화하고 (is_dist, rank, world_size, local_rank) 반환"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0

@torch.no_grad()
def evaluate(model, dataloader, device):
    is_main = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
    model.eval()
    losses = []
    it = tqdm(dataloader, desc="Validating", leave=False) if is_main else dataloader
    for batch in it:
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        labels[labels == enc.eot_token] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=batch["attention_mask"].to(device),
                labels=None,
                use_cache=False,
                global_step=(1 << 62),
            )
            logits = outputs.logits
            loss = chunked_cross_entropy(logits, labels, ignore_index=-100, chunk_tokens=8192)
            losses.append(loss.detach())

    mean_loss = torch.stack(losses).mean()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        mean_loss = mean_loss / dist.get_world_size()
    model.train()
    ppl = math.exp(mean_loss.item())
    return mean_loss.item(), ppl

def compute_moe_stats(model, config, mode):
    moe_layers = [m.moe for m in model.modules() if isinstance(m, GPT2LayerMoE)]
    balance_scores = []
    if len(moe_layers) == 0:
        return {"balance": 0.0}
    for moe in moe_layers:
        if moe.mode == "hash":
            continue
        if moe.mode == "ours_com" and getattr(moe, "last_scores", None) is not None:
            probs = moe.last_scores.mean(0)
            entropy = -(probs * (probs + 1e-9).log()).sum().item()
            balance_scores.append(entropy)
            continue
        if moe.mode == "xmoe" and getattr(moe, "last_scores", None) is not None:
            probs = moe.last_scores.mean(0)
            entropy = -(probs * (probs + 1e-9).log()).sum().item()
            balance_scores.append(entropy)
            continue
        if moe.mode == "hypermoe" and getattr(moe, "last_scores", None) is not None:
            probs = moe.last_scores.mean(0)
            entropy = -(probs * (probs + 1e-9).log()).sum().item()
            balance_scores.append(entropy)
            continue
        if moe.mode == "expert_choice" and moe.last_scores is not None:
            probs = moe.last_scores.mean(0)
            entropy = -(probs * (probs + 1e-9).log()).sum().item()
            balance_scores.append(entropy)
        elif hasattr(moe, 'router') and getattr(moe.router, 'last_scores', None) is not None:
            probs = moe.router.last_scores.mean(0)
            entropy = -(probs * (probs + 1e-9).log()).sum().item()
            balance_scores.append(entropy)
    return {"balance": sum(balance_scores) / len(balance_scores) if balance_scores else 0.0}

def train_moe(mode="switch", num_experts=8, batch_size=32, seq_len=1024, grad_accum=1, continue_training=False):
    is_dist, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    def is_main(): return (not is_dist) or (rank == 0)
    
    save_dir = f"/workspace/checkpoints/{mode}_exp1"
    os.makedirs(save_dir, exist_ok=True)
    if is_main():
        print(f"💾 Checkpoint directory: {save_dir}")

    eff_num_experts = num_experts * 4 if mode == "xmoe" else num_experts
    if mode == "xmoe" and eff_num_experts != num_experts and is_main():
        print(f"🧮 xmoe mode: num_experts overridden {num_experts} → {eff_num_experts} (×4)")

    freq_dict = None
    if mode == "hash":
        if not os.path.exists(HASH_TABLE_PATH):
            raise FileNotFoundError(f"Hash table not found at {HASH_TABLE_PATH}\nPlease run `create_global_hash_table()` first.")
        if is_main():
            print(f"🔹 Loading global hash table from: {HASH_TABLE_PATH}")
        freq_dict = {'__load_from_file__': HASH_TABLE_PATH}

    if is_dist:
        if rank == 0:
            _ = load_or_prepare_pile(verbose=False)
        dist.barrier()

    trainer_state = None
    if continue_training:
        print("🔄 Loading model from last checkpoint...")
        config = GPT2Config.from_pretrained(save_dir)
        model = GPT2LMHeadModel(config)
        model = convert_gpt2_to_moe(
            model, config, mode=mode, num_experts=eff_num_experts, alpha=0.01, freq_dict=freq_dict
        )
        last_ckpt = os.path.join(save_dir, "last_checkpoint.safetensors")
        trainer_path = os.path.join(save_dir, "last_checkpoint.safetensors_trainer.pt")
        if not (os.path.exists(last_ckpt) and os.path.exists(trainer_path)):
            raise FileNotFoundError("Checkpoint files not found for continuation.")
        load_model(model, last_ckpt)
        print(f"🔹 Loaded model weights from: {last_ckpt}")
        trainer_state = torch.load(trainer_path, map_location="cpu")
        print(f"🔹 Restoring optimizer/scheduler from {trainer_path}")
    else:
        config = GPT2Config(vocab_size=50257, n_positions=1024, n_ctx=1024, n_embd=1024, n_layer=8, n_head=8)
        model = GPT2LMHeadModel(config)
        stable_args = dict(stable_routing_dim=50, stable_balance_alpha=0.3)
        
        model = convert_gpt2_to_moe(
            model, config,
            mode=mode,
            num_experts=eff_num_experts,
            alpha=0.01,
            freq_dict=freq_dict,
            **(stable_args if mode == "stablemoe" else {})
        )

    if mode == "stablemoe":
        patch_model_for_stablemoe(model)
    elif mode == "hash":
        patch_model_for_hash_moe(model)
    elif mode == "ours_com":
        patch_model_for_ours_com(model)
    else:
        patch_model_basic(model)

    train_dataset, valid_dataset = load_or_prepare_pile(verbose=is_main())
    if is_main():
        print(f"Using FULL datasets: train={len(train_dataset):,}, valid={len(valid_dataset):,}")

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    NUM_WORKERS = max(1, os.cpu_count() // 2)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if is_dist else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False) if is_dist else None
    
    # 워커 시드 고정을 위한 Generator 생성
    train_generator = get_dataloader_generator(rank if is_dist else 0)
    valid_generator = get_dataloader_generator(rank if is_dist else 0)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True,
        worker_init_fn=worker_init_fn, generator=train_generator
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        sampler=valid_sampler, num_workers=NUM_WORKERS,
        pin_memory=True, prefetch_factor=2, persistent_workers=True,
        worker_init_fn=worker_init_fn, generator=valid_generator
    )

    num_batches = len(train_loader)
    optim_steps_per_epoch = math.ceil(num_batches / grad_accum)

    from utils import print_model_info as _pmi
    if is_main():
        _pmi(model, config, mode, eff_num_experts, batch_size=batch_size, grad_accum_steps=grad_accum,
             effective_batch=batch_size * (world_size if is_dist else 1) * grad_accum)

    if is_main() and not os.path.exists(os.path.join(save_dir, "config.json")):
        config.loss_type = "ForCausalLMLoss"
        config.save_pretrained(save_dir)

    model.to(device)
    base_model = model
    if is_dist:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=False,
            broadcast_buffers=False,
            find_unused_parameters=True,
            static_graph=False,
        )
    optimizer = get_default_optimizer(model)

    runs_root = RUNS_DIR
    writer = None
    if is_main():
        if os.path.exists(runs_root):
            shutil.rmtree(runs_root)
        writer = SummaryWriter(log_dir=os.path.join(runs_root, "exp1"))

    if continue_training and trainer_state is not None:
        optimizer.load_state_dict(trainer_state["optimizer"])
        # 총 옵티 스텝 수를 복원(없으면 현재 에폭의 추정치 사용)
        total_steps = trainer_state.get("total_train_steps", optim_steps_per_epoch)
        scheduler = get_default_scheduler(optimizer, total_steps, warmup_ratio=0.1)
        if "scheduler" in trainer_state:
            scheduler.load_state_dict(trainer_state["scheduler"])
        best_loss   = trainer_state["best_loss"]
        start_step  = trainer_state["step"]  # 주의: 이건 'optim_step'으로 해석할 예정
        try:
            print("🔎 Resume LR:", scheduler.get_last_lr()[0])
        except Exception:
            print("🔎 Resume LR (param_group[0]):", optimizer.param_groups[0]["lr"])
        print(f"🔹 Resumed training from step {start_step}, best_loss={best_loss:.4f}")
    else:
        # 한 에폭만 학습하므로 해당 에폭 내 옵티 스텝 수
        total_steps = optim_steps_per_epoch
        scheduler   = get_default_scheduler(optimizer, total_steps, warmup_ratio=0.1)
        best_loss   = float("inf")
        start_step  = 0

    if mode == "stablemoe":
        stage1_ratio = 0.10
        stage1_steps = max(1, int(round(stage1_ratio * total_steps)))

        for m in model.modules():
            if isinstance(m, GPT2LayerMoE) and m.mode == "stablemoe":
                m.moe.stable_stage1_steps = int(stage1_steps)

        if is_main():
            print(f"🧭 StableMoE schedule: Stage-1 = {stage1_steps} steps "
                  f"({stage1_ratio*100:.1f}% of {total_steps}), Stage-2 thereafter.")
        
        # Stage-2 진입 시 명시적 고정 보장 (안전 가드)
        print("🔒 StableMoE: Pre-freezing routing components for Stage-2 safety...")
        for m in model.modules():
            if isinstance(m, GPT2LayerMoE) and m.mode == "stablemoe":
                # Stage-1 종료 직후 즉시 freeze 준비 (forward에서도 안전가드 동작)
                # 논문: Stage-2에서 D(·)와 Ě를 동결
                pass  # forward에서 _maybe_freeze_stage2()가 처리하지만, 명시적 준비
        
        if writer:
            writer.add_scalar("stablemoe/stage1_steps", stage1_steps, 0)
            writer.add_text("stablemoe/config",
                            f"stage1_ratio={stage1_ratio}, total_steps={total_steps}", 0)
    
    tokens_per_batch = batch_size * seq_len
    total_tokens = len(train_loader) * tokens_per_batch
    if is_main():
        print(f"🔹 Training for 1 epoch = {len(train_loader):,} steps ≈ {total_tokens/1e9:.2f}B tokens")

    progress_bar = tqdm(total=len(train_loader), desc=f"Training (rank {rank})", leave=True) if is_main() else None
    total_train_steps = total_steps
    point_one_step = max(1, total_train_steps // 1000)
    five_percent_interval = max(1, total_train_steps // 20)

    # 🔹 중단지점 재개 시 중복 학습 방지
    remaining_optim_steps = total_steps - start_step
    if remaining_optim_steps <= 0 and is_main():
        print("✅ Already completed planned steps; skipping training.")
        return

    num_epochs = 1
    optim_step = start_step
    reached_budget = False  # 2중 루프 탈출용 플래그
    
    for epoch in range(num_epochs):
        if is_dist and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            labels[labels == enc.eot_token] = -100

            # ⬇️ no_sync 문맥
            sync_now = (step % grad_accum) == 0
            ctx = (model.no_sync() if (is_dist and not sync_now) else contextlib.nullcontext())
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=batch["attention_mask"].to(device),
                        labels=None,
                        use_cache=False,
                        global_step=step,
                    )
                    logits = outputs.logits
                    main_loss = chunked_cross_entropy(logits, labels, ignore_index=-100, chunk_tokens=8192)

                balance_losses = [m.last_balance_loss for m in model.modules()
                                if isinstance(m, GPT2LayerMoE) and m.last_balance_loss is not None]
                aux_loss = torch.stack(balance_losses).mean() if balance_losses else 0.0
                loss = main_loss + aux_loss

                # StableMoE 분해 로깅 (생략 가능)
                if mode == "stablemoe":
                    balances, distills, overflows = [], [], []
                    for m in model.modules():
                        if isinstance(m, GPT2LayerMoE) and hasattr(m, "moe"):
                            aux = m.moe.last_aux
                            if aux:
                                if aux["balance"] is not None: balances.append(float(aux["balance"]))
                                if aux["distill"]  is not None: distills.append(float(aux["distill"]))
                                if aux["overflow_rate"] is not None: overflows.append(float(aux["overflow_rate"]))
                    if balances and writer: writer.add_scalar("stablemoe/balance_loss", sum(balances)/len(balances), step)
                    if distills and writer: writer.add_scalar("stablemoe/distill_loss",  sum(distills)/len(distills), step)
                    if overflows and writer: writer.add_scalar("stablemoe/overflow_rate", sum(overflows)/len(overflows), step)

                (loss / grad_accum).backward()

            if sync_now:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optim_step += 1  # ✅ 옵티 스텝 증가
                
                # 🔹 총 스텝 도달 시 탈출
                if optim_step >= total_steps:
                    reached_budget = True
                    break

                # 🔹 LR 로깅은 옵티 스텝 기준
                if writer and (optim_step % 50 == 0):
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], optim_step)

                # 🔹 검증도 옵티 스텝 기준
                if (optim_step % five_percent_interval == 0) or (optim_step == point_one_step):
                    msg = ("🔍 Full validation (every 5%) at optim step "
                        f"{optim_step}..." if (optim_step % five_percent_interval == 0)
                        else f"🔍 Quick validation (0.1%) at optim step {optim_step}...")
                    if progress_bar:
                        progress_bar.write(msg)

                    valid_loss, valid_ppl = evaluate(model, valid_loader, device)
                    stats = compute_moe_stats(base_model, config, mode) if (optim_step % five_percent_interval == 0) else {"balance": 0.0}

                    if is_main():
                        if optim_step % five_percent_interval == 0:
                            print(f"[OptStep {optim_step}] Valid Loss {valid_loss:.4f}, PPL {valid_ppl:.2f}, Balance {stats['balance']:.4f}")
                            if writer:
                                writer.add_scalar("valid/balance", stats["balance"], optim_step)
                        if writer:
                            writer.add_scalar("valid/loss", valid_loss, optim_step)
                            writer.add_scalar("valid/ppl", valid_ppl, optim_step)
                        if valid_loss < best_loss:
                            best_loss = valid_loss
                            save_checkpoint(base_model, optimizer, scheduler, optim_step, best_loss, total_train_steps, save_dir, "best_checkpoint.safetensors")
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=loss.item(),
                    main=main_loss.item(),
                    aux=aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0.0
                )
        
        # 🔹 2중 루프 탈출 처리
        if reached_budget:
            if is_main():
                print(f"✅ Reached training budget at step {optim_step}/{total_steps}")
            break

    if progress_bar: progress_bar.close()

    if is_main():
        save_checkpoint(base_model, optimizer, scheduler, optim_step, best_loss, total_train_steps, save_dir, "last_checkpoint.safetensors")
    
    # 🔹 종료 시 프로세스 정리
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
