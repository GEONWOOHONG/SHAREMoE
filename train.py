# train.py — 학습/평가/통계 (원본 로직 그대로)
import os, math, torch, tiktoken, shutil
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import load_model

from config import RUNS_DIR, HASH_TABLE_PATH
from data import load_or_prepare_pile
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

# 환경 변수는 사용자 환경에서 설정했다고 가정 (원본 주석 유지 불필요)

ensure_flash_attn()  # 원본과 동일하게 모듈 레벨에서 즉시 호출

enc = tiktoken.get_encoding("gpt2")

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    losses = []
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        labels[labels == enc.eot_token] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=batch["attention_mask"].to(device),
                labels=labels,
                # >>> 추가: 평가 시엔 항상 Stage-2 사용
                global_step=(1 << 62),
            )
            losses.append(outputs.loss.item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)
    return mean_loss, ppl

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
        if moe.mode == "expert_choice" and moe.last_scores is not None:
            probs = moe.last_scores.mean(0)
            entropy = -(probs * (probs + 1e-9).log()).sum().item()
            balance_scores.append(entropy)
        elif hasattr(moe, 'router') and getattr(moe.router, 'last_scores', None) is not None:
            probs = moe.router.last_scores.mean(0)
            entropy = -(probs * (probs + 1e-9).log()).sum().item()
            balance_scores.append(entropy)
    return {"balance": sum(balance_scores) / len(balance_scores) if balance_scores else 0.0}

def train_moe(mode="switch", num_experts=8, batch_size=32, seq_len=1024, continue_training=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = f"/workspace/checkpoints/{mode}_exp1"
    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 Checkpoint directory: {save_dir}")

    eff_num_experts = num_experts * 4 if mode == "xmoe" else num_experts
    if mode == "xmoe" and eff_num_experts != num_experts:
        print(f"🧮 xmoe mode: num_experts overridden {num_experts} → {eff_num_experts} (×4)")

    freq_dict = None
    if mode == "hash":
        if not os.path.exists(HASH_TABLE_PATH):
            raise FileNotFoundError(f"Hash table not found at {HASH_TABLE_PATH}\nPlease run `create_global_hash_table()` first.")
        print(f"🔹 Loading global hash table from: {HASH_TABLE_PATH}")
        freq_dict = {'__load_from_file__': HASH_TABLE_PATH}

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

    if mode == "hash":
        patch_model_for_hash_moe(model)
    elif mode == "ours_com":
        patch_model_for_ours_com(model)
    elif mode == "stablemoe":
        patch_model_for_stablemoe(model)
    elif mode != "dense":
        print(f"🔹 Applying forward patches for mode: {mode}")
        patch_model_basic(model)

    train_dataset, valid_dataset = load_or_prepare_pile()
    print(f"✅ Using FULL datasets: train={len(train_dataset):,}, valid={len(valid_dataset):,}")

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    NUM_WORKERS = max(1, os.cpu_count() // 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, prefetch_factor=2, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=NUM_WORKERS,
                              pin_memory=True, prefetch_factor=2, persistent_workers=True, shuffle=False)

    from utils import print_model_info as _pmi
    _pmi(model, config, mode, eff_num_experts, batch_size=batch_size, grad_accum_steps=1, effective_batch=batch_size)

    if not os.path.exists(os.path.join(save_dir, "config.json")):
        config.save_pretrained(save_dir)

    model.to(device)
    optimizer = get_default_optimizer(model)

    runs_root = RUNS_DIR
    if os.path.exists(runs_root):
        shutil.rmtree(runs_root)
    writer = SummaryWriter(log_dir=os.path.join(runs_root, "exp1"))

    if continue_training and trainer_state is not None:
        optimizer.load_state_dict(trainer_state["optimizer"])
        total_steps = trainer_state.get("total_train_steps", len(train_loader))
        scheduler = get_default_scheduler(optimizer, total_steps, warmup_ratio=0.1)
        if "scheduler" in trainer_state:
            scheduler.load_state_dict(trainer_state["scheduler"])
        best_loss   = trainer_state["best_loss"]
        start_step  = trainer_state["step"]
        try:
            print("🔎 Resume LR:", scheduler.get_last_lr()[0])
        except Exception:
            print("🔎 Resume LR (param_group[0]):", optimizer.param_groups[0]["lr"])
        print(f"🔹 Resumed training from step {start_step}, best_loss={best_loss:.4f}")
    else:
        total_steps = len(train_loader)
        scheduler   = get_default_scheduler(optimizer, total_steps, warmup_ratio=0.1)
        best_loss   = float("inf")
        start_step  = 0

    if mode == "stablemoe":
        stage1_ratio = 0.10
        stage1_steps = max(1, int(round(stage1_ratio * total_steps)))

        from modeling import GPT2LayerMoE
        for m in model.modules():
            if isinstance(m, GPT2LayerMoE) and m.mode == "stablemoe":
                m.moe.stable_stage1_steps = int(stage1_steps)

        print(f"🧭 StableMoE schedule: Stage-1 = {stage1_steps} steps "
            f"({stage1_ratio*100:.1f}% of {total_steps}), Stage-2 thereafter.")
        
        # Stage-2 진입 시 명시적 고정 보장 (안전 가드)
        print("🔒 StableMoE: Pre-freezing routing components for Stage-2 safety...")
        for m in model.modules():
            if isinstance(m, GPT2LayerMoE) and m.mode == "stablemoe":
                # Stage-1 종료 직후 즉시 freeze 준비 (forward에서도 안전가드 동작)
                # 논문: Stage-2에서 D(·)와 Ě를 동결
                pass  # forward에서 _maybe_freeze_stage2()가 처리하지만, 명시적 준비
        
        writer.add_scalar("stablemoe/stage1_steps", stage1_steps, 0)
        writer.add_text("stablemoe/config",
                        f"stage1_ratio={stage1_ratio}, total_steps={total_steps}", 0)
    
    tokens_per_batch = batch_size * seq_len
    total_tokens = len(train_loader) * tokens_per_batch
    print(f"🔹 Training for 1 epoch = {len(train_loader):,} steps ≈ {total_tokens/1e9:.2f}B tokens")

    progress_bar = tqdm(total=len(train_loader), desc="Training", leave=True)
    total_train_steps = total_steps
    point_one_step = max(1, total_train_steps // 1000)
    five_percent_interval = max(1, total_train_steps // 20)

    for step, batch in enumerate(train_loader, start=start_step + 1):
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        labels[labels == enc.eot_token] = -100

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=batch["attention_mask"].to(device),
                labels=labels,
                global_step=step,
            )
            main_loss = outputs.loss

        balance_losses = [m.last_balance_loss for m in model.modules()
                          if isinstance(m, GPT2LayerMoE) and m.last_balance_loss is not None]
        aux_loss = torch.stack(balance_losses).mean() if balance_losses else 0.0
        loss = main_loss + aux_loss

        # stablemoe 보조 로스 분해 로깅
        if mode == "stablemoe":
            balances, distills, overflows = [], [], []
            for m in model.modules():
                if isinstance(m, GPT2LayerMoE) and hasattr(m, "moe") and isinstance(getattr(m, "moe", None), type(model.transformer.h[0].mlp.moe)):
                    aux = m.moe.last_aux
                    if aux:
                        if aux["balance"] is not None: balances.append(float(aux["balance"]))
                        if aux["distill"]  is not None: distills.append(float(aux["distill"]))
                        if aux["overflow_rate"] is not None: overflows.append(float(aux["overflow_rate"]))
            if balances: writer.add_scalar("stablemoe/balance_loss", sum(balances)/len(balances), step)
            if distills: writer.add_scalar("stablemoe/distill_loss",  sum(distills)/len(distills), step)
            if overflows: writer.add_scalar("stablemoe/overflow_rate", sum(overflows)/len(overflows), step)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item(), main=main_loss.item(),
                                 aux=aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0.0)
        writer.add_scalar("train/loss", loss.item(), step)

        if step % five_percent_interval == 0 or step == point_one_step:
            msg = (f"🔍 Full validation (every 5%) at step {step}..."
                   if step % five_percent_interval == 0 else f"🔍 Quick validation (0.1%) at step {step}...")
            progress_bar.write(msg)
            valid_loss, valid_ppl = evaluate(model, valid_loader, device)
            stats = compute_moe_stats(model, config, mode) if step % five_percent_interval == 0 else {"balance": 0.0}
            if step % five_percent_interval == 0:
                progress_bar.write(f"[Step {step}] Valid Loss {valid_loss:.4f}, PPL {valid_ppl:.2f}, Balance {stats['balance']:.4f}")
                writer.add_scalar("valid/balance", stats["balance"], step)
            writer.add_scalar("valid/loss", valid_loss, step)
            writer.add_scalar("valid/ppl", valid_ppl, step)
            if valid_loss < best_loss:
                best_loss = valid_loss
                save_checkpoint(model, optimizer, scheduler, step, best_loss, total_train_steps, save_dir, "best_checkpoint.safetensors")

        if step % 50 == 0:
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

    progress_bar.close()
    save_checkpoint(model, optimizer, scheduler, step, best_loss, total_train_steps, save_dir, "last_checkpoint.safetensors")
