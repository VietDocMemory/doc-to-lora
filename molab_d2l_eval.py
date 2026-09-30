# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "accelerate>=1.6.0",
#     "ctx-to-lora",
#     "fsspec>=2025.3.0",
#     "gcsfs>=2025.3.0",
#     "gdrivefs>=0.1.5",
#     "huggingface-hub>=0.32.0",
#     "marimo>=0.15.0",
#     "peft>=0.15.0",
#     "psutil>=6.1.0",
#     "s3fs>=2025.3.0",
#     "torch>=2.7.0",
#     "transformers==4.51.3",
#     "wandb>=0.19.0",
# ]
# [tool.uv.sources]
# ctx-to-lora = { path = ".", editable = true }
# ///

"""Đánh giá Doc-to-LoRA trên Molab bằng một notebook Marimo reactive."""

import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full", app_title="Doc-to-LoRA Evaluation on Molab")


# Cell 1: dependencies dùng chung. PEP 723 ở đầu file cho phép Marimo/uv cài đặt.
@app.cell
def _():
    import importlib.metadata as importlib_metadata
    import json
    import os
    import posixpath
    import re
    import string
    import time
    import unicodedata
    from collections import defaultdict

    import fsspec
    import marimo as mo
    import peft
    import psutil
    import torch
    import wandb
    from huggingface_hub import login as hf_login
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Nếu chạy không có --sandbox, thông báo này giúp phát hiện môi trường thiếu gói sớm.
    required_packages = (
        "marimo",
        "fsspec",
        "huggingface-hub",
        "wandb",
        "torch",
        "transformers",
        "peft",
    )
    dependency_versions = {
        package: importlib_metadata.version(package) for package in required_packages
    }
    assert peft.__version__ == dependency_versions["peft"]

    mo.md(
        "### Môi trường đã sẵn sàng\n"
        + " · ".join(
            f"`{name}={version}`" for name, version in dependency_versions.items()
        )
    )
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        defaultdict,
        fsspec,
        hf_login,
        json,
        mo,
        os,
        posixpath,
        psutil,
        re,
        string,
        time,
        torch,
        unicodedata,
        wandb,
    )


# Cell 2: filesystem trừu tượng và xác thực Hugging Face / Weights & Biases.
@app.cell
def _(fsspec, hf_login, json, mo, os, posixpath, wandb):
    storage_url = os.environ.get("MOLAB_FS_URL", "file://.")
    storage_options = json.loads(os.environ.get("MOLAB_FS_OPTIONS_JSON", "{}"))
    fs, fs_root = fsspec.core.url_to_fs(storage_url, **storage_options)

    # Các đường dẫn dưới đây là path nội bộ của filesystem, không phải path local cố định.
    data_root = os.environ.get(
        "D2L_DATA_ROOT", posixpath.join(fs_root, "data", "eval")
    )
    output_root = os.environ.get(
        "D2L_OUTPUT_ROOT", posixpath.join(fs_root, "eval_results", "molab")
    )
    checkpoint_path = os.environ.get(
        "D2L_CHECKPOINT",
        posixpath.join(
            fs_root,
            "trained_d2l",
            "gemma_demo",
            "checkpoint-80000",
            "pytorch_model.bin",
        ),
    )
    base_model_id = os.environ.get("D2L_BASE_MODEL", "google/gemma-2-2b-it")

    hf_token = os.environ.get("HF_TOKEN")
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if hf_token:
        hf_login(token=hf_token, add_to_git_credential=False, skip_if_logged_in=True)
    if wandb_api_key:
        wandb.login(key=wandb_api_key, relogin=True)

    auth_status = {
        "filesystem": storage_url,
        "hf_token": "đã cấu hình" if hf_token else "chưa cấu hình",
        "wandb_api_key": "đã cấu hình" if wandb_api_key else "chưa cấu hình",
    }
    mo.callout(
        mo.md(
            f"**Filesystem:** `{storage_url}`  \n"
            f"**HF_TOKEN:** {auth_status['hf_token']}  \n"
            f"**WANDB_API_KEY:** {auth_status['wandb_api_key']}"
        ),
        kind="success" if hf_token and wandb_api_key else "warn",
    )
    return (
        base_model_id,
        checkpoint_path,
        data_root,
        fs,
        fs_root,
        hf_token,
        output_root,
    )


# Cell 3: tham số thực nghiệm tương tác.
@app.cell
def _(mo):
    dataset_select = mo.ui.dropdown(
        options=[
            "squad",
            "drop",
            "ropes",
            "2wikimqa_e",
            "qasper_e",
            "multifieldqa_en_e",
        ],
        value="squad",
        label="Chọn Dataset",
    )
    batch_size_slider = mo.ui.slider(
        start=1, stop=16, step=1, value=4, label="Eval Batch Size"
    )
    max_tokens_slider = mo.ui.slider(
        start=64, stop=512, step=64, value=128, label="Max New Tokens"
    )
    use_d2l_toggle = mo.ui.checkbox(
        value=True, label="Kích hoạt Doc-to-LoRA (Tắt: chạy Base Model ICL)"
    )
    controls_panel = mo.hstack(
        [
            dataset_select,
            batch_size_slider,
            max_tokens_slider,
            use_d2l_toggle,
        ],
        justify="space-around",
        wrap=True,
    )
    controls_panel
    return (
        batch_size_slider,
        dataset_select,
        max_tokens_slider,
        use_d2l_toggle,
    )


# Cell 4: kiểm tra Blackwell và mở một run W&B theo cấu hình UI hiện tại.
@app.cell
def _(
    batch_size_slider,
    dataset_select,
    max_tokens_slider,
    mo,
    os,
    torch,
    use_d2l_toggle,
    wandb,
):
    if not torch.cuda.is_available():
        raise RuntimeError("Notebook này cần GPU CUDA; Molab phải được bật GPU Blackwell.")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_properties = torch.cuda.get_device_properties(0)
    gpu_total_vram_gb = gpu_properties.total_memory / (1024**3)
    gpu_compute_capability = torch.cuda.get_device_capability(0)
    bf16_supported = torch.cuda.is_bf16_supported()
    is_blackwell = (
        "blackwell" in gpu_name.lower() or gpu_compute_capability[0] >= 10
    )
    if not bf16_supported:
        raise RuntimeError(f"GPU {gpu_name} không hỗ trợ bfloat16.")

    experiment_config = {
        "dataset": dataset_select.value,
        "eval_batch_size_gen": batch_size_slider.value,
        "max_new_tokens": max_tokens_slider.value,
        "use_d2l": use_d2l_toggle.value,
        "gpu_name": gpu_name,
        "gpu_total_vram_gb": round(gpu_total_vram_gb, 2),
        "gpu_compute_capability": ".".join(map(str, gpu_compute_capability)),
        "is_blackwell": is_blackwell,
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
    }
    wb_run = wandb.init(
        project="KLTN-Doc2LoRA-Blackwell",
        config=experiment_config,
        mode=os.environ.get("WANDB_MODE", "online"),
        reinit="finish_previous",
        tags=["molab", "blackwell", "d2l" if use_d2l_toggle.value else "base-icl"],
    )

    hardware_info = {
        **experiment_config,
        "bf16_supported": bf16_supported,
    }
    mo.callout(
        mo.md(
            f"### GPU: {gpu_name}\n"
            f"- VRAM: **{gpu_total_vram_gb:.1f} GiB**\n"
            f"- Compute capability: **{gpu_compute_capability[0]}."
            f"{gpu_compute_capability[1]}**\n"
            f"- bfloat16: **{bf16_supported}** · Blackwell: **{is_blackwell}**"
        ),
        kind="success" if is_blackwell else "warn",
    )
    return (hardware_info, wb_run)


# Cell 5: nạp model/tokenizer trong một hàm, không làm rò rỉ state trung gian.
@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    base_model_id,
    checkpoint_path,
    fs,
    hf_token,
    mo,
    torch,
    use_d2l_toggle,
    wb_run,
):
    def load_evaluation_model(use_d2l, checkpoint_uri, model_id, filesystem, token):
        """Nạp D2L checkpoint hoặc base model ở BF16 + FlashAttention-2."""
        if use_d2l:
            from ctx_to_lora.model_loading import get_tokenizer
            from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

            if not filesystem.exists(checkpoint_uri):
                raise FileNotFoundError(
                    f"Không tìm thấy D2L checkpoint trên filesystem: {checkpoint_uri}"
                )
            with filesystem.open(checkpoint_uri, "rb") as checkpoint_stream:
                state_dict = torch.load(
                    checkpoint_stream, map_location="cpu", weights_only=False
                )
            checkpoint_model_id = state_dict.get("base_model_name_or_path", model_id)
            loaded_model = ModulatedPretrainedModel.from_state_dict(
                state_dict,
                train=False,
                base_model_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "device_map": "cuda",
                    "attn_implementation": "flash_attention_2",
                },
                use_flash_attn=True,
                use_sequence_packing=False,
            )
            loaded_tokenizer = get_tokenizer(checkpoint_model_id, train=False)
            loaded_model.reset()
        else:
            loaded_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=token,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=True,
            )
            loaded_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=token,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )

        if loaded_tokenizer.pad_token_id is None:
            loaded_tokenizer.pad_token_id = loaded_tokenizer.eos_token_id
        loaded_tokenizer.padding_side = "left"
        loaded_tokenizer.truncation_side = "left"
        loaded_model.eval()
        return loaded_model, loaded_tokenizer

    model, tokenizer = load_evaluation_model(
        use_d2l_toggle.value,
        checkpoint_path,
        base_model_id,
        fs,
        hf_token,
    )
    wb_run.config.update(
        {
            "resolved_model_name": getattr(model, "base_model_name_or_path", None)
            or getattr(model.config, "name_or_path", base_model_id),
            "checkpoint_path": checkpoint_path if use_d2l_toggle.value else None,
        },
        allow_val_change=True,
    )
    mo.md(
        f"✅ Đã nạp **{'Doc-to-LoRA' if use_d2l_toggle.value else 'Base ICL'}** "
        "bằng `bfloat16` và `flash_attention_2`."
    )
    return (model, tokenizer)


# Cell 6: đọc dữ liệu, internalize, sinh batch, tính QA metrics và log realtime.
@app.cell
def _(
    batch_size_slider,
    data_root,
    dataset_select,
    defaultdict,
    fs,
    json,
    max_tokens_slider,
    model,
    os,
    posixpath,
    psutil,
    re,
    string,
    time,
    tokenizer,
    torch,
    unicodedata,
    use_d2l_toggle,
    wandb,
    wb_run,
):
    def _read_json_records(filesystem, path):
        with filesystem.open(path, "r", encoding="utf-8") as source:
            if path.lower().endswith(".jsonl"):
                return [json.loads(line) for line in source if line.strip()]
            payload = json.load(source)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "samples", "records"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        raise ValueError("JSON phải là một list hoặc chứa key data/samples/records.")

    def _as_gold_list(value):
        if isinstance(value, dict):
            value = value.get("text", value.get("spans", value.get("answers", [])))
        if value is None:
            return []
        if isinstance(value, (str, int, float)):
            return [str(value)]
        return [str(item) for item in value]

    def _normalise_records(raw_records):
        normalised = []
        for raw_index, raw in enumerate(raw_records):
            context = raw.get("context") or raw.get("passage")
            if not context and (raw.get("background") or raw.get("situation")):
                context = "\n".join(
                    part
                    for part in (raw.get("background"), raw.get("situation"))
                    if part
                )
            context = str(context or "")
            questions = raw.get("prompts")
            responses = raw.get("responses")
            if isinstance(questions, list):
                response_list = responses if isinstance(responses, list) else []
                for pair_index, question in enumerate(questions):
                    gold_value = (
                        response_list[pair_index]
                        if pair_index < len(response_list)
                        else raw.get("answers", raw.get("answer"))
                    )
                    normalised.append(
                        {
                            "sample_id": f"{raw.get('id', raw_index)}:{pair_index}",
                            "context": context,
                            "question": str(question),
                            "golds": _as_gold_list(gold_value),
                        }
                    )
                continue

            question = raw.get("question", raw.get("prompt", raw.get("input", "")))
            gold_value = raw.get(
                "gold",
                raw.get(
                    "answers",
                    raw.get(
                        "answers_spans", raw.get("response", raw.get("answer"))
                    ),
                ),
            )
            normalised.append(
                {
                    "sample_id": str(raw.get("id", raw_index)),
                    "context": context,
                    "question": str(question),
                    "golds": _as_gold_list(gold_value),
                }
            )
        invalid = [row["sample_id"] for row in normalised if not row["golds"]]
        if invalid:
            raise ValueError(f"Các sample không có ground truth: {invalid[:5]}")
        return normalised

    def _normalise_answer(text):
        lowered = unicodedata.normalize("NFKC", str(text)).lower()
        no_punctuation = "".join(
            " " if char in string.punctuation else char for char in lowered
        )
        no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
        return " ".join(no_articles.split())

    def _qa_scores(prediction, gold):
        pred_tokens = _normalise_answer(prediction).split()
        gold_tokens = _normalise_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            exact_empty = float(pred_tokens == gold_tokens)
            return exact_empty, exact_empty, exact_empty
        pred_counts = defaultdict(int)
        gold_counts = defaultdict(int)
        for token_value in pred_tokens:
            pred_counts[token_value] += 1
        for token_value in gold_tokens:
            gold_counts[token_value] += 1
        common = sum(
            min(count, gold_counts[token_value])
            for token_value, count in pred_counts.items()
        )
        if common == 0:
            return 0.0, 0.0, 0.0
        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        f1_value = 2 * precision * recall / (precision + recall)
        return f1_value, precision, recall

    def _max_over_ground_truths(prediction, golds):
        candidates = [_qa_scores(prediction, gold) for gold in golds]
        return max(candidates, key=lambda score: score[0])

    def _instruction(dataset_name, question):
        if dataset_name == "qasper_e":
            return (
                "Answer as concisely as possible. If the article does not contain "
                'the answer, write "unanswerable". Output only the answer.\n\n'
                f"Question: {question}"
            )
        return (
            "Answer the following question using only the supplied document. "
            "Output only the answer.\n\n"
            f"Question: {question}"
        )

    def _chat_prompt(active_tokenizer, user_text):
        messages = [{"role": "user", "content": user_text}]
        if getattr(active_tokenizer, "chat_template", None):
            return active_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return user_text

    def _cuda_sync():
        torch.cuda.synchronize()

    def _run_evaluation(
        records,
        active_model,
        active_tokenizer,
        dataset_name,
        batch_size,
        max_new_tokens,
        use_d2l,
    ):
        prediction_table = wandb.Table(
            columns=[
                "sample_id",
                "question",
                "context_len",
                "gold",
                "prediction",
                "f1",
                "recall",
                "prep_latency",
                "gen_latency",
            ]
        )
        grouped_records = defaultdict(list)
        for item in records:
            grouped_records[item["context"]].append(item)

        rows = []
        prep_latencies = []
        generation_latencies = []
        evaluation_started = time.perf_counter()
        global_step = 0
        max_input_tokens = int(os.environ.get("D2L_MAX_INPUT_TOKENS", "16384"))

        for context, context_records in grouped_records.items():
            prep_latency = 0.0
            prep_peak_mb = 0.0
            if use_d2l:
                active_model.reset()
                torch.cuda.reset_peak_memory_stats()
                _cuda_sync()
                prep_started = time.perf_counter()
                active_model.internalize(context)
                _cuda_sync()
                prep_latency = time.perf_counter() - prep_started
                prep_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
                prep_latencies.append(prep_latency)

            context_len = len(
                active_tokenizer.encode(context, add_special_tokens=False)
            )
            for batch_start in range(0, len(context_records), batch_size):
                batch = context_records[batch_start : batch_start + batch_size]
                prompt_texts = []
                for item in batch:
                    instruction = _instruction(dataset_name, item["question"])
                    user_text = (
                        instruction
                        if use_d2l
                        else f"Document:\n{context}\n\n{instruction}"
                    )
                    prompt_texts.append(_chat_prompt(active_tokenizer, user_text))

                tokenized = active_tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_tokens,
                ).to(active_model.device)
                torch.cuda.reset_peak_memory_stats()
                _cuda_sync()
                generation_started = time.perf_counter()
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "use_cache": True,
                    "pad_token_id": active_tokenizer.pad_token_id,
                    "eos_token_id": active_tokenizer.eos_token_id,
                }
                if use_d2l:
                    # Một LoRA đã internalize được broadcast cho toàn bộ query trong batch.
                    generation_kwargs["n_queries"] = torch.tensor(
                        [len(batch)], dtype=torch.int32, device=active_model.device
                    )
                with torch.inference_mode():
                    generated = active_model.generate(**tokenized, **generation_kwargs)
                _cuda_sync()
                batch_gen_latency = time.perf_counter() - generation_started
                per_sample_gen_latency = batch_gen_latency / len(batch)
                generation_latencies.extend([per_sample_gen_latency] * len(batch))
                generated_only = generated[:, tokenized["input_ids"].shape[1] :]
                predictions = active_tokenizer.batch_decode(
                    generated_only, skip_special_tokens=True
                )
                gen_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

                for item, prediction in zip(batch, predictions):
                    cleaned_prediction = prediction.strip()
                    f1_value, precision, recall = _max_over_ground_truths(
                        cleaned_prediction, item["golds"]
                    )
                    ram_used_mb = psutil.virtual_memory().used / (1024**2)
                    allocated_mb = torch.cuda.memory_allocated() / (1024**2)
                    reserved_mb = torch.cuda.memory_reserved() / (1024**2)
                    result_row = {
                        "sample_id": item["sample_id"],
                        "question": item["question"],
                        "context_len": context_len,
                        "gold": item["golds"],
                        "prediction": cleaned_prediction,
                        "f1": f1_value,
                        "precision": precision,
                        "recall": recall,
                        "prep_latency": prep_latency,
                        "gen_latency": per_sample_gen_latency,
                        "gpu_peak_mem_mb": max(prep_peak_mb, gen_peak_mb),
                    }
                    rows.append(result_row)
                    prediction_table.add_data(
                        item["sample_id"],
                        item["question"],
                        context_len,
                        " | ".join(item["golds"]),
                        cleaned_prediction,
                        f1_value,
                        recall,
                        prep_latency,
                        per_sample_gen_latency,
                    )
                    wb_run.log(
                        {
                            "step_f1": f1_value,
                            "step_precision": precision,
                            "step_recall": recall,
                            "step_prep_latency": prep_latency,
                            "step_gen_latency": per_sample_gen_latency,
                            "gpu_mem_allocated_mb": allocated_mb,
                            "gpu_mem_reserved_mb": reserved_mb,
                            "gpu_peak_mem_mb": max(prep_peak_mb, gen_peak_mb),
                            "system_ram_used_mb": ram_used_mb,
                        },
                        step=global_step,
                    )
                    global_step += 1

        if use_d2l:
            active_model.reset()
        return {
            "rows": rows,
            "prediction_table": prediction_table,
            "prep_latencies": prep_latencies,
            "generation_latencies": generation_latencies,
            "evaluation_wall_time_s": time.perf_counter() - evaluation_started,
            "num_contexts": len(grouped_records),
        }

    dataset_filename = f"{dataset_select.value}.jsonl"
    dataset_path = posixpath.join(data_root, dataset_filename)
    if not fs.exists(dataset_path):
        json_fallback = posixpath.join(data_root, f"{dataset_select.value}.json")
        if fs.exists(json_fallback):
            dataset_path = json_fallback
        else:
            raise FileNotFoundError(
                f"Không tìm thấy {dataset_path} hoặc {json_fallback} trên filesystem."
            )

    raw_records = _read_json_records(fs, dataset_path)
    evaluation_records = _normalise_records(raw_records)
    eval_bundle = _run_evaluation(
        evaluation_records,
        model,
        tokenizer,
        dataset_select.value,
        batch_size_slider.value,
        max_tokens_slider.value,
        use_d2l_toggle.value,
    )
    return (dataset_path, eval_bundle)


# Cell 7: tổng hợp, ghi qua fsspec, log Artifact và đóng run.
@app.cell
def _(
    checkpoint_path,
    dataset_path,
    dataset_select,
    eval_bundle,
    fs,
    json,
    mo,
    output_root,
    posixpath,
    time,
    use_d2l_toggle,
    wandb,
    wb_run,
):
    result_rows = eval_bundle["rows"]
    if not result_rows:
        raise RuntimeError("Dataset không chứa sample hợp lệ để đánh giá.")

    overall_macro_f1 = sum(row["f1"] for row in result_rows) / len(result_rows)
    overall_macro_precision = sum(row["precision"] for row in result_rows) / len(
        result_rows
    )
    overall_macro_recall = sum(row["recall"] for row in result_rows) / len(
        result_rows
    )
    average_generation_s = sum(eval_bundle["generation_latencies"]) / len(
        eval_bundle["generation_latencies"]
    )
    average_internalization_s = (
        sum(eval_bundle["prep_latencies"]) / len(eval_bundle["prep_latencies"])
        if eval_bundle["prep_latencies"]
        else 0.0
    )
    average_e2e_s = eval_bundle["evaluation_wall_time_s"] / len(result_rows)

    aggregate_metrics = {
        "dataset": dataset_select.value,
        "mode": "d2l" if use_d2l_toggle.value else "base_icl",
        "num_samples": len(result_rows),
        "num_contexts": eval_bundle["num_contexts"],
        "overall_macro_f1": overall_macro_f1,
        "overall_macro_precision": overall_macro_precision,
        "overall_macro_recall": overall_macro_recall,
        "avg_internalization_latency_s": average_internalization_s,
        "avg_generation_latency_s": average_generation_s,
        "avg_e2e_latency_s": average_e2e_s,
        "evaluation_wall_time_s": eval_bundle["evaluation_wall_time_s"],
        "source_dataset_path": dataset_path,
    }
    wb_run.summary.update(aggregate_metrics)
    wb_run.log({"predictions": eval_bundle["prediction_table"]})

    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    run_output_dir = posixpath.join(
        output_root,
        dataset_select.value,
        f"{aggregate_metrics['mode']}-{run_stamp}",
    )
    fs.makedirs(run_output_dir, exist_ok=True)
    predictions_path = posixpath.join(run_output_dir, "predictions.jsonl")
    summary_path = posixpath.join(run_output_dir, "summary.json")
    with fs.open(predictions_path, "w", encoding="utf-8") as predictions_file:
        for row in result_rows:
            predictions_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with fs.open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(aggregate_metrics, summary_file, ensure_ascii=False, indent=2)

    result_artifact = wandb.Artifact(
        name=f"d2l-eval-{dataset_select.value}-{wb_run.id}",
        type="evaluation",
        metadata=aggregate_metrics,
    )
    fs_protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
    if fs_protocol in ("file", "local"):
        result_artifact.add_file(predictions_path, name="predictions.jsonl")
        result_artifact.add_file(summary_path, name="summary.json")
        if use_d2l_toggle.value:
            result_artifact.add_file(
                checkpoint_path,
                name=f"checkpoint/{posixpath.basename(checkpoint_path)}",
            )
    elif fs_protocol in ("gs", "gcs", "s3", "s3a", "http", "https"):
        result_artifact.add_reference(fs.unstrip_protocol(predictions_path))
        result_artifact.add_reference(fs.unstrip_protocol(summary_path))
        if use_d2l_toggle.value:
            result_artifact.add_reference(fs.unstrip_protocol(checkpoint_path))
    else:
        # Backend như Google Drive chưa được W&B hỗ trợ reference trực tiếp.
        # Đọc nguồn bằng fsspec rồi stream vào staging area do W&B quản lý.
        def _stream_into_artifact(source_path, artifact_name):
            with fs.open(source_path, "rb") as source_file:
                with result_artifact.new_file(artifact_name, mode="wb") as target_file:
                    while chunk := source_file.read(8 * 1024 * 1024):
                        target_file.write(chunk)

        _stream_into_artifact(predictions_path, "predictions.jsonl")
        _stream_into_artifact(summary_path, "summary.json")
        if use_d2l_toggle.value:
            _stream_into_artifact(
                checkpoint_path,
                f"checkpoint/{posixpath.basename(checkpoint_path)}",
            )
    wb_run.log_artifact(result_artifact)
    wandb.finish()

    summary_card = mo.md(
        f"""
        ## Kết quả `{dataset_select.value}`

        | Metric | Giá trị |
        |---|---:|
        | Samples / contexts | {len(result_rows)} / {eval_bundle['num_contexts']} |
        | Macro F1 | **{overall_macro_f1:.4f}** |
        | Macro Precision | {overall_macro_precision:.4f} |
        | Macro Recall | **{overall_macro_recall:.4f}** |
        | Internalization trung bình | {average_internalization_s:.3f} s/context |
        | Generation trung bình | {average_generation_s:.3f} s/sample |
        | E2E trung bình | {average_e2e_s:.3f} s/sample |

        Kết quả: `{run_output_dir}`  
        W&B run: `{wb_run.name}` (`{wb_run.id}`)
        """
    )
    summary_card
    return


if __name__ == "__main__":
    app.run()
