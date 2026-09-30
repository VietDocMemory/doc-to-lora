# Chạy Doc-to-LoRA evaluation trên Molab

Notebook [`molab_d2l_eval.py`](molab_d2l_eval.py) là một Marimo app reactive dùng
GPU CUDA, BF16, FlashAttention-2, `fsspec` và Weights & Biases.

## 1. Biến môi trường

Bắt buộc (trừ khi model đã có trong Hugging Face cache hoặc W&B chạy offline):

```bash
export HF_TOKEN="hf_..."
export WANDB_API_KEY="..."
```

Cấu hình filesystem và dữ liệu:

```bash
# Local virtual sandbox (mặc định là file://.)
export MOLAB_FS_URL="file:///workspace/doc-to-lora"

# Có thể thay bằng gs://bucket/prefix hoặc gdrive://...
# Các option dành riêng cho backend được truyền dưới dạng JSON.
export MOLAB_FS_OPTIONS_JSON='{}'

export D2L_DATA_ROOT="/workspace/doc-to-lora/data/eval"
export D2L_OUTPUT_ROOT="/workspace/doc-to-lora/eval_results/molab"
export D2L_CHECKPOINT="/workspace/doc-to-lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
export D2L_BASE_MODEL="google/gemma-2-2b-it"
export D2L_MAX_INPUT_TOKENS="16384"
```

Với GCS, dùng Application Default Credentials của workload hoặc set
`GOOGLE_APPLICATION_CREDENTIALS`; ví dụ `MOLAB_FS_URL=gs://my-bucket/d2l`. Với
Google Drive, backend `gdrivefs` cần credential/options phù hợp được truyền qua
`MOLAB_FS_OPTIONS_JSON`. Có thể đặt `WANDB_MODE=offline` để thử mà không gửi dữ
liệu lên W&B.

> `D2L_DATA_ROOT`, `D2L_OUTPUT_ROOT` và `D2L_CHECKPOINT` là path **bên trong**
> filesystem đã chọn, không phải URL có protocol. Nếu không set, chúng được nối
> tự động bên dưới root của `MOLAB_FS_URL`.

## 2. Chuẩn bị dữ liệu

Tạo một file theo tên dataset, ví dụ `data/eval/squad.jsonl`. Mỗi dòng hỗ trợ
một trong các schema phổ biến sau:

```json
{"id":"1","context":"...","question":"...","answers":{"text":["..."]}}
{"id":"2","context":"...","input":"...","answers":["...", "..."]}
{"id":"3","context":"...","prompts":["..."],"responses":["..."]}
```

Notebook cũng hỗ trợ `.json`, các field `passage`, `background` + `situation`,
`prompt`, `gold`, `response`, `answer`, và `answers_spans.spans`. Dataset
LongBench dùng tên file rút gọn như `2wikimqa_e.jsonl`, `qasper_e.jsonl`.

## 3. Import vào Molab

1. Upload/clone toàn bộ repository `doc-to-lora` vào workspace Molab (notebook
   cần package local `ctx_to_lora`).
2. Upload checkpoint và các file dataset vào backend `fsspec` đã chọn.
3. Thêm các biến môi trường/secret ở phần cấu hình runtime của Molab.
4. Mở `molab_d2l_eval.py` trong Molab và chọn runtime GPU Blackwell 96 GB.
5. Chọn dataset, batch size (khuyến nghị `4` trở lên), số token và chế độ D2L.

Molab/Marimo có thể đọc dependency PEP 723 nằm ngay đầu notebook; metadata cũng
cài editable package `ctx-to-lora` từ repository hiện tại. Nếu image đã có sẵn
CUDA PyTorch và FlashAttention-2 thì nên dùng các package đó để giữ đúng CUDA ABI
của runtime.

## 4. Chạy local

Trong môi trường project đã cài `flash-attn` tương thích CUDA:

```bash
uv add --dev marimo fsspec gcsfs gdrivefs s3fs psutil
uv run marimo edit molab_d2l_eval.py
# hoặc
uv run marimo run molab_d2l_eval.py
```

Để Marimo tự tạo sandbox từ metadata trong file:

```bash
uvx marimo edit --sandbox molab_d2l_eval.py
```

`FlashAttention-2` là bắt buộc cho notebook này. Trên Molab, ưu tiên wheel/image
được nền tảng cung cấp thay vì build lại `flash-attn` không khớp CUDA/PyTorch.

## 5. Output

Mỗi lần chạy tạo:

- `predictions.jsonl`: prediction, gold, F1/Precision/Recall và latency từng mẫu.
- `summary.json`: Macro F1/Recall và latency tổng hợp.
- một `wandb.Table` và một W&B Artifact chứa output; ở D2L mode artifact còn
  chứa checkpoint local hoặc reference tới checkpoint cloud.

W&B đồng thời nhận metrics từng step, GPU allocated/reserved/peak VRAM và RAM hệ
thống. Thay đổi UI tạo một run reactive mới và run trước được kết thúc tự động.
