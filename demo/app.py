import gc
import os
import re
import sys
from importlib.util import find_spec
from pathlib import Path

import gradio as gr
import torch
from transformers import BitsAndBytesConfig

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ctx_to_lora.data.processing import split_too_long_ctx, tokenize_ctx_text
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling import hypernet

sys.modules["ctx_to_lora.modeling_utils"] = hypernet

# Global state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flash_attention_available = find_spec("flash_attn") is not None
modulated_model = None
chat_history = []
ctx_tokenizer = None
base_tokenizer = None
context_chunk_tokens = -1

try:
    DEFAULT_CONTEXT = Path("data/sakana_wiki.txt").read_text(encoding="utf-8").strip()
except FileNotFoundError:
    DEFAULT_CONTEXT = ""

WARNING_MESSAGE = (
    "⚠️ **Caution**: This is an educational proof-of-concept demonstration.\n"
    "The model may generate inaccurate information or hallucinate facts."
)

FOOTER = """
⚠️ This model is an experimental prototype and is only available for educational and research and development purposes. It is not suitable for commercial use or in environments where failure can have significant effects (mission-critical environments).
The use of this model is at the user's own risk and its performance and results is not guaranteed in any way.
Sakana AI is not responsible for any direct or indirect loss resulting from using this model, regardless of the outcome.
"""


def load_custom_chat_template(tokenizer, model_name):
    model_name_lower = model_name.lower()
    template_paths = {
        "gemma": "chat_templates/google/gemma-2-2b-it.jinja",
        "mistral": "chat_templates/mistralai/Mistral-7B-Instruct-v0.2.jinja",
        "qwen": "chat_templates/Qwen/Qwen3-4B-Instruct-2507.jinja",
    }
    for family, template_path in template_paths.items():
        if family not in model_name_lower or not os.path.exists(template_path):
            continue
        with open(template_path, encoding="utf-8") as template_file:
            tokenizer.chat_template = template_file.read()
        print(f"Loaded custom chat template from {template_path}")
        return True
    return False


def get_available_checkpoints():
    trained_d2l_checkpoints = {
        str(path)
        for path in Path().glob("trained_d2l/**/pytorch_model.bin")
        if path.is_file()
    }
    run_output_checkpoints = {
        str(path)
        for path in Path().glob("train_outputs/runs/**/pytorch_model.bin")
        if path.is_file()
    }
    checkpoints = sorted(trained_d2l_checkpoints) + sorted(
        run_output_checkpoints - trained_d2l_checkpoints
    )
    return checkpoints if checkpoints else ["No checkpoints found"]


def load_checkpoint(
    checkpoint_path: str,
) -> tuple[str, gr.update, gr.update, gr.update, gr.update]:
    global modulated_model, ctx_tokenizer, base_tokenizer, chat_history
    global context_chunk_tokens

    if not checkpoint_path or checkpoint_path == "No checkpoints found":
        return (
            "⚠️ Please select a valid checkpoint",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

        # Release the previous model before constructing another checkpoint.
        # Keeping both alive can exhaust a 10 GB GPU during model switches.
        old_model = modulated_model
        modulated_model = None
        ctx_tokenizer = None
        base_tokenizer = None
        del old_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        args_path = Path(checkpoint_path).parent.parent / "args.yaml"
        context_chunk_tokens = -1
        if args_path.exists():
            chunk_match = re.search(
                r"^max_ctx_chunk_len:\s*(-?\d+)\s*$",
                args_path.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
            if chunk_match:
                context_chunk_tokens = int(chunk_match.group(1))
        print(f"Context chunk size: {context_chunk_tokens}")
        model_name = state_dict["base_model_name_or_path"]
        gpu_memory = (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else 0
        )
        use_4bit_base = gpu_memory and gpu_memory < 16 * 1024**3 and any(
            family in model_name.lower() for family in ("qwen", "mistral")
        )
        base_model_kwargs = None
        if use_4bit_base:
            print(f"Loading {model_name} in 4-bit mode for the available GPU memory")
            base_model_kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            }
        modulated_model = ModulatedPretrainedModel.from_state_dict(
            state_dict,
            train=False,
            base_model_kwargs=base_model_kwargs,
            use_flash_attn=flash_attention_available,
            use_sequence_packing=False,
        )
        if use_4bit_base:
            # Transformers already places quantized modules; converting them via
            # Module.to(dtype) is unsupported. HyperLoRA remains safe to convert.
            modulated_model.hypernet.to(device=device, dtype=torch.bfloat16)
        else:
            modulated_model = modulated_model.to(device).to(torch.bfloat16)
        modulated_model.eval()

        ctx_encoder_model_name_or_path = (
            modulated_model.ctx_encoder_args.ctx_encoder_model_name_or_path
            or modulated_model.base_model.config.name_or_path
        )
        ctx_tokenizer = get_tokenizer(ctx_encoder_model_name_or_path, train=False)
        base_tokenizer = get_tokenizer(
            modulated_model.base_model.config.name_or_path, train=False
        )

        load_custom_chat_template(ctx_tokenizer, ctx_encoder_model_name_or_path)
        load_custom_chat_template(
            base_tokenizer, modulated_model.base_model.config.name_or_path
        )

        chat_history = [{"role": "system", "content": ""}]

        model_name = modulated_model.base_model.config.name_or_path
        success_msg = (
            f"✅ Successfully loaded checkpoint!\n\nBase Model: {model_name}\n\n"
            "You can now add context and start chatting."
        )
        return (
            success_msg,
            gr.update(interactive=True),  # msg
            gr.update(interactive=True),  # send_btn
            gr.update(interactive=True),  # system_msg
            gr.update(interactive=True),  # clear_btn
        )

    except Exception as e:
        import traceback

        error_msg = (
            f"❌ Error loading checkpoint:\n{str(e)}\n\n{traceback.format_exc()}"
        )
        print(error_msg)
        return (
            error_msg,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )


def process_context(context: str) -> dict:
    context = context.strip() if context else ""
    tokenized_contexts = tokenize_ctx_text({"context": [context]}, ctx_tokenizer)
    ctx_ids = tokenized_contexts["ctx_ids"]
    if context_chunk_tokens > 0:
        split_context = split_too_long_ctx(
            {"ctx_ids": ctx_ids[0]},
            model_name_or_path=ctx_tokenizer.name_or_path,
            num_chunk_probs=None,
            max_chunk_len=context_chunk_tokens,
            min_chunk_len=-1,
            max_num_split=None,
            is_train=False,
        )
        ctx_ids = split_context["ctx_ids"]
        print(f"Context chunks: {split_context['n_ctx_chunks']}")
    ctx_ids = [
        torch.tensor(ctx_id, dtype=torch.long, device=device) for ctx_id in ctx_ids
    ]
    ctx_attn_mask = [torch.ones_like(ids) for ids in ctx_ids]
    ctx_ids = torch.nn.utils.rnn.pad_sequence(
        ctx_ids,
        batch_first=True,
        padding_value=0,
    )
    ctx_attn_mask = torch.nn.utils.rnn.pad_sequence(
        ctx_attn_mask,
        batch_first=True,
        padding_value=0,
    )
    return {"ctx_ids": ctx_ids, "ctx_attn_mask": ctx_attn_mask}


def add_user_message(message: str, history):
    if not message.strip():
        return history, ""
    return history + [[message, None]], ""


def generate_response(
    history: list[list[str]],
    system_msg: str,
    context: str,
    context_scaler: float,
    bias_scaler: float,
):
    global modulated_model, chat_history, ctx_tokenizer, base_tokenizer

    if modulated_model is None:
        history[-1][1] = "Please load a checkpoint first."
        yield history
        return

    if not history or history[-1][0] is None:
        yield history
        return

    try:
        user_message = history[-1][0]

        if system_msg.strip() and chat_history[0]["role"] == "system":
            chat_history[0]["content"] = system_msg.strip()

        chat_history.append({"role": "user", "content": user_message})

        context = context.strip() if context else ""
        print(f"Processing single context with scaler: {context_scaler}")
        print(f"Bias scaler: {bias_scaler}")

        with torch.inference_mode(), torch.amp.autocast(str(device)):
            ctx_inputs = process_context(context)
            ctx_ids = ctx_inputs["ctx_ids"].to(device)
            ctx_attn_mask = ctx_inputs["ctx_attn_mask"].to(device)

            scalers_tensor = torch.tensor(
                [context_scaler], dtype=torch.float32, device=device
            )

            model_inputs = base_tokenizer.apply_chat_template(
                chat_history, return_tensors="pt", add_generation_prompt=True
            ).to(device)

            print(f"Context: {context}")
            print(f"Chat history: {chat_history}")

            outputs = modulated_model.generate(
                ctx_ids=ctx_ids,
                ctx_attn_mask=ctx_attn_mask,
                n_ctx_chunks=torch.tensor([len(ctx_ids)], device=ctx_ids.device),
                scalers=scalers_tensor,
                bias_scaler=bias_scaler,
                input_ids=model_inputs,
                max_new_tokens=192,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.1,
                no_repeat_ngram_size=4,
                use_cache=True,
                eos_token_id=base_tokenizer.eos_token_id,
                pad_token_id=base_tokenizer.pad_token_id
                or base_tokenizer.eos_token_id,
            )

            response = base_tokenizer.decode(
                outputs[0][model_inputs.shape[1] :], skip_special_tokens=True
            )

            chat_history.append({"role": "assistant", "content": response})

            words = response.split()
            partial_response = ""
            for word in words:
                partial_response += word + " "
                history[-1][1] = partial_response.strip()
                yield history

            history[-1][1] = response
            yield history

    except Exception as e:
        import traceback

        error_msg = f"❌ Error: {str(e)}"
        print(f"Error generating response: {str(e)}\n\n{traceback.format_exc()}")
        history[-1][1] = error_msg
        yield history


def reset_chat(system_msg: str):
    global chat_history
    chat_history = [
        {"role": "system", "content": system_msg.strip() if system_msg else ""}
    ]
    return [[None, WARNING_MESSAGE]], "Chat history reset successfully!"


custom_css = """
:root {
    color-scheme: light;
}

.gradio-container {
    font-family: 'Inter', sans-serif;
}

.chat-container {
    border-radius: 10px;
    border: 2px solid #d1d5db;
}

.context-field {
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    border: 2px solid #d1d5db;
}

.status-box {
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    border: 2px solid #e5e7eb;
}

.primary-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 6px;
    color: white;
    font-weight: 600;
}

.secondary-button {
    background-color: #607d8b;
    border: none;
    border-radius: 6px;
    color: white;
}

#chatbot {
    height: 500px;
}

.instruction-text {
    font-style: italic;
    color: #666;
    font-size: 0.9em;
    margin-bottom: 10px;
}

.warning-box {
    background-color: #fff3cd;
    border: 2px solid #ffc107;
    border-radius: 8px;
    padding: 12px;
    margin: 10px 0;
    color: #856404;
    font-size: 0.95em;
}

.warning-box strong {
    color: #d97706;
}

.disabled-overlay {
    background-color: #f5f5f5;
    border: 3px dashed #999;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    color: #999;
}

.chat-disabled-notice {
    border: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    color: #92400e;
    font-weight: 600;
    text-align: center;
    background-color: #fef3c7;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.internalization-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 15px;
    text-align: center;
    font-weight: 600;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    border: 3px solid #5568d3;
}

.internalization-banner h3 {
    margin: 0 0 10px 0;
    font-size: 1.2em;
    color: white;
}

.internalization-banner p {
    margin: 5px 0;
    font-size: 0.95em;
    font-weight: 400;
    color: white;
}

.context-section-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: 3px solid #5568d3;
    border-left: 6px solid #4c51bf;
    padding: 15px;
    margin-bottom: 15px;
    border-radius: 6px;
    color: white;
    box-shadow: 0 2px 6px rgba(102, 126, 234, 0.3);
}

.context-section-header strong,
.context-section-header small {
    color: white;
}

.chat-section-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: 3px solid #2563eb;
    border-left: 6px solid #1d4ed8;
    padding: 15px;
    margin-bottom: 15px;
    border-radius: 6px;
    color: white;
    box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);
}

.chat-section-header strong,
.chat-section-header small {
    color: white;
}

.panel-box {
    background-color: rgba(249, 250, 251, 0.5);
    border: 2px solid rgba(209, 213, 219, 0.5);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

.chat-panel-box {
    background-color: rgba(249, 250, 251, 0.5);
    border: 2px solid rgba(209, 213, 219, 0.5);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

#checkpoint-dropdown {
    position: relative;
    z-index: 20;
}

#checkpoint-dropdown [role="listbox"] {
    z-index: 9999 !important;
}

/* Dark mode support */
.dark .panel-box,
.dark .chat-panel-box {
    background-color: rgba(31, 41, 55, 0.5);
    border: 2px solid rgba(75, 85, 99, 0.5);
}

.dark .context-field,
.dark .chat-container {
    border-color: rgba(75, 85, 99, 0.5);
}

.dark .status-box {
    border-color: rgba(55, 65, 81, 0.5);
}
"""


def create_demo():
    with gr.Blocks(
        title="Doc-to-LoRA Chat Interface",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as demo:
        gr.Markdown(
            """
            # 📜 Doc-to-LoRA Chat Interface
            
            Load a hypernetwork checkpoint and chat with a context-modulated language model.
            Add one context with a scaling parameter to influence the model's responses.
            """
        )

        gr.HTML(
            """
            <div class="internalization-banner">
                <h3>🧠 How Context Internalization Works</h3>
                <p>📥 Contexts are processed by the hypernetworkto dynamically modulate the base model's parameters</p>
                <p>🚫 Contexts are NOT passed as text to the base model — they influence behavior internally</p>
                <p>💬 Only your chat messages (below) are sent to the language model</p>
            </div>
            """
        )

        gr.Markdown(
            """
            ### 📖 Usage Instructions
            
            1. **Load a Checkpoint**: Select a hypernetwork checkpoint from the dropdown and click "Load Checkpoint"
            2. **Configure Context**:
               - Enter your context information in the text field
               - Adjust the scaling slider to control context influence
            3. **Set Bias Scaler**: Adjust the bias scaler to control overall model behavior
            4. **Start Chatting**: Once the model is loaded, type your message and press Shift+Enter or click Send
            5. **Reset**: Use the "Reset Chat" button to start a new conversation
            
            💡 **Tip**: You can use context to provide background information or specific knowledge
            that should influence the model's responses.
            """
        )

        with gr.Row():
            with gr.Column(scale=1, elem_classes="panel-box"):
                gr.Markdown("### 📦 Load Checkpoint")
                gr.Markdown(
                    "*Select a trained hypernetwork checkpoint to begin.*",
                    elem_classes="instruction-text",
                )

                checkpoint_dropdown = gr.Dropdown(
                    choices=get_available_checkpoints(),
                    label="Select Checkpoint",
                    value=None,
                    interactive=True,
                    elem_id="checkpoint-dropdown",
                )

                load_btn = gr.Button("Load Checkpoint", variant="primary", size="lg")

                status_box = gr.Textbox(
                    label="Status",
                    lines=8,
                    interactive=False,
                    elem_classes="status-box",
                )

                gr.Markdown("---")

                gr.HTML(
                    """
                    <div class="context-section-header">
                        <strong>🧠 Context Internalization (Hypernetwork Input)</strong><br>
                        <small>This context modulates the model internally — it is NOT shown to the base model</small>
                    </div>
                    """
                )

                context = gr.Textbox(
                    label="🧠 Context (Internalized via Hypernetwork)",
                    placeholder="Enter context to be internalized by the hypernetwork...",
                    lines=4,
                    value=DEFAULT_CONTEXT,
                )
                context_scaler = gr.Slider(
                    minimum=-2.0,
                    maximum=2.0,
                    step=0.01,
                    value=1.0,
                    label="Context Scaling",
                )

                gr.Markdown("---")

                bias_scaler = gr.Slider(
                    minimum=-2.0,
                    maximum=2.0,
                    step=0.01,
                    value=1.0,
                    label="Bias Scaler",
                    info="A single scalar applied to bias parameters (independent of contexts)",
                )

            with gr.Column(scale=2, elem_classes="chat-panel-box"):
                gr.HTML(
                    """
                    <div class="chat-section-header">
                        <strong>💬 Chat Interface (Direct Input to Base Model)</strong><br>
                        <small>Your messages here are the ONLY text the base model sees — contexts above influence it internally</small>
                    </div>
                    """
                )

                chat_status_notice = gr.HTML(
                    """
                    <div class="chat-disabled-notice">
                        🔒 <strong>Chat Disabled:</strong> Please load a checkpoint first to enable chat functionality.
                    </div>
                    """,
                    visible=True,
                )

                system_msg = gr.Textbox(
                    label="System Message (Optional - Sent to Base Model)",
                    placeholder="Load a checkpoint to enable chat...",
                    lines=2,
                    interactive=False,
                )

                chatbot = gr.Chatbot(
                    label="Conversation",
                    show_copy_button=True,
                    height=500,
                    elem_id="chatbot",
                    elem_classes="chat-container",
                    value=[
                        [
                            None,
                            "🔒 Chat is currently disabled. Please load a checkpoint from the left panel to begin chatting.",
                        ]
                    ],
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Message (Sent Directly to Base Model)",
                        placeholder="⚠️ Load a checkpoint first to start chatting...",
                        lines=2,
                        scale=4,
                        interactive=False,
                    )
                    send_btn = gr.Button(
                        "🔒 Send (Disabled)",
                        variant="primary",
                        scale=1,
                        interactive=False,
                    )

                with gr.Row():
                    clear_btn = gr.Button(
                        "🔒 Reset Chat (Disabled)",
                        variant="secondary",
                        interactive=False,
                    )
                    reset_status = gr.Textbox(label="Reset Status", visible=False)

        load_btn.click(
            fn=load_checkpoint,
            inputs=[checkpoint_dropdown],
            outputs=[status_box, msg, send_btn, system_msg, clear_btn],
        ).then(
            fn=lambda: (
                gr.update(visible=False),
                gr.update(
                    placeholder="Type your message here... (Shift+Enter for new line)"
                ),
                gr.update(value="Send"),
                gr.update(value="🔄 Reset Chat"),
                gr.update(value=[[None, WARNING_MESSAGE]]),
            ),
            outputs=[chat_status_notice, msg, send_btn, clear_btn, chatbot],
        )

        msg.submit(
            fn=add_user_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
        ).then(
            fn=generate_response,
            inputs=[
                chatbot,
                system_msg,
                context,
                context_scaler,
                bias_scaler,
            ],
            outputs=[chatbot],
        )

        send_btn.click(
            fn=add_user_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
        ).then(
            fn=generate_response,
            inputs=[
                chatbot,
                system_msg,
                context,
                context_scaler,
                bias_scaler,
            ],
            outputs=[chatbot],
        )

        clear_btn.click(
            fn=reset_chat,
            inputs=[system_msg],
            outputs=[chatbot, reset_status],
        )

        gr.Markdown(f"---\n{FOOTER.strip()}")
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=True,
    )
