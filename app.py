import asyncio
import base64
import os
import sys
import io
import threading

# pythonw.exe sets stdout/stderr to None — redirect to devnull so logging doesn't crash
if sys.stdout is None or not hasattr(sys.stdout, "buffer"):
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from browser_use import Agent, ChatAnthropic
from browser_use.browser.profile import BrowserProfile


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def build_sample_images(image_paths: list[str]) -> list[dict]:
    """Convert uploaded images to the format Browser Use expects."""
    media_types = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
    images = []
    for path in image_paths:
        b64 = encode_image_to_base64(path)
        ext = path.rsplit(".", 1)[-1].lower()
        media_type = media_types.get(ext, "image/png")
        images.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}})
    return images


# Shared state for the running agent
current_agent = None
should_stop = False


async def check_should_stop() -> bool:
    return should_stop


def step_callback(browser_state, agent_output, step_number):
    """Called after each agent step — collects progress messages."""
    parts = []
    if agent_output.evaluation_previous_goal:
        parts.append(agent_output.evaluation_previous_goal)
    if agent_output.next_goal:
        parts.append(f"Next: {agent_output.next_goal}")
    if parts:
        msg = f"**Step {step_number}** — " + " | ".join(parts)
        step_callback.messages.append(msg)


step_callback.messages = []


async def run_agent_task(task: str, image_paths: list[str], headless: bool):
    """Run a Browser Use agent and yield progress updates."""
    global current_agent, should_stop
    should_stop = False
    step_callback.messages = []

    llm = ChatAnthropic(model="claude-sonnet-4-0", temperature=0.0)

    agent_kwargs = {
        "task": task,
        "llm": llm,
        "register_new_step_callback": step_callback,
        "register_should_stop_callback": check_should_stop,
        "browser_profile": BrowserProfile(headless=headless),
    }

    if image_paths:
        agent_kwargs["sample_images"] = build_sample_images(image_paths)

    current_agent = Agent(**agent_kwargs)

    try:
        result = await current_agent.run(max_steps=50)
        final = result.final_result()
        return final if final else "Task completed (no text result returned)."
    except Exception as e:
        return f"Error: {e}"
    finally:
        current_agent = None


def stop_agent():
    global should_stop, current_agent
    should_stop = True
    if current_agent:
        current_agent.stop()
    return gr.update(interactive=False, value="Stopping...")


def respond(message: str, files, chat_history: list, headless: bool):
    """Handle user message — run the agent and stream progress."""
    stop_reset = gr.update(value="Stop", interactive=True)

    if not message.strip():
        return chat_history, "", None, stop_reset

    # gr.File returns a list of file paths (or None)
    image_paths = [f.name if hasattr(f, "name") else f for f in files] if files else []

    # Add user message to history
    user_msg = message
    if image_paths:
        user_msg += f"\n\n({len(image_paths)} image{'s' if len(image_paths) != 1 else ''} attached)"
    chat_history = chat_history + [{"role": "user", "content": user_msg}]

    # Add a placeholder for assistant response
    chat_history = chat_history + [{"role": "assistant", "content": "Starting agent..."}]
    yield chat_history, "", None, gr.update()

    # Run the agent in a background thread so we can stream updates
    loop = asyncio.new_event_loop()
    result_holder = [None]

    def run_in_thread():
        result_holder[0] = loop.run_until_complete(
            run_agent_task(message, image_paths, headless)
        )

    thread = threading.Thread(target=run_in_thread)
    thread.start()

    # Stream progress while agent runs
    seen = 0
    while thread.is_alive():
        thread.join(timeout=1.0)
        if len(step_callback.messages) > seen:
            progress = "\n\n".join(step_callback.messages)
            chat_history[-1] = {"role": "assistant", "content": progress + "\n\n_Working..._"}
            seen = len(step_callback.messages)
            yield chat_history, "", None, gr.update()

    # Final result — reset the Stop button
    progress = "\n\n".join(step_callback.messages) if step_callback.messages else ""
    separator = "\n\n---\n\n" if progress else ""
    final_text = result_holder[0] or "No result."
    chat_history[-1] = {"role": "assistant", "content": f"{progress}{separator}**Result:** {final_text}"}
    yield chat_history, "", None, stop_reset


EXAMPLES = [
    ("Search top CRMs", "Go to google.com and search for 'best CRM for small business 2026'. List the top 5 results."),
    ("B2B sales jobs", "Go to linkedin.com and find 3 companies hiring for B2B sales in Amsterdam."),
    ("Cheap wireless mouse", "Go to amazon.com and find the cheapest wireless mouse with 4+ star rating."),
    ("Hacker News top 3", "Go to news.ycombinator.com and tell me the top 3 stories right now."),
]


def set_example(example_text):
    return example_text


# Build the Gradio UI
with gr.Blocks(title="Browser Use AI Agent") as app:
    gr.Markdown("# Browser Use AI Agent\nTell the AI what to do — it will open a browser and do it for you.")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                placeholder="Type a task and press Enter...",
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="e.g. Go to google.com and find the best rated coffee machine under $100",
                    show_label=False,
                    scale=6,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)
                stop_btn = gr.Button("Stop", variant="stop", scale=1, min_width=80)

            image_input = gr.File(
                label="Drop images here or click to upload",
                file_count="multiple",
                file_types=["image"],
                height=120,
                elem_id="image-drop",
            )

        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### Settings")
            headless = gr.Checkbox(label="Run browser in background", value=False)

            gr.Markdown("### Quick Tasks")
            for label, task in EXAMPLES:
                btn = gr.Button(label, size="sm")
                btn.click(set_example, inputs=[gr.State(task)], outputs=[msg])

    # Wire events
    submit_args = dict(
        fn=respond,
        inputs=[msg, image_input, chatbot, headless],
        outputs=[chatbot, msg, image_input, stop_btn],
    )
    msg.submit(**submit_args)
    send_btn.click(**submit_args)
    stop_btn.click(stop_agent, outputs=[stop_btn])


if __name__ == "__main__":
    app.launch(
        inbrowser=True,
        css="""
        footer { display: none !important; }
        .chatbot-container .label-wrap { display: none !important; }
        #image-drop {
            border: 2px dashed #666 !important;
            border-radius: 10px !important;
            background: rgba(255,255,255,0.03) !important;
            transition: border-color 0.2s, background 0.2s;
        }
        #image-drop:hover {
            border-color: #4a9eff !important;
            background: rgba(74,158,255,0.06) !important;
        }
        """,
    )
