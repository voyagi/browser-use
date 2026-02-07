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


def build_sample_images(image_path: str) -> list[dict]:
    """Convert an uploaded image to the format Browser Use expects."""
    b64 = encode_image_to_base64(image_path)
    ext = image_path.rsplit(".", 1)[-1].lower()
    media_types = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
    media_type = media_types.get(ext, "image/png")
    return [
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
    ]


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


async def run_agent_task(task: str, image_path: str | None, headless: bool):
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

    if image_path:
        agent_kwargs["sample_images"] = build_sample_images(image_path)

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


def respond(message: str, image, chat_history: list, headless: bool):
    """Handle user message — run the agent and stream progress."""
    if not message.strip():
        return chat_history, "", None

    image_path = image if isinstance(image, str) else None

    # Add user message to history
    user_msg = message
    if image_path:
        user_msg += "\n\n(Image attached)"
    chat_history = chat_history + [{"role": "user", "content": user_msg}]

    # Add a placeholder for assistant response
    chat_history = chat_history + [{"role": "assistant", "content": "Starting agent..."}]
    yield chat_history, "", None

    # Run the agent in a background thread so we can stream updates
    loop = asyncio.new_event_loop()
    result_holder = [None]

    def run_in_thread():
        result_holder[0] = loop.run_until_complete(
            run_agent_task(message, image_path, headless)
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
            yield chat_history, "", None

    # Final result
    progress = "\n\n".join(step_callback.messages) if step_callback.messages else ""
    separator = "\n\n---\n\n" if progress else ""
    final_text = result_holder[0] or "No result."
    chat_history[-1] = {"role": "assistant", "content": f"{progress}{separator}**Result:** {final_text}"}
    yield chat_history, "", None


EXAMPLES = [
    "Go to google.com and search for 'best CRM for small business 2026'. List the top 5 results.",
    "Go to linkedin.com and find 3 companies hiring for B2B sales in Amsterdam.",
    "Go to amazon.com and find the cheapest wireless mouse with 4+ star rating.",
    "Go to news.ycombinator.com and tell me the top 3 stories right now.",
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
                placeholder="Type a task and press Enter...",
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="e.g. Go to google.com and find the best rated coffee machine under $100",
                    show_label=False,
                    scale=4,
                    container=False,
                )
                stop_btn = gr.Button("Stop", variant="stop", scale=1, min_width=80)

            image_input = gr.Image(
                label="Attach image (optional — e.g. screenshot of what you want the AI to interact with)",
                type="filepath",
                height=100,
            )

        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### Settings")
            headless = gr.Checkbox(label="Run browser in background", value=False)

            gr.Markdown("### Quick Tasks")
            for ex in EXAMPLES:
                short_label = ex[:60] + "..." if len(ex) > 60 else ex
                btn = gr.Button(short_label, size="sm")
                btn.click(set_example, inputs=[gr.State(ex)], outputs=[msg])

    # Wire events
    msg.submit(
        respond,
        inputs=[msg, image_input, chatbot, headless],
        outputs=[chatbot, msg, image_input],
    )
    stop_btn.click(stop_agent, outputs=[stop_btn])


if __name__ == "__main__":
    app.launch(
        inbrowser=True,
        css="""
        footer { display: none !important; }
        """,
    )
