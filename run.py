import asyncio
import sys
import io
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

load_dotenv()

from browser_use import Agent, ChatAnthropic

async def main():
    llm = ChatAnthropic(model="claude-sonnet-4-0", temperature=0.0)

    task = input("\nWhat do you want the AI to do?\n> ")

    if not task.strip():
        print("No task provided. Exiting.")
        return

    print(f"\nStarting agent on task: {task}\n")

    agent = Agent(task=task, llm=llm)
    result = await agent.run()

    # Extract the final answer
    final = result.final_result()
    print(f"\n{'='*50}")
    print(f"DONE: {final}")
    print(f"{'='*50}")

if __name__ == "__main__":
    asyncio.run(main())
