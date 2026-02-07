import asyncio
import sys
import io
from dotenv import load_dotenv

# Fix Windows console encoding for emoji output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

load_dotenv()

from browser_use import Agent, ChatAnthropic

async def main():
    llm = ChatAnthropic(model="claude-sonnet-4-0", temperature=0.0)

    agent = Agent(
        task="Go to google.com and search for 'browser use python automation'. Return the title of the first search result.",
        llm=llm,
    )

    result = await agent.run()
    print("\n=== RESULT ===")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
