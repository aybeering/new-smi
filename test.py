from src.sim import Sim

result = Sim.agent(
    "Freddie Mac market capitalization between $220B and $300B at IPO close by December 31, 2025?",
    "/Users/ayang/agent/new-smi/data/title_metadata.json",
    model_path="dir",  # 可省略，默认用 metadata 里的 model_id 或 dir
)
print(result)
