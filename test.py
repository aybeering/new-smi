from src.sim import Sim

text, idx_path = Sim.test(
    "Darren White wins Albuquerque mayoral election by November 1, 2025?",
    "/Users/ayang/agent/new-smi/data/title_metadata.json",
    model_path="dir",  # 可省略，默认用 metadata 里的 model_id 或 dir
)
print(text)