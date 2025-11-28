from src.sim import Sim

result = Sim.agent(
    "Egg price < $3.07 by September 30, 2025?",
    "/Users/ayang/agent/new-smi/data/title_metadata.json",
    model_path="dir",  # 可省略，默认用 metadata 里的 model_id 或 dir
)
print("进行时间测试",result)
