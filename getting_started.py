# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("/lpai/models/openvla__openvla-7b/24-09-16-2142", trust_remote_code=True) #AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "/lpai/models/openvla__openvla-7b/24-09-16-2142", #"openvla/openvla-7b", 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:1")

print("Load success")

# Grab image input & format prompt
# image: Image.Image = get_from_camera(...)

# Grab image input & format prompt
try:
    image = Image.open("robot.jpg")
except FileNotFoundError:
    print("未找到 robot.jpg 文件，请检查文件是否存在于当前目录。")
    exit(1)

prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:1", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(f"Prompt: {prompt}")
print(f"ACTION: {action}")
print("Test Program END")

# Execute...
# robot.act(action, ...)
