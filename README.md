# CFG fade #
### extension for Forge webui for Stable Diffusion ###
---
## Basic usage ##
A variety of ways to adjust CFG dynamically.
Settings used are saved with metadata, and restored from loading through the **PNG Info** tab.

---
## Advanced / Details ##
Steps to start shuffling are selected as a proportion of the way through the diffusion process: 0.0 is the first step, 1.0 is last, 1.01 is never. 
Each step from that point will receive a new shuffle.


---
## To do? ##
1. perp_neg? The [Neutral Prompt extension](https://github.com/ljleb/sd-webui-neutral-prompt) is already aiming to cover this
2. slew limiting: not convinced by this. Seems better overall to limit change using the scheduler, though the effects are different.
3. different tonemappers
---
## License ##
Public domain. Unlicense. Free to a good home.
All terrible code is my own. Use at your own risk, read the code.

---
## Credits ##
Thanks to Alex "mcmonkey" Goodwin for the Dynamic Thresholding extension (Forge built-in version). I started this project with zero knowledge, and this source got me started. The first - basic, unreleased - version was essentially hacked out of that extension.
Also thanks to https://github.com/Haoming02. I learned a lot about how to implement this from their work.
rescaleCFG and Reinhard tonemapping based on https://github.com/comfyanonymous/ComfyUI_experiments/

[combating-mean-drift-in-cfg (birchlabs.co.uk)](https://birchlabs.co.uk/machine-learning#combating-mean-drift-in-cfg)

---


> Written with [StackEdit](https://stackedit.io/).
