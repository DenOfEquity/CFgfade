import gradio as gr

from modules import scripts
import modules.shared as shared
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
import torch, math

#import torchvision.transforms.functional as TF

import ldm_patched.modules.samplers as LDM
import modules_forge.forge_sampler

#   button to spit weighted cfg to console, better: gradio lineplot for display of weights



class CFGfadeForge(scripts.Script):
    weight = 1.0
    backup_sampling_function = None

    def __init__(self):
        self.boostStep = 0.0
        self.highStep = 0.5
        self.maxScale = 1.0
        self.fadeStep = 0.5
        self.zeroStep = 1.0
        self.minScale = 0.0
        self.reinhard = 1.0
        self.rfcgmult = 1.0
        self.centreMean = False
        self.heuristic = 0
        self.hStart = 0.0

    def title(self):
        return "CFG fade"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Row():
                enabled    = gr.Checkbox(value=False, label='Enable modifications to CFG')
                cntrMean   = gr.Checkbox(value=False, label='centre conds to mean')
#                scaleCFGs  = gr.Checkbox(value=False, label='scale hCFG and rCFG')
            with gr.Row():
                lowCFG1   = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.1, label='CFG 1 until step')
                maxScale  = gr.Slider(minimum=1.0, maximum=4.0,  step=0.01, value=1.0, label='boost factor')
            with gr.Row():
                boostStep = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.2, label='CFG boost start step')
                minScale  = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=1.0, label='fade factor')
            with gr.Row():
                highStep  = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.4, label='full boost at step')
                heuristic = gr.Slider(minimum=0.0, maximum=16.0, step=0.5,  value=0,   label='Heuristic CFG')
            with gr.Row():
                fadeStep  = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.5, label='CFG fade start step')
                hStart    = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.0, label='... start step')
            with gr.Row():
                zeroStep  = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.7, label='full fade at step')
                reinhard  = gr.Slider(minimum=0.0, maximum=16.0, step=0.5,  value=0.0, label='Reinhard CFG')
            with gr.Row():
                highCFG1  = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.8, label='CFG 1 after step')
                rcfgmult  = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.0, label='Rescale CFG')

        self.infotext_fields = [
            (enabled, lambda d: enabled.update(value=("cfgfade_enabled" in d))),
            (cntrMean,  "cfgfade_cntrMean"),
            (boostStep, "cfgfade_boostStep"),
            (highStep,  "cfgfade_highStep"),
            (maxScale,  "cfgfade_maxScale"),
            (fadeStep,  "cfgfade_fadeStep"),
            (zeroStep,  "cfgfade_zeroStep"),
            (minScale,  "cfgfade_minScale"),
            (lowCFG1,   "cfgfade_lowCFG1"),
            (highCFG1,  "cfgfade_highCFG1"),
            (reinhard,  "cfgfade_reinhard"),
            (rcfgmult,  "cfgfade_rcfgmult"),
            (heuristic, "cfgfade_heuristic"),
            (hStart,    "cfgfade_hStart"),
        ]

        return enabled, cntrMean, boostStep, highStep, maxScale, fadeStep, zeroStep, minScale, lowCFG1, highCFG1, reinhard, rcfgmult, heuristic, hStart

#   edited from ldm_patched/modules/samplers to add cond_scaling (initial 3 lines)
    def sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
            cond_scale *= CFGfadeForge.weight
            if cond_scale < 1.0:
                cond_scale = 1.0

            edit_strength = sum((item['strength'] if 'strength' in item else 1) for item in cond)

            if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
                uncond_ = None
            else:
                uncond_ = uncond

            for fn in model_options.get("sampler_pre_cfg_function", []):
                model, cond, uncond_, x, timestep, model_options = fn(model, cond, uncond_, x, timestep, model_options)

            cond_pred, uncond_pred = LDM.calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options)

            if "sampler_cfg_function" in model_options:
                args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                        "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
                cfg_result = x - model_options["sampler_cfg_function"](args)
            elif not math.isclose(edit_strength, 1.0):
                cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale * edit_strength
            else:
                cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

            for fn in model_options.get("sampler_post_cfg_function", []):
                args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                        "sigma": timestep, "model_options": model_options, "input": x}
                cfg_result = fn(args)

            return cfg_result

    def patch(self, model):
#        sigmin = model.model.model_sampling.sigma(model.model.model_sampling.timestep(model.model.model_sampling.sigma_min))
#        sigmax = model.model.model_sampling.sigma(model.model.model_sampling.timestep(model.model.model_sampling.sigma_max))

        def sampler_cfgfade(args):
            cond = args["cond"]
            cond_scale = args["cond_scale"]

            if cond_scale == 1.0:
                return cond
            else:
                uncond = args["uncond"]

#   sometimes this scaling seems like a win, but only when heuristic/reinhard CFG is too high
#            if self.scaleCFGs == True:
#                heuristic = max(1.0, self.heuristic * CFGfadeForge.weight) if (self.heuristic > 0.0) else 0.0
#                reinhard  = max(1.0, self.reinhard  * CFGfadeForge.weight) if (self.reinhard  > 0.0) else 0.0
#            else:
            heuristic = self.heuristic
            reinhard = self.reinhard

            if self.centreMean == True:     # better after, but value here too?
                for b in range(len(cond)):
                    for c in range(4):
                        cond[b][c] -= cond[b][c].mean()
                        uncond[b][c] -= uncond[b][c].mean()

#   cond_scale weighting now applied in sampling_function, can avoid processing of uncond for performance increase

            thisStep = shared.state.sampling_step
            lastStep = shared.state.sampling_steps

#   heuristic scaling, higher hcfg acts to boost contrast/detail/sharpness; low reduces; quantile has effect, but not significant for quality IMO
            noisePrediction = cond - uncond
            if heuristic != 0.0 and heuristic != cond_scale and thisStep >= self.hStart * lastStep:
                base = uncond + cond_scale * noisePrediction
                heur = uncond + heuristic * noisePrediction

                #   center both on zero
                baseC = base - base.mean()
                heurC = heur - heur.mean()
                del base, heur

                #   calc 99.0% quartiles - doesn't seem to have value as an option
                baseQ = torch.quantile(baseC.abs(), 0.99)
                heurQ = torch.quantile(heurC.abs(), 0.99)
                del baseC, heurC

                if baseQ != 0.0 and heurQ != 0.0:
                    cond *= (baseQ / heurQ)
                    uncond *= (baseQ / heurQ)
#   end: heuristic scaling

                
#   reinhard tonemap from comfy
            noisePrediction = cond - uncond
            if reinhard != 0.0 and reinhard != cond_scale:
                multiplier = 1.0 / cond_scale * reinhard
                noise_pred_vector_magnitude = (torch.linalg.vector_norm(noisePrediction, dim=(1)) + 0.0000000001)[:,None]
                noisePrediction /= noise_pred_vector_magnitude

                mean = torch.mean(noise_pred_vector_magnitude, dim=(1,2,3), keepdim=True)
                std = torch.std(noise_pred_vector_magnitude, dim=(1,2,3), keepdim=True)
                top = (std * 3 + mean) * multiplier

                noise_pred_vector_magnitude *= (1.0 / top)
                new_magnitude = noise_pred_vector_magnitude / (noise_pred_vector_magnitude + 1.0)
                new_magnitude *= top
                cond_scale *= new_magnitude
#   end: reinhard


#   rescaleCFG - maybe should be exclusive of other effects, but why restrict?
            result = uncond + cond_scale * noisePrediction
            if self.rcfgmult != 0.0:
                ro_pos = torch.std(cond, dim=(1,2,3), keepdim=True)
                ro_cfg = torch.std(result, dim=(1,2,3), keepdim=True)

                if ro_pos != 0.0 and ro_cfg != 0.0:
                    x_rescaled = result * (ro_pos / ro_cfg)
                    result = torch.lerp (result, x_rescaled, self.rcfgmult)
                    del x_rescaled                
#   end: rescaleCFG
            del noisePrediction

            return result

        m = model.clone()
        m.set_model_sampler_cfg_function(sampler_cfgfade)
        return (m, )


    def denoiser_callback(self, params):
        lastStep = params.total_sampling_steps - 1
        thisStep = params.sampling_step
        sigma = params.sigma[0]

        lowCFG1   = self.lowCFG1   * lastStep
        highStep  = self.highStep  * lastStep
        boostStep = self.boostStep * lastStep
        highCFG1  = self.highCFG1  * lastStep
        fadeStep  = self.fadeStep  * lastStep
        zeroStep  = self.zeroStep  * lastStep

        if thisStep < lowCFG1:
            boostWeight = 0.0
        elif thisStep < boostStep:
            boostWeight = 1.0
        elif thisStep < highStep:
            boostWeight = 1.0 + (self.maxScale - 1.0) * ((thisStep - boostStep) / (highStep - boostStep))
        else:
            boostWeight = self.maxScale

        if thisStep > highCFG1:
            fadeWeight = 0.0
        else:
            if thisStep < fadeStep:
                fadeWeight = 1.0
            elif thisStep < zeroStep:
                fadeWeight = 1.0 - (thisStep - fadeStep) / (zeroStep  - fadeStep)
            else:
                fadeWeight = 0.0

            # at this point, weight is in the range 0.0->1.0
            fadeWeight *= (1.0 - self.minScale)
            fadeWeight += self.minScale
            # now it is minimum->1.0

        CFGfadeForge.weight = boostWeight * fadeWeight


    def process_before_every_sampling(self, params, *script_args, **kwargs):
        enabled = script_args[0]
        if enabled:
            unet = params.sd_model.forge_objects.unet
            unet = CFGfadeForge.patch(self, unet)[0]
            params.sd_model.forge_objects.unet = unet

        return

    def process(self, params, *script_args, **kwargs):
        enabled, cntrMean, boostStep, highStep, maxScale, fadeStep, zeroStep, minScale, lowCFG1, highCFG1, reinhard, rcfgmult, heuristic, hStart = script_args

        if not enabled:
            return

        self.centreMean = cntrMean
        self.boostStep  = boostStep
        self.highStep   = highStep
        self.maxScale   = maxScale
        self.fadeStep   = fadeStep
        self.zeroStep   = zeroStep
        self.minScale   = minScale
        self.lowCFG1    = lowCFG1
        self.highCFG1   = highCFG1
        self.reinhard   = reinhard
        self.rcfgmult   = rcfgmult
        self.heuristic  = heuristic
        self.hStart     = hStart

        # logs, could save boost start/full only if boost factor > 1
        #       similar for fade
        params.extra_generation_params.update(dict(
            cfgfade_enabled   = enabled,
            cfgfade_cntrMean  = cntrMean,
            cfgfade_boostStep = boostStep,
            cfgfade_highStep  = highStep,
            cfgfade_maxScale  = maxScale,
            cfgfade_fadeStep  = fadeStep,
            cfgfade_zeroStep  = zeroStep,
            cfgfade_minScale  = minScale,
            cfgfade_lowCFG1   = lowCFG1,
            cfgfade_highCFG1  = highCFG1,
            cfgfade_reinhard  = reinhard,
            cfgfade_rcfgmult  = rcfgmult,
            cfgfade_heuristic = heuristic,
            cfgfade_hStart    = hStart,
        ))

        #   must log the parameters before fixing minScale
        self.minScale /= self.maxScale

        on_cfg_denoiser(self.denoiser_callback)

        if CFGfadeForge.backup_sampling_function == None:
            CFGfadeForge.backup_sampling_function = modules_forge.forge_sampler.sampling_function

        modules_forge.forge_sampler.sampling_function = CFGfadeForge.sampling_function
        return

    def postprocess(self, params, processed, *args):
        enabled = args[0]
        if enabled:
            if CFGfadeForge.backup_sampling_function != None:
                modules_forge.forge_sampler.sampling_function = CFGfadeForge.backup_sampling_function

        remove_current_script_callbacks()
        return

