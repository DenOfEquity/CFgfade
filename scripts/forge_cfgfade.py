import gradio as gr

from modules import scripts
import modules.shared as shared
from modules.prompt_parser import SdConditioning
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
import torch, math
import torchvision.transforms.functional as TF

##  hacked out of:
######################### DynThresh Core #########################

#   button to spit weighted cfg to console, better: gradio lineplot for display of weights



class CFGfadeForge(scripts.Script):
    def __init__(self):
        self.actualCFG = 0          #   dummy value
        self.weight = 1.0
        self.lowSigma = 0.0
        self.boostStep = 0.0
        self.highStep = 0.5
        self.maxScale = 1.0
        self.fadeStep = 0.5
        self.zeroStep = 1.0
        self.minScale = 0.0
        self.reinhard = 1.0
        self.rfcgmult = 1.0
        self.antidrfS = 0.0
        self.antidrfE = 1.0


    def title(self):
        return "CFG fade"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Row():
                enabled = gr.Checkbox(value=False, label='Enable modifications to CFG')
            with gr.Row(equalWidth=True):
                reinhard = gr.Slider(minimum=0.0, maximum=16.0, step=0.5, value=0.0, label='Reinhard target CFG')
                rcfgmult = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label='RescaleCFG multiplier')
                lowSigma = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=0.0, label='clamp CFG @ sigma')
            with gr.Row():
                boostStep = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label='CFG boost start')
                highStep = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label='boost end')
                maxScale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, value=2.0, label='maximum weight')
            with gr.Row():
                fadeStep = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label='CFG fade start')
                zeroStep = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label='fade end')
                minScale = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label='minimum weight')
            with gr.Row(equalWidth=True):
                antidrfS = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label='start anti drift of latents')
                antidrfE = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label='end anti drift')


        self.infotext_fields = [
            (enabled, lambda d: enabled.update(value=("cfgfade_enabled" in d))),
            (boostStep, "cfgfade_boostStep"),
            (highStep, "cfgfade_highStep"),
            (maxScale, "cfgfade_maxScale"),
            (fadeStep, "cfgfade_fadeStep"),
            (zeroStep, "cfgfade_zeroStep"),
            (minScale, "cfgfade_minScale"),
            (lowSigma, "cfgfade_lowSigma"),
            (reinhard, "cfgfade_reinhard"),
            (rcfgmult, "cfgfade_rcfgmult"),
            (antidrfS, "cfgfade_antidrfS"),
            (antidrfE, "cfgfade_antidrfE"),
        ]

        return enabled, boostStep, highStep, maxScale, fadeStep, zeroStep, minScale, lowSigma, reinhard, rcfgmult, antidrfS, antidrfE


    def patch(self, model):
        self.previousStep = None
        self.limit = 1.6        #make adjustable
        self.blur = 5           #make adjustable

        def sampler_cfgfade(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]

            cond_scale *= self.weight

            if cond_scale < 1.0:
                return cond

            thisStep = shared.state.sampling_step
            lastStep = shared.state.sampling_steps

#   perp-neg here?

#   reinhard tonemap from comfy
#   changes too much? - unpredictable difference when enabled, so useless?
            noisePrediction = cond - uncond
            if self.reinhard != 0.0 and self.reinhard != cond_scale:
                multiplier = 1.0 / cond_scale * self.reinhard
                noise_pred_vector_magnitude = (torch.linalg.vector_norm(noisePrediction, dim=(1)) + 0.0000000001)[:,None]
                noisePrediction /= noise_pred_vector_magnitude

                mean = torch.mean(noise_pred_vector_magnitude, dim=(1,2,3), keepdim=True)
                std = torch.std(noise_pred_vector_magnitude, dim=(1,2,3), keepdim=True)
                top = (std * 3 + mean) * multiplier

                #reinhard
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

                x_rescaled = result * (ro_pos / ro_cfg)
                result = torch.lerp (result, x_rescaled, self.rcfgmult)
                del x_rescaled
#   end: rescaleCFG
            del noisePrediction



#subtract mean of result - seems like a free win
            if thisStep >= self.antidrfS * lastStep and thisStep < self.antidrfE * lastStep:
                result -= result.mean(dim=(1, 2, 3), keepdim=True)

            return result

#slew limiting, to restrict change from step to step? not same as reducing max sigma

            if thisStep > 3 and self.previousStep != None:
                rc = result
                diff = result - self.previousStep
                result = diff.clamp(-1 * self.limit, self.limit) + self.previousStep
                del diff
                if self.blur > 1:
                    result = TF.gaussian_blur(result, self.blur)
                    result_clean_hi = rc - TF.gaussian_blur(rc, self.blur)
                    result = result + result_clean_hi
                    del result_clean_hi


            self.previousStep = result
            return result

   

        m = model.clone()
        m.set_model_sampler_cfg_function(sampler_cfgfade)
        return (m, )


    def denoiser_callback(self, params):
        lastStep = params.total_sampling_steps
        thisStep = params.sampling_step
        sigma = params.sigma
        
        highStep = self.highStep * lastStep
        boostStep = self.boostStep * lastStep
        fadeStep = self.fadeStep * lastStep
        zeroStep = self.zeroStep * lastStep

        if thisStep > highStep:
            boostWeight = self.maxScale
        elif thisStep <= boostStep:
            boostWeight = 1.0
        else:
            boostWeight = 1.0 + (self.maxScale - 1.0) * ((thisStep - boostStep) / (highStep - boostStep))


        if sigma < self.lowSigma:
            fadeWeight = self.minScale
        else:
            if thisStep >= zeroStep:
                fadeWeight = 0.0
            elif thisStep < fadeStep:
                fadeWeight = 1.0
            else:
                fadeWeight = 1.0 - (thisStep - fadeStep) / (zeroStep  - fadeStep)

            # at this point, weight is in the range 0.0->1.0
            fadeWeight *= (1.0 - self.minScale)
            fadeWeight += self.minScale
            # now it is minimum->1.0

        self.weight = boostWeight * fadeWeight

#   don't have access to change cfg_scale at this point?

    def process_before_every_sampling(self, params, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        enabled, boostStep, highStep, maxScale, fadeStep, zeroStep, minScale, lowSigma, reinhard, rcfgmult, antidrfS, antidrfE = script_args

        if not enabled:
            return

        self.actualCFG = params.cfg_scale

        self.boostStep = boostStep
        self.highStep = highStep
        self.maxScale = maxScale
        self.fadeStep = fadeStep
        self.zeroStep = zeroStep
        self.minScale = minScale
        self.lowSigma = lowSigma
        self.reinhard = reinhard
        self.rcfgmult = rcfgmult
        self.antidrfS = antidrfS
        self.antidrfE = antidrfE

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        params.extra_generation_params.update(dict(
            cfgfade_enabled = enabled,
            cfgfade_boostStep = boostStep,
            cfgfade_highStep = highStep,
            cfgfade_maxScale = maxScale,
            cfgfade_fadeStep = fadeStep,
            cfgfade_zeroStep = zeroStep,
            cfgfade_minScale = minScale,
            cfgfade_lowSigma = lowSigma,
            cfgfade_reinhard = reinhard,
            cfgfade_rcfgmult = rcfgmult,
            cfgfade_antidrfS = antidrfS,
            cfgfade_antidrfE = antidrfE,
        ))

        #   must log the parameters before fixing minScale
        self.minScale /= self.maxScale

        on_cfg_denoiser(self.denoiser_callback)

        unet = params.sd_model.forge_objects.unet
        unet = CFGfadeForge.patch(self, unet)[0]
        params.sd_model.forge_objects.unet = unet

        return

    def postprocess(self, params, processed, *args):
        self.weight = 1.0

        remove_current_script_callbacks()

