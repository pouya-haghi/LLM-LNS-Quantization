import math
import torch
import torch.nn.functional as F
import transformers
import peft
from peft import __version__ as PEFT_VERSION
from pathlib import Path
from typing import List, Mapping, NewType, Optional, Tuple, Union
from tqdm import tqdm

from transformers import BatchEncoding

from lm_eval import utils
from lm_eval.base import BaseLM

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
from transformers import BitsAndBytesConfig

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

_DeviceMapping = NewType("DeviceMapping", Mapping[str, Union[int, str, torch.device]])


def _get_accelerate_args(
    low_cpu_mem_usage: Optional[bool] = True,
    device_map_option: Optional[str] = "auto",
    max_memory_per_gpu: Optional[Union[int, str]] = None,
    max_cpu_memory: Optional[Union[int, str]] = None,
    offload_folder: Optional[str] = "./offload",
) -> dict:
    """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
    max_memory = {}
    if max_memory_per_gpu is not None:
        max_memory_per_gpu_map = {
            device_idx: max_memory_per_gpu
            for device_idx in range(torch.cuda.device_count())
        }
        max_memory.update(max_memory_per_gpu_map)
    if max_cpu_memory is not None:
        max_memory["cpu"] = max_cpu_memory

    args = {}
    if max_memory:
        args["max_memory"] = max_memory
    args["low_cpu_mem_usage"] = low_cpu_mem_usage
    args["device_map"] = device_map_option
    args["offload_folder"] = offload_folder
    return args


def _get_dtype(
    dtype: Union[str, torch.dtype], config: Optional[transformers.AutoConfig] = None
) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible."""
    if dtype is None and config is not None:
        _torch_dtype = config.torch_dtype
    elif isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class HuggingFaceAutoLM(BaseLM):
    AUTO_CONFIG_CLASS: transformers.AutoConfig = transformers.AutoConfig
    AUTO_TOKENIZER_CLASS: transformers.AutoTokenizer = transformers.AutoTokenizer
    AUTO_MODEL_CLASS: transformers.AutoModel = None
    AUTO_PEFT_CLASS: peft.PeftModel = None

    # Default max sequence length setting for when no `max_length` is provided
    # or no max length config setting is found in the model or tokenizer.
    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(
        self,
        pretrained: str,
        quantized: Optional[Union[bool, str]] = False,
        tokenizer: Optional[str] = None,
        subfolder: Optional[str] = None,
        revision: Optional[str] = "main",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 512,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        add_special_tokens: Optional[bool] = None,
        use_accelerate: Optional[bool] = False,
        low_cpu_mem_usage: Optional[bool] = True,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[int, str]] = "cuda",
        peft: str = None,
        load_in_8bit: Optional[bool] = False,
        load_in_4bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        gptq_use_triton: Optional[bool] = False,
        inject_fused_attention: Optional[bool] = True,
        bnb_4bit_quant_type: Optional[str] = None,
        bnb_4bit_compute_dtype: Optional[Union[str, torch.dtype]] = None,
        bnb_4bit_use_double_quant: Optional[bool] = False,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation.
        Args:
            pretrained (str):
                The HuggingFace Hub model ID name or the path to a pre-trained
                model to load. This is effectively the `pretrained_model_name_or_path`
                argument of `from_pretrained` in the HuggingFace `transformers` API.
            quantized (str or bool, optional, defaults to False):
                File name of a GPTQ quantized model to load. Set to `True` to use the
                default name of the quantized model.
            add_special_tokens (bool, optional, defaults to True):
                Whether to add special tokens to the input sequences. If `None`, the
                default value will be set to `True` for seq2seq models (e.g. T5) and
                `False` for causal models.
                WARNING: Evaluating causal models with `add_special_tokens=True` is
                currently __not__ supported.
            > Large model loading `accelerate` arguments
            use_accelerate (bool, optional, defaults to False):
                If True, uses the `accelerate` library to load a large model across
                multiple devices.
            low_cpu_mem_usage (bool, optional, defaults to True):
                It True, uses the `accelerate` library to accelerate loading the model.
            device_map_option (str, optional, defaults to "auto"):
                The device map option to use when loading the model with
                `accelerate`.
                Options:
                    "auto", "balanced", "balanced_low_0", "sequential"
                See the `accelerate` docs for more details on these options:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.device_map
            max_memory_per_gpu (Union[int, str], optional, defaults to None):
                The maximum memory available for each GPU in bytes as `int` or in
                the format f"{significand}{unit_symbol}" where {unit_symbol} is
                any of ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in
                the "Parameters for big model inference" section of the following
                docs:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.max_memory
            max_cpu_memory (Union[int, str], optional, defaults to None):
                The maximum available CPU RAM in bytes as `int` or in the format
                f"{significand}{unit_symbol}" where {unit_symbol} is any of
                ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in the
                "Parameters for big model inference" section of the following docs:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.max_memory
            offload_folder (str, optional, defaults to "./offload"):
                The folder to offload weights into if `device_map` contains any
                "disk" value.
            dtype (Union[str, torch.dtype], optional, defaults to None):):
                Converts the model weights to `dtype`, if specified. Strings get
                converted to `torch.dtype` objects (e.g. `float16` -> `torch.float16`).
                Use `dtype="auto"` to derive the type from the model’s weights.
            peft (str, optional, defaults to None):
                Path of the adapter weights to load from Huggingface. This will usually
                include a directory that includes the files `adapter_config.json` and
                `adapter_model.bin`. Compatible with [PEFT](https://github.com/huggingface/peft)
            load_in_8bit (bool, optional, defaults to False):
                If True, will convert the loaded model into mixed-8bit quantized model. See:
                https://huggingface.co/docs/transformers/main/en/main_classes/quantization#load-a-large-model-in-8bit
            load_in_4bit (bool, optional, defaults to False):
                If True, will convert the loaded model into mixed-4bit quantized model. See:
                https://huggingface.co/docs/transformers/main/en/main_classes/quantization#load-a-large-model-in-4bit
            trust_remote_code (bool, optional, defaults to False):
                If True, will trust the remote code when loading the model.
            gptq_use_triton (bool, optional, defaults to False):
                Use Triton for GPTQ inference.
            inject_fused_attention (bool, optional, defaults to True):
                Inject fused attention into GPTQ model.
            bnb_4bit_quant_type (str, optional, defaults to None):
                The quantization type to use for BnB 4bit quantization. See:
                https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L77
            bnb_4bit_compute_dtype (Union[str, torch.dtype], optional, defaults to None):
                The compute dtype to use for BnB 4bit quantization. See:
                https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L74
            bnb_4bit_use_double_quant (bool, optional, defaults to False):
                Whether or not to use double quant to quantize the absmax.
                https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L80

        """
        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(device, str)
        assert isinstance(batch_size, (int, str))
        if (
            add_special_tokens is not None
            and self.AUTO_MODEL_CLASS is transformers.AutoModelForCausalLM
        ):
            # TODO: Support evaluating causal models with special tokens. Currently,
            # this is not possible because the `_loglikelihood_tokens()` method for
            # causal LMs makes a no-special-tokens assumption given that contexts
            # and labels/continuations are tokenized separately without special
            # tokens, concatenated, and then processed as inputs.
            assert (
                not add_special_tokens
            ), "Evaluating causal models with `add_special_tokens=True` is currently not supported."

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self._batch_size = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self._batch_size = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._config = self.AUTO_CONFIG_CLASS.from_pretrained(
            pretrained,
            trust_remote_code=trust_remote_code,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
        )

        self._add_special_tokens = add_special_tokens
        self.tokenizer = self._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer.model_max_length = self.max_length

        model_kwargs = {}
        if use_accelerate:
            model_kwargs = _get_accelerate_args(
                low_cpu_mem_usage,
                device_map_option,
                max_memory_per_gpu,
                max_cpu_memory,
                offload_folder,
            )
        self.model = self._create_auto_model(
            pretrained=pretrained,
            quantized=quantized,
            trust_remote_code=trust_remote_code,
            revision=revision,
            subfolder=subfolder,
            torch_dtype=_get_dtype(dtype, self._config),
            gptq_use_triton=gptq_use_triton,
            inject_fused_attention=inject_fused_attention,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            **model_kwargs,
        )
        # note: peft_path can be different than pretrained model path
        if peft is not None:
            # print("PEFT is called")
            self.model = self._create_auto_model_peft(
                model=self.model,
                peft=peft,
                revision=revision,
                subfolder=subfolder,
                load_in_4bit=load_in_4bit,
            )

        # PH: start
        # # peft_model_id = "pouya-haghi/llama2_finetune_pile"
        # # config = PeftConfig.from_pretrained(peft_model_id)
        # # lm = PeftModel.from_pretrained(lm, peft_model_id)
        # # PH: end

        # # PH: start (pre-processing) finding # of times that inference is called.
        # For keeping track of activations:
        # class ReferenceCounter:
        #     def __init__(self):
        #         self.count = 0
        #         self.count_shape = 0
        #     def increase(self):
        #         self.count += 1
        #     def add_shape(self, num):
        #         self.count_shape += num
        #     def get_count(self):
        #         return self.count

        # counter = ReferenceCounter()
        # # list_output_activation = {}

        # class STEFunction_structured(torch.autograd.Function):
        #     """ define straight through estimator with overrided gradient (gate) """
        #     @staticmethod
        #     def forward(ctx, input):
        #         # ctx.save_for_backward(input.clone()) # if you want to use input during backward calculation
        #         # output = input.clone()
        #         if isinstance(input, tuple):
        #             # Clone each tensor in the tuple
        #             output = tuple(t.clone() for t in input)
        #             print("count", counter.count)
        #             # print(output.shape)
        #             return output                
        #         else:
        #             # If input is not a tuple, clone it
        #             output = input.clone()
        #             # print(output.shape)
        #             print("count", counter.count)
        #             if len(output.shape) == 3: # 3D
        #                 counter.add_shape(output.shape[1])
        #             elif len(output.shape) == 2: # 2D
        #                 counter.add_shape(output.shape[0])
        #             else:
        #                 print("Out of shape")
        #             print("shape", counter.count_shape)
        #             print("avg", counter.count_shape/counter.count)
        #             return output

        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         # # aux1 = ctx.saved_tensors # if you want to use input during backward calculation
        #         grad_input = grad_output.clone()
        #         return grad_input

        # def activation_hook(module, input, output):
        #     counter.increase() #$$$
        #     output = STEFunction_structured.apply(output)
        #     # for keeping track of activations
        #     # list_output_activation[str(module.__class__.__name__)+str("_")+str(counter.get_count())] = output #$$$
        #     return output

        # EXCLUDED_ACTIVATIONS = (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.PReLU)
        # for name, module in self.model.named_modules():
        #     if not isinstance(module, nn.ModuleList) and not list(module.children()) and "intermediate_act_fn" not in name and not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Dropout) and not any(isinstance(module, activation) for activation in EXCLUDED_ACTIVATIONS):
        #         module.register_forward_hook(activation_hook)
        #         break
        # self.model.model.layers[0].self_attn.q_proj.register_forward_hook(activation_hook)
        # # # PH: end

        # # PH: start (float8)
        # convert to float8 (and any custom float)
        # DONT FORGET TO FIRST QUANTIZE WEIGHTS TO LNS16
        # IN YOUR HOOK, YOU HAVE TO USE .CLONE OF THE ARGUMNETS AND THEN MODIFY IT AND FINALLY RETURN IT. IT DOESNT WORK WITHOUT CLONE (IT HAS TO BE OUT OF PLACE COMPUTATION, NOT IN-PLACE)

        # Create a 32-bit float tensor 3 bit mantissa, 4 bit exponent
        # num_bit_exponent = 4
        # num_bit_mantissa  = 3
        num_bit_exponent = 5
        num_bit_mantissa  = 2
        offset = torch.tensor(2**(num_bit_exponent-1))
        scale = torch.tensor(2 ** num_bit_mantissa)
        threshold_clamp = 2**(num_bit_exponent-1)
        threshold_up = float(2**threshold_clamp)
        threshold_down = float(2**-(threshold_clamp))

        # float32_tensor = torch.tensor(3.14159, dtype=torch.float32)

        # # Extract sign, exponent, and mantissa bits from the 32-bit float
        # # sign_bit = float32_tensor.sign()
        # exponent_bits = torch.floor(torch.log2(torch.abs(float32_tensor))) + offset
        # exponent = torch.pow(2, (exponent_bits - offset))
        # mantissa_bits = torch.round(((float32_tensor / exponent) - 1) * scale)
        # apx_float = ((mantissa_bits/scale) + 1) * exponent

        # For keeping track of activations:
        # class ReferenceCounter:
        #     def __init__(self):
        #         self.count = 0
        #     def increase(self):
        #         self.count += 1
        #     def get_count(self):
        #         return self.count

        # counter = ReferenceCounter()
        # list_output_activation = {}

        class STEFunction_structured(torch.autograd.Function):
            """ define straight through estimator with overrided gradient (gate) """
            @staticmethod
            def forward(ctx, input):
                # ctx.save_for_backward(input.clone()) # if you want to use input during backward calculation
                # output = input.clone()
                if isinstance(input, tuple):
                    # Clone each tensor in the tuple
                    output = tuple(t.clone() for t in input)
                    output = tuple(torch.where(t < 0, -torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up), torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up)) for t in output)
                    output = tuple(((torch.round(((t / torch.pow(2, (torch.floor(torch.log2(torch.abs(t)))))) - 1) * scale)/scale) + 1) * torch.pow(2, (torch.floor(torch.log2(torch.abs(t))))) for t in output)
                    return output                
                else:
                    # If input is not a tuple, clone it
                    output = input.clone()
                    print(output.dtype)
                    # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
                    clamped_output = torch.clamp(torch.abs(output), min=threshold_down, max=threshold_up)
                    output = torch.where(output<0, -clamped_output, clamped_output)

                    exponent_bits = torch.floor(torch.log2(torch.abs(output))) + offset
                    exponent = torch.pow(2, (exponent_bits - offset))
                    mantissa_bits = torch.round(((output / exponent) - 1) * scale)
                    output = ((mantissa_bits/scale) + 1) * exponent
                    return output

            @staticmethod
            def backward(ctx, grad_output):
                # # aux1 = ctx.saved_tensors # if you want to use input during backward calculation
                grad_input = grad_output.clone()
                return grad_input
                # # aux1 = ctx.saved_tensors # if you want to use input during backward calculation
                # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
                # grad_input = grad_output.clone()
                # clamped_output = torch.clamp(torch.abs(grad_input), min=threshold_down, max=threshold_up)
                # grad_input = torch.where(grad_input<0, -clamped_output, clamped_output)

                # exponent_bits = torch.floor(torch.log2(torch.abs(grad_input))) + offset
                # exponent = torch.pow(2, (exponent_bits - offset))
                # mantissa_bits = torch.round(((grad_input / exponent) - 1) * scale)
                # grad_input = ((mantissa_bits/scale) + 1) * exponent
                # return grad_input

        def activation_hook(module, input, output):
            output = STEFunction_structured.apply(output)
            # for keeping track of activations
            # list_output_activation[str(module.__class__.__name__)+str("_")+str(counter.get_count())] = output #$$$
            # counter.increase() #$$$
            return output

        EXCLUDED_ACTIVATIONS = (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.PReLU)

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.ModuleList) and not list(module.children()) and "intermediate_act_fn" not in name and not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Dropout) and not any(isinstance(module, activation) for activation in EXCLUDED_ACTIVATIONS):
                module.register_forward_hook(activation_hook)
        # model.model.layers[0].self_attn.q_proj.register_forward_hook(activation_hook)
        # # # PH: end

        # # # PH: start (LNS8)
        # num_bit_mantissa = 4 # for 8 bit repr.
        # # num_frac = 10 # fractional bits for 16 bit repr.
        # num_frac = 3 # fractional bits for 8 bit repr.

        # scale = float(2**num_frac)
        # threshold_clamp = 2**(num_bit_mantissa-1)
        # threshold_up = float(2**threshold_clamp)
        # threshold_down = float(2**-(threshold_clamp))

        # # For keeping track of activations:
        # # class ReferenceCounter:
        # #     def __init__(self):
        # #         self.count = 0
        # #     def increase(self):
        # #         self.count += 1
        # #     def get_count(self):
        # #         return self.count

        # # counter = ReferenceCounter()
        # # list_output_activation = {}

        # class STEFunction_structured(torch.autograd.Function):
        #     """ define straight through estimator with overrided gradient (gate) """
        #     @staticmethod
        #     def forward(ctx, input):
        #         # ctx.save_for_backward(input.clone()) # if you want to use input during backward calculation
        #         if isinstance(input, tuple):
        #             # Clone each tensor in the tuple
        #             output = tuple(t.clone() for t in input)
        #             output = tuple(torch.where(t<0, -torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up), torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up)) for t in output)
        #             output = tuple(torch.where(t > 0, torch.pow(2,(torch.round(torch.log2(t)*scale))/scale), torch.where(t < 0, -torch.pow(2,(torch.round(torch.log2(-t)*scale)/scale)), t)) for t in output)
        #             return output             
        #         else:
        #             output = input.clone()
        #             # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
        #             clamped_output = torch.clamp(torch.abs(output), min=threshold_down, max=threshold_up)
        #             output = torch.where(output<0, -clamped_output, clamped_output)
        #             # v1: concise
        #             output = torch.where(output > 0, torch.pow(2,(torch.round(torch.log2(output)*scale))/scale), torch.where(output < 0, -torch.pow(2,(torch.round(torch.log2(-output)*scale)/scale)), output))
        #             return output

        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         grad_input = grad_output.clone()
        #         return grad_input

        # def activation_hook(module, input, output):
        #     output = STEFunction_structured.apply(output)
        #     return output

        # EXCLUDED_ACTIVATIONS = (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.PReLU)

        # for name, module in self.model.named_modules():
        #     if not isinstance(module, nn.ModuleList) and not list(module.children()) and "intermediate_act_fn" not in name and not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Dropout) and not any(isinstance(module, activation) for activation in EXCLUDED_ACTIVATIONS):
        #         module.register_forward_hook(activation_hook)
        # # PH: end

        # # # PH: start (modified LNS8 - old)
        # num_bit_mantissa = 5 # for 16 bit repr.
        # num_bit_mantissa = 5 # for 8 bit repr.
        # threshold_mantissa = 2**(num_bit_mantissa-1)
        # threshold_up = float(2**threshold_mantissa)
        # threshold_down = float(2**-(threshold_mantissa))

        # # new version:
        # max_num_bit_mantissa_needed = 1 # according to the distribution you can get this number
        # log_domain_threshold = 2** max_num_bit_mantissa_needed # 4
        # real_domain_threshold_up = float(2**log_domain_threshold) # 16
        # real_domain_threshold_down = float(2**(-log_domain_threshold)) # 1/16

        # # num_frac_low_prec = 10 # number of fractional bits for 16 bit repr.
        # num_frac_low_prec = 2 # number of fractional bits for 8 bit repr.
        # # num_frac_high_prec = num_frac_low_prec + (num_bit_mantissa-max_num_bit_mantissa_needed) # 13
        # num_frac_high_prec = num_frac_low_prec + 2 # 13
        # scale_low_prec = 2**(num_frac_low_prec)
        # scale_high_prec = 2**(num_frac_high_prec)
        # # v3:
        # num_frac_highest_prec = num_frac_high_prec + 2 # for extreme outliers
        # scale_highest_prec = 2**(num_frac_highest_prec)

        # # For keeping track of activations:
        # # class ReferenceCounter:
        # #     def __init__(self):
        # #         self.count = 0
        # #     def increase(self):
        # #         self.count += 1
        # #     def get_count(self):
        # #         return self.count

        # # counter = ReferenceCounter()
        # # list_output_activation = {}

        # class STEFunction_structured(torch.autograd.Function):
        #     """ define straight through estimator with overrided gradient (gate) """
        #     @staticmethod
        #     def forward(ctx, input):
        #         # ctx.save_for_backward(input.clone()) # if you want to use input during backward calculation
        #         if isinstance(input, tuple):
        #             output = tuple(t.clone() for t in input)
        #             output = tuple(torch.where(t<0, -torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up), torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up)) for t in output)
        #             output = tuple(torch.where(t > 0, torch.pow(2, torch.where(torch.log2(t)>torch.max(torch.log2(t))-5, torch.where(torch.log2(t)>torch.max(torch.log2(t))-3, torch.round(torch.log2(t) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.log2(t) * scale_high_prec)/ scale_high_prec), torch.round(torch.log2(t) * scale_low_prec)/ scale_low_prec)), torch.where(t < 0, -torch.pow(2, torch.where(torch.log2(-t)>torch.max(torch.log2(-t))-5, torch.where(torch.log2(-t)>torch.max(torch.log2(-t))-3, torch.round(torch.log2(-t) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.log2(-t) * scale_high_prec)/ scale_high_prec), torch.round(torch.log2(-t) * scale_low_prec)/ scale_low_prec)), t)) for t in output)
        #             return output
        #         else:
        #             output = input.clone()
        #             # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
        #             clamped_output = torch.clamp(torch.abs(output), min=threshold_down, max=threshold_up)
        #             output = torch.where(output<0, -clamped_output, clamped_output)

        #             # v3:
        #             if len(output.shape) == 3: # 3D
        #               non_zero_indices = output.nonzero()
        #               non_zero_values = output[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]] # 0 because the first dimension is batch, 1 b/c next one is first dimension of feature, 2 b/c it is second dimension of features
        #               # if 1D: non_zero_indices
        #               if len(non_zero_values) > 0: # any nonzero avail
        #                 log_x = torch.where(non_zero_values > 0, torch.log2(non_zero_values), torch.log2(-non_zero_values))
        #                 quant_exponent_low_prec = torch.round(log_x * scale_low_prec)/ scale_low_prec # 2**3 - round(+ 0.5)
        #                 quant_exponent_high_prec = torch.round(log_x * scale_high_prec)/ scale_high_prec # 2**3 - round(+ 0.5)
        #                 # --------  v3 (including extreme outliers) ---------
        #                 quant_exponent_highest_prec = torch.round(log_x * scale_highest_prec)/ scale_highest_prec # 2**3 - round(+ 0.5)
        #                 max_val = torch.max(log_x)
        #                 quant_exponent = torch.where(log_x>max_val-5, torch.where(log_x>max_val-3, quant_exponent_highest_prec, quant_exponent_high_prec), quant_exponent_low_prec) # max_val-3 and max_val-5 are thresholds for extreme and moderate outliers (beta nd gamma)
        #                 # ------- end v3 ---------
        #                 quantized_values = torch.where(non_zero_values > 0, torch.pow(2, quant_exponent), -(torch.pow(2, quant_exponent)))
        #                 output[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]] = quantized_values
        #             elif len(output.shape) == 2: # 2D
        #               non_zero_indices = output.nonzero()
        #               non_zero_values = output[non_zero_indices[:, 0], non_zero_indices[:, 1]] # 0 because the first dimension is batch, 1 b/c next one is first dimension of feature, 2 b/c it is second dimension of features
        #               if len(non_zero_values) > 0:
        #                 log_x = torch.where(non_zero_values > 0, torch.log2(non_zero_values), torch.log2(-non_zero_values))
        #                 quant_exponent_low_prec = torch.round(log_x * scale_low_prec) / scale_low_prec # 2**3 - round(+ 0.5)
        #                 quant_exponent_high_prec = torch.round(log_x * scale_high_prec) / scale_high_prec # 2**3 - round(+ 0.5)
        #                 # --------  v3 (including extreme outliers) ---------
        #                 quant_exponent_highest_prec = torch.round(log_x * scale_highest_prec)/ scale_highest_prec # 2**3 - round(+ 0.5)
        #                 max_val = torch.max(log_x)
        #                 quant_exponent = torch.where(log_x>max_val-5, torch.where(log_x>max_val-3, quant_exponent_highest_prec, quant_exponent_high_prec), quant_exponent_low_prec) # max_val-3 and max_val-5 are thresholds for extreme and moderate outliers (beta nd gamma)
        #                 # ------- end v3 ---------
        #                 quantized_values = torch.where(non_zero_values > 0, torch.pow(2, quant_exponent), -(torch.pow(2, quant_exponent)))
        #                 output[non_zero_indices[:, 0], non_zero_indices[:, 1]] = quantized_values
        #             else:
        #               print("Out of shape")
        #             return output

        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         # aux1 = ctx.saved_tensors # if you want to use input during backward calculation
        #         grad_input = grad_output.clone()
        #         return grad_input

        # def activation_hook(module, input, output):
        #     output = STEFunction_structured.apply(output)
        #     return output

        # EXCLUDED_ACTIVATIONS = (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.PReLU)

        # for name, module in self.model.named_modules():
        #     if not isinstance(module, nn.ModuleList) and not list(module.children()) and "intermediate_act_fn" not in name and not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Dropout) and not any(isinstance(module, activation) for activation in EXCLUDED_ACTIVATIONS):
                # module.register_forward_hook(activation_hook)

        # # # PH: end

        # # # # PH: start (modified LNS8 with pervector quant optimization, combined)
        # # num_bit_mantissa = 5 # for 16 bit repr.
        # num_bit_mantissa = 5 # for 8 bit repr.
        # threshold_mantissa = 2**(num_bit_mantissa-1)
        # threshold_up = float(2**threshold_mantissa)
        # threshold_down = float(2**-(threshold_mantissa))

        # # new version:
        # # max_num_bit_mantissa_needed = 1 # according to the distribution you can get this number
        # # log_domain_threshold = 2** max_num_bit_mantissa_needed # 4
        # # real_domain_threshold_up = float(2**log_domain_threshold) # 16
        # # real_domain_threshold_down = float(2**(-log_domain_threshold)) # 1/16

        # # num_frac_low_prec = 10 # number of fractional bits for 16 bit repr.
        # num_frac_low_prec = 2 # number of fractional bits for 8 bit repr.
        # # num_frac_high_prec = num_frac_low_prec + (num_bit_mantissa-max_num_bit_mantissa_needed) # 13
        # num_frac_high_prec = num_frac_low_prec + 2 # 13
        # scale_low_prec = 2**(num_frac_low_prec)
        # scale_high_prec = 2**(num_frac_high_prec)
        # # v3:
        # num_frac_highest_prec = num_frac_high_prec + 2 # for extreme outliers
        # scale_highest_prec = 2**(num_frac_highest_prec)

        # # For keeping track of activations:
        # # class ReferenceCounter:
        # #     def __init__(self):
        # #         self.count = 0
        # #     def increase(self):
        # #         self.count += 1
        # #     def get_count(self):
        # #         return self.count

        # # counter = ReferenceCounter()
        # # list_output_activation = {}

        # class STEFunction_structured(torch.autograd.Function):
        #     """ define straight through estimator with overrided gradient (gate) """
        #     @staticmethod
        #     def forward(ctx, input):
        #         # ctx.save_for_backward(input.clone()) # if you want to use input during backward calculation
        #         if isinstance(input, tuple):
        #             output = tuple(t.clone() for t in input)
        #             output = tuple(torch.where(t<0, -torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up), torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up)) for t in output)
        #             # output = tuple(torch.where(t > 0, torch.pow(2, torch.where(torch.log2(t)>torch.max(torch.log2(t))-5, torch.where(torch.log2(t)>torch.max(torch.log2(t))-3, torch.round(torch.log2(t) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.log2(t) * scale_high_prec)/ scale_high_prec), torch.round(torch.log2(t) * scale_low_prec)/ scale_low_prec)), torch.where(t < 0, -torch.pow(2, torch.where(torch.log2(-t)>torch.max(torch.log2(-t))-5, torch.where(torch.log2(-t)>torch.max(torch.log2(-t))-3, torch.round(torch.log2(-t) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.log2(-t) * scale_high_prec)/ scale_high_prec), torch.round(torch.log2(-t) * scale_low_prec)/ scale_low_prec)), t)) for t in output)
        #             # output = tuple(torch.where(t > 0, torch.pow(4, torch.where((torch.log2(t)/2)>torch.max((torch.log2(t)/2))-4, torch.where((torch.log2(t)/2)>torch.max((torch.log2(t)/2))-3, torch.round((torch.log2(t)/2) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(t)/2) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(t)/2) * scale_low_prec)/ scale_low_prec)), torch.where(t < 0, -torch.pow(4, torch.where((torch.log2(-t)/2)>torch.max((torch.log2(-t)/2))-4, torch.where((torch.log2(-t)/2)>torch.max((torch.log2(-t)/2))-3, torch.round((torch.log2(-t)/2) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(-t)/2) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(-t)/2) * scale_low_prec)/ scale_low_prec)), t)) for t in output)                    
        #             # output = tuple(torch.where(t > 0, torch.pow(2, torch.where((torch.log2(t))>torch.max((torch.log2(t)))-4, torch.where((torch.log2(t))>torch.max((torch.log2(t)))-3, torch.round((torch.log2(t)) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(t)) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(t)) * scale_low_prec)/ scale_low_prec)), torch.where(t < 0, -torch.pow(2, torch.where((torch.log2(-t))>torch.max((torch.log2(-t)))-4, torch.where((torch.log2(-t))>torch.max((torch.log2(-t)))-3, torch.round((torch.log2(-t)) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(-t)) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(-t)) * scale_low_prec)/ scale_low_prec)), t)) for t in output)                    
        #             output = tuple(torch.where(t<0, -(torch.pow(2, torch.where(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))))-5, torch.where(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))))-3, torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_high_prec)/ scale_high_prec), torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_low_prec)/ scale_low_prec))), torch.where(t>0, torch.pow(2, torch.where(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))))-5, torch.where(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))))-3, torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_high_prec)/ scale_high_prec), torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_low_prec)/ scale_low_prec)), t)) for t in output)
        #             return output
        #         else:
        #             output = input.clone()
        #             # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
        #             clamped_output = torch.clamp(torch.abs(output), min=threshold_down, max=threshold_up)
                    # output = torch.where(output<0, -clamped_output, clamped_output)
                    # # v3:
                    # log_x = torch.where(output<0, torch.log2(-output), torch.where(output > 0, torch.log2(output), torch.tensor(-64000.0)))
                    # quant_exponent_low_prec = torch.round(log_x * scale_low_prec)/ scale_low_prec # 2**3 - round(+ 0.5)
                    # quant_exponent_high_prec = torch.round(log_x * scale_high_prec)/ scale_high_prec # 2**3 - round(+ 0.5)
                    # quant_exponent_highest_prec = torch.round(log_x * scale_highest_prec)/ scale_highest_prec # 2**3 - round(+ 0.5)
                    # max_val = torch.max(log_x)
                    # quant_exponent = torch.where(log_x>max_val-5, torch.where(log_x>max_val-3, quant_exponent_highest_prec, quant_exponent_high_prec), quant_exponent_low_prec) # max_val-3 and max_val-5 are thresholds for extreme and moderate outliers (beta nd gamma)
                    # output = torch.where(output<0, -(torch.pow(2, quant_exponent)), torch.where(output>0, torch.pow(2, quant_exponent), output))
                    # return output

        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         # aux1 = ctx.saved_tensors # if you want to use input during backward calculation
        #         grad_input = grad_output.clone()
        #         return grad_input

        # def activation_hook(module, input, output):
        #     output = STEFunction_structured.apply(output)
        #     return output

        # EXCLUDED_ACTIVATIONS = (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.PReLU)

        # for name, module in self.model.named_modules():
        #     if not isinstance(module, nn.ModuleList) and not list(module.children()) and "intermediate_act_fn" not in name and not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Dropout) and not any(isinstance(module, activation) for activation in EXCLUDED_ACTIVATIONS):
        #         module.register_forward_hook(activation_hook)
        # PH: end

        # # # # PH: start (modified LNS8 without pervector quant optimization)
        # num_bit_mantissa = 5 # for 16 bit repr.
        # num_bit_mantissa = 5 # for 8 bit repr.
        # threshold_mantissa = 2**(num_bit_mantissa-1)
        # threshold_up = float(2**threshold_mantissa)
        # threshold_down = float(2**-(threshold_mantissa))

        # # new version:
        # # max_num_bit_mantissa_needed = 1 # according to the distribution you can get this number
        # # log_domain_threshold = 2** max_num_bit_mantissa_needed # 4
        # # real_domain_threshold_up = float(2**log_domain_threshold) # 16
        # # real_domain_threshold_down = float(2**(-log_domain_threshold)) # 1/16

        # # num_frac_low_prec = 10 # number of fractional bits for 16 bit repr.
        # num_frac_low_prec = 2 # number of fractional bits for 8 bit repr.
        # # num_frac_high_prec = num_frac_low_prec + (num_bit_mantissa-max_num_bit_mantissa_needed) # 13
        # num_frac_high_prec = num_frac_low_prec + 2 # 13
        # scale_low_prec = 2**(num_frac_low_prec)
        # scale_high_prec = 2**(num_frac_high_prec)
        # # v3:
        # num_frac_highest_prec = num_frac_high_prec + 2 # for extreme outliers
        # scale_highest_prec = 2**(num_frac_highest_prec)

        # # For keeping track of activations:
        # # class ReferenceCounter:
        # #     def __init__(self):
        # #         self.count = 0
        # #     def increase(self):
        # #         self.count += 1
        # #     def get_count(self):
        # #         return self.count

        # # counter = ReferenceCounter()
        # # list_output_activation = {}

        # class STEFunction_structured(torch.autograd.Function):
        #     """ define straight through estimator with overrided gradient (gate) """
        #     @staticmethod
        #     def forward(ctx, input):
        #         # ctx.save_for_backward(input.clone()) # if you want to use input during backward calculation
        #         if isinstance(input, tuple):
        #             output = tuple(t.clone() for t in input)
        #             output = tuple(torch.where(t<0, -torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up), torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up)) for t in output)
        #             # output = tuple(torch.where(t > 0, torch.pow(2, torch.where(torch.log2(t)>torch.max(torch.log2(t))-5, torch.where(torch.log2(t)>torch.max(torch.log2(t))-3, torch.round(torch.log2(t) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.log2(t) * scale_high_prec)/ scale_high_prec), torch.round(torch.log2(t) * scale_low_prec)/ scale_low_prec)), torch.where(t < 0, -torch.pow(2, torch.where(torch.log2(-t)>torch.max(torch.log2(-t))-5, torch.where(torch.log2(-t)>torch.max(torch.log2(-t))-3, torch.round(torch.log2(-t) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.log2(-t) * scale_high_prec)/ scale_high_prec), torch.round(torch.log2(-t) * scale_low_prec)/ scale_low_prec)), t)) for t in output)
        #             # output = tuple(torch.where(t > 0, torch.pow(4, torch.where((torch.log2(t)/2)>torch.max((torch.log2(t)/2))-4, torch.where((torch.log2(t)/2)>torch.max((torch.log2(t)/2))-3, torch.round((torch.log2(t)/2) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(t)/2) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(t)/2) * scale_low_prec)/ scale_low_prec)), torch.where(t < 0, -torch.pow(4, torch.where((torch.log2(-t)/2)>torch.max((torch.log2(-t)/2))-4, torch.where((torch.log2(-t)/2)>torch.max((torch.log2(-t)/2))-3, torch.round((torch.log2(-t)/2) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(-t)/2) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(-t)/2) * scale_low_prec)/ scale_low_prec)), t)) for t in output)                    
        #             # output = tuple(torch.where(t > 0, torch.pow(2, torch.where((torch.log2(t))>torch.max((torch.log2(t)))-4, torch.where((torch.log2(t))>torch.max((torch.log2(t)))-3, torch.round((torch.log2(t)) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(t)) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(t)) * scale_low_prec)/ scale_low_prec)), torch.where(t < 0, -torch.pow(2, torch.where((torch.log2(-t))>torch.max((torch.log2(-t)))-4, torch.where((torch.log2(-t))>torch.max((torch.log2(-t)))-3, torch.round((torch.log2(-t)) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(-t)) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(-t)) * scale_low_prec)/ scale_low_prec)), t)) for t in output)                    
        #             # output = tuple(torch.where(t<0, -(torch.pow(2, torch.where(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))))-5, torch.where(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))))-3, torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_high_prec)/ scale_high_prec), torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_low_prec)/ scale_low_prec))), torch.where(t>0, torch.pow(2, torch.where(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))))-5, torch.where(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))))-3, torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_high_prec)/ scale_high_prec), torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_low_prec)/ scale_low_prec)), t)) for t in output)
        #             output = tuple(torch.where(t<0, -(torch.pow(2, torch.where(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))), dim=0).values.unsqueeze(0).expand_as(t)-5, torch.where(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))), dim=0).values.unsqueeze(0).expand_as(t)-3, torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_high_prec)/ scale_high_prec), torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_low_prec)/ scale_low_prec))), torch.where(t>0, torch.pow(2, torch.where(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))), dim=0).values.unsqueeze(0).expand_as(t)-5, torch.where(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))), dim=0).values.unsqueeze(0).expand_as(t)-3, torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_high_prec)/ scale_high_prec), torch.round(torch.where(t<0, torch.log2(-t), torch.where(t > 0, torch.log2(t), torch.tensor(-64000.0))) * scale_low_prec)/ scale_low_prec)), t)) for t in output)
        #             return output
        #         else:
        #             output = input.clone()
        #             # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
        #             clamped_output = torch.clamp(torch.abs(output), min=threshold_down, max=threshold_up)
        #             output = torch.where(output<0, -clamped_output, clamped_output)
        #             # v3:
        #             log_x = torch.where(output<0, torch.log2(-output), torch.where(output > 0, torch.log2(output), torch.tensor(-64000.0)))
        #             quant_exponent_low_prec = torch.round(log_x * scale_low_prec)/ scale_low_prec # 2**3 - round(+ 0.5)
        #             quant_exponent_high_prec = torch.round(log_x * scale_high_prec)/ scale_high_prec # 2**3 - round(+ 0.5)
        #             quant_exponent_highest_prec = torch.round(log_x * scale_highest_prec)/ scale_highest_prec # 2**3 - round(+ 0.5)
        #             if len(output.shape) == 3: # 3D
        #                 max_val = torch.max(log_x, dim=1).values.unsqueeze(1).expand_as(log_x)
        #             elif len(output.shape) == 2: # 2D
        #                 max_val = torch.max(log_x, dim=0).values.unsqueeze(0).expand_as(log_x)
        #             else:
        #                 print("Out of shape")
        #             quant_exponent = torch.where(log_x>max_val-5, torch.where(log_x>max_val-3, quant_exponent_highest_prec, quant_exponent_high_prec), quant_exponent_low_prec) # max_val-3 and max_val-5 are thresholds for extreme and moderate outliers (beta nd gamma)
        #             output = torch.where(output<0, -(torch.pow(2, quant_exponent)), torch.where(output>0, torch.pow(2, quant_exponent), output))
        #             return output

        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         # aux1 = ctx.saved_tensors # if you want to use input during backward calculation
        #         grad_input = grad_output.clone()
        #         return grad_input

        # def activation_hook(module, input, output):
        #     output = STEFunction_structured.apply(output)
        #     return output

        # EXCLUDED_ACTIVATIONS = (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.PReLU)

        # for name, module in self.model.named_modules():
        #     if not isinstance(module, nn.ModuleList) and not list(module.children()) and "intermediate_act_fn" not in name and not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Dropout) and not any(isinstance(module, activation) for activation in EXCLUDED_ACTIVATIONS):
        #         module.register_forward_hook(activation_hook)
        # PH: end

        # # # PH: start (zeroquant) per-row quant for both activation and weights
        # num_bit = 8

        # # For keeping track of activations:
        # # class ReferenceCounter:
        # #     def __init__(self):
        # #         self.count = 0
        # #     def increase(self):
        # #         self.count += 1
        # #     def get_count(self):
        # #         return self.count

        # # counter = ReferenceCounter()
        # # list_output_activation = {}

        # class STEFunction_structured(torch.autograd.Function):
        #     """ define straight through estimator with overrided gradient (gate) """
        #     @staticmethod
        #     def forward(ctx, input):
        #         # ctx.save_for_backward(input.clone()) # if you want to use input during backward calculation
        #         if isinstance(input, tuple):
        #             # Clone each tensor in the tuple
        #             output = tuple(t.clone() for t in input)
        #             output = tuple(torch.where(t<0, -torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))-1)).unsqueeze(1)), torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))-1)).unsqueeze(1))) for t in output)
        #             output = tuple((torch.round(t*(torch.pow(2, torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))).unsqueeze(1))))/(torch.pow(2, torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))).unsqueeze(1)) for t in output)
        #             # output = tuple((torch.round(torch.where(torch.where(t<0, -torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1)).unsqueeze(1)), torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1)).unsqueeze(1))) == 0, torch.tensor(0.0), torch.where(t<0, -torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1)).unsqueeze(1)), torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1)).unsqueeze(1))))*torch.pow(2, torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(torch.where(torch.where(t<0, -torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1)).unsqueeze(1)), torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1)).unsqueeze(1))) == 0, torch.tensor(0.0), torch.where(t<0, -torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1)).unsqueeze(1)), torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1)).unsqueeze(1))))), dim=1)[0])), min=0, max=num_bit)).unsqueeze(1)))/torch.pow(2, torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(torch.where(torch.where(t<0, -torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1)).unsqueeze(1)), torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1)).unsqueeze(1))) == 0, torch.tensor(0.0), torch.where(t<0, -torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1)).unsqueeze(1)), torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0])), min=0, max=num_bit)-1)).unsqueeze(1))))), dim=1)[0])), min=0, max=num_bit)).unsqueeze(1) for t in output)
        #             return output             
        #         else:
        #             output = input.clone()
        #             max_values, _ = torch.max(torch.abs(output), dim=1) # it is now a vector, - is needed b/c otherwise torch.max returns both maximum values and indices of maximums
        #             # num_frac = torch.floor(torch.log2((2**(num_bit-1) - 1)/max_values)) # fractional bits for 8 bit repr.
        #             num_frac = torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/max_values)), min=0, max=num_bit) #!!#
        #             num_bit_mantissa = num_bit -  num_frac # these are also vectors
        #             scale = torch.pow(2, num_frac)
        #             threshold_clamp = torch.pow(2, num_bit_mantissa-1)
        #             threshold_up = torch.pow(2, threshold_clamp)
        #             threshold_down = torch.pow(2, -(threshold_clamp))
        #             # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
        #             clamped_output = torch.clamp(torch.abs(output), min=threshold_down.unsqueeze(1), max=threshold_up.unsqueeze(1))
        #             output = torch.where(output<0, -clamped_output, clamped_output)
        #             output = torch.where(output == 0, torch.tensor(0.0), output) #!!#
        #             output = (torch.round(output*scale.unsqueeze(1)))/scale.unsqueeze(1)
        #             return output

        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         grad_input = grad_output.clone()
        #         return grad_input

        # def activation_hook(module, input, output):
        #     output = STEFunction_structured.apply(output)
        #     return output

        # EXCLUDED_ACTIVATIONS = (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.PReLU)

        # for name, module in self.model.named_modules():
        #     if not isinstance(module, nn.ModuleList) and not list(module.children()) and "intermediate_act_fn" not in name and not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Dropout) and not any(isinstance(module, activation) for activation in EXCLUDED_ACTIVATIONS):
        #         module.register_forward_hook(activation_hook)
        # # PH: end

        # # PH: start (W8A8) per-tensor quant for both activation and weights
        # num_bit = 8

        # # For keeping track of activations:
        # # class ReferenceCounter:
        # #     def __init__(self):
        # #         self.count = 0
        # #     def increase(self):
        # #         self.count += 1
        # #     def get_count(self):
        # #         return self.count

        # # counter = ReferenceCounter()
        # # list_output_activation = {}

        # class STEFunction_structured(torch.autograd.Function):
        #     """ define straight through estimator with overrided gradient (gate) """
        #     @staticmethod
        #     def forward(ctx, input):
        #         # ctx.save_for_backward(input.clone()) # if you want to use input during backward calculation
        #         if isinstance(input, tuple):
        #             # Clone each tensor in the tuple
        #             output = tuple(t.clone() for t in input)
        #             output = tuple(torch.where(t<0, -torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t))))-1))), max=torch.pow(2, torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t))))-1))), torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t))))-1))), max=torch.pow(2, torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t))))-1)))) for t in output)
        #             output = tuple((torch.round(t*torch.pow(2, torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t)))))))/torch.pow(2, torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t))))) for t in output)
        #             return output             
        #         else:
        #             output = input.clone()
        #             max_values = torch.max(torch.abs(output))
        #             num_frac = torch.floor(torch.log2((2**(num_bit-1) - 1)/max_values)) # fractional bits for 8 bit repr.
        #             num_bit_mantissa = num_bit -  num_frac # these are also vectors
        #             scale = torch.pow(2, num_frac)
        #             threshold_clamp = torch.pow(2, num_bit_mantissa-1)
        #             threshold_up = torch.pow(2, threshold_clamp)
        #             threshold_down = torch.pow(2, -(threshold_clamp))
        #             # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
        #             clamped_output = torch.clamp(torch.abs(output), min=threshold_down, max=threshold_up)
        #             output = torch.where(output<0, -clamped_output, clamped_output)
        #             output = (torch.round(output*scale))/scale
        #             return output

        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         grad_input = grad_output.clone()
        #         return grad_input

        # def activation_hook(module, input, output):
        #     output = STEFunction_structured.apply(output)
        #     return output

        # EXCLUDED_ACTIVATIONS = (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.PReLU)

        # for name, module in self.model.named_modules():
        #     if not isinstance(module, nn.ModuleList) and not list(module.children()) and "intermediate_act_fn" not in name and not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Dropout) and not any(isinstance(module, activation) for activation in EXCLUDED_ACTIVATIONS):
        #         module.register_forward_hook(activation_hook)
        # # PH: end

        # # PH: start (smoothquant) scaling per column for activation and per row for weights. Then do a zeroquant.
        # num_bit = 8

        # # For keeping track of activations:
        # # class ReferenceCounter:
        # #     def __init__(self):
        # #         self.count = 0
        # #     def increase(self):
        # #         self.count += 1
        # #     def get_count(self):
        # #         return self.count

        # # counter = ReferenceCounter()
        # # list_output_activation = {}

        # class STEFunction_structured(torch.autograd.Function):
        #     """ define straight through estimator with overrided gradient (gate) """
        #     @staticmethod
        #     def forward(ctx, input):
        #         # ctx.save_for_backward(input.clone()) # if you want to use input during backward calculation
        #         # v1:
        #         # if isinstance(input, tuple):
        #         #     # Clone each tensor in the tuple
        #         #     output = tuple(t.clone() for t in input)
        #         #     # First do scaling
        #         #     # output = tuple(t/torch.max(torch.abs(t), dim=0)[0].unsqueeze(0) for t in output)
        #         #     output = tuple(t/torch.where(torch.max(torch.abs(t), dim=0)[0]==0, torch.tensor(1.0), torch.max(torch.abs(t), dim=0)[0]).unsqueeze(0) for t in output)
        #         #     # now do zeroquant
        #         #     output = tuple(torch.where(t<0, -torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))-1)).unsqueeze(1)), torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))-1)).unsqueeze(1))) for t in output)
        #         #     output = tuple((torch.round(t*torch.pow(2, torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))).unsqueeze(1)))/torch.pow(2, torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))).unsqueeze(1) for t in output)
        #         #     # Then scale back
        #         #     # output = tuple(t*torch.max(torch.abs(t), dim=0)[0].unsqueeze(0) for t in output)
        #         #     output = tuple(t*torch.where(torch.max(torch.abs(t), dim=0)[0]==0, torch.tensor(1.0), torch.max(torch.abs(t), dim=0)[0]).unsqueeze(0) for t in output)
        #         #     return output             
        #         # else:
        #         #     output = input.clone()
        #         #     # First do scaling
        #         #     max_val_c = torch.max(torch.abs(output), dim=0)[0] # get max for each column
        #         #     max_val_c = torch.where(max_val_c==0, torch.tensor(1.0), max_val_c) # VERY IMPORTANT: YOU NEED TO REPLACE ZEROS WITH ONES JUST IN CASE IF THE MAX WAS ZERO WHICH LEADS TO NAN
        #         #     output = output/max_val_c.unsqueeze(0)
        #         #     # now do zeroquant
        #         #     max_values = torch.max(torch.abs(output), dim=1)[0] # it is now a vector, - is needed b/c otherwise torch.max returns both maximum values and indices of maximums
        #         #     num_frac = torch.floor(torch.log2((2**(num_bit-1) - 1)/max_values)) # fractional bits for 8 bit repr.
        #         #     num_bit_mantissa = num_bit -  num_frac # these are also vectors
        #         #     scale = torch.pow(2, num_frac)
        #         #     threshold_clamp = torch.pow(2, num_bit_mantissa-1)
        #         #     threshold_up = torch.pow(2, threshold_clamp)
        #         #     threshold_down = torch.pow(2, -(threshold_clamp))
        #         #     # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
        #         #     clamped_output = torch.clamp(torch.abs(output), min=threshold_down.unsqueeze(1), max=threshold_up.unsqueeze(1))
        #         #     output = torch.where(output<0, -clamped_output, clamped_output)
        #         #     output = (torch.round(output*scale.unsqueeze(1)))/scale.unsqueeze(1)
        #         #     # Then scale back
        #         #     output = output*max_val_c.unsqueeze(0)
        #         #     return output
        #         if isinstance(input, tuple):
        #             # Clone each tensor in the tuple
        #             output = tuple(t.clone() for t in input)
        #             # print("Error")
        #             # First do scaling
        #             # output = tuple(t/torch.max(torch.abs(t), dim=0)[0].unsqueeze(0) for t in output)
        #             output = tuple(t/torch.where(torch.max(torch.abs(t), dim=0)[0]==0, torch.tensor(1.0), torch.max(torch.abs(t), dim=0)[0]).unsqueeze(0) for t in output)
        #             # now do zeroquant
        #             output = tuple(torch.where(t<0, -torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))-1)).unsqueeze(1)), torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit -  torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))-1)).unsqueeze(1))) for t in output)
        #             output = tuple((torch.round(t*torch.pow(2, torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))).unsqueeze(1)))/torch.pow(2, torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.max(torch.abs(t), dim=1)[0]))).unsqueeze(1) for t in output)
        #             # Then scale back
        #             # output = tuple(t*torch.max(torch.abs(t), dim=0)[0].unsqueeze(0) for t in output)
        #             output = tuple(t*torch.where(torch.max(torch.abs(t), dim=0)[0]==0, torch.tensor(1.0), torch.max(torch.abs(t), dim=0)[0]).unsqueeze(0) for t in output)
        #             return output             
        #         else:
        #             output = input.clone()
        #             # First do scaling
        #             if len(output.shape) == 3: # 3D
        #                 max_val_c = torch.max(torch.abs(output), dim=1)[0] # get max for each column
        #             elif len(output.shape) == 2: # 2D
        #                 max_val_c = torch.max(torch.abs(output), dim=0)[0] # get max for each column
        #             else:
        #                 print("Out of shape")
        #             max_val_c = torch.where(max_val_c==0, torch.tensor(1.0), max_val_c) # VERY IMPORTANT: YOU NEED TO REPLACE ZEROS WITH ONES JUST IN CASE IF THE MAX WAS ZERO WHICH LEADS TO NAN
        #             if len(output.shape) == 3: # 3D
        #                 output = output/max_val_c.unsqueeze(1)
        #             elif len(output.shape) == 2: # 2D
        #                 output = output/max_val_c.unsqueeze(0)
        #             else:
        #                 print("Out of shape")
        #             # now do zeroquant
        #             if len(output.shape) == 3: # 3D
        #                 max_values = torch.max(torch.abs(output), dim=2)[0] # it is now a vector, - is needed b/c otherwise torch.max returns both maximum values and indices of maximums
        #             elif len(output.shape) == 2: # 2D
        #                 max_values = torch.max(torch.abs(output), dim=1)[0] # it is now a vector, - is needed b/c otherwise torch.max returns both maximum values and indices of maximums
        #             else:
        #                 print("Out of shape")
        #             num_frac = torch.floor(torch.log2((2**(num_bit-1) - 1)/max_values)) # fractional bits for 8 bit repr.
        #             num_bit_mantissa = num_bit -  num_frac # these are also vectors
        #             scale = torch.pow(2, num_frac)
        #             threshold_clamp = torch.pow(2, num_bit_mantissa-1)
        #             threshold_up = torch.pow(2, threshold_clamp)
        #             threshold_down = torch.pow(2, -(threshold_clamp))
        #             # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
        #             if len(output.shape) == 3: # 3D
        #                 clamped_output = torch.clamp(torch.abs(output), min=threshold_down.unsqueeze(2), max=threshold_up.unsqueeze(2))
        #             elif len(output.shape) == 2: # 2D
        #                 clamped_output = torch.clamp(torch.abs(output), min=threshold_down.unsqueeze(1), max=threshold_up.unsqueeze(1))
        #             else:
        #                 print("Out of shape")
        #             output = torch.where(output<0, -clamped_output, clamped_output)
        #             if len(output.shape) == 3: # 3D
        #                 output = (torch.round(output*scale.unsqueeze(2)))/scale.unsqueeze(2)
        #             elif len(output.shape) == 2: # 2D
        #                 output = (torch.round(output*scale.unsqueeze(1)))/scale.unsqueeze(1)
        #             else:
        #                 print("Out of shape")
        #             # Then scale back
        #             if len(output.shape) == 3: # 3D
        #                 output = output*max_val_c.unsqueeze(1)
        #             elif len(output.shape) == 2: # 2D
        #                 output = output*max_val_c.unsqueeze(0)
        #             else:
        #                 print("Out of shape")
        #             return output
        #             # if len(output.shape) == 3: # 3D
        #             #     max_val = torch.max(log_x, dim=1).values.unsqueeze(1).expand_as(log_x)
        #             # elif len(output.shape) == 2: # 2D
        #             #     max_val = torch.max(log_x, dim=0).values.unsqueeze(0).expand_as(log_x)
        #             # else:
        #             #     print("Out of shape")

        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         grad_input = grad_output.clone()
        #         return grad_input

        # def activation_hook(module, input, output):
        #     output = STEFunction_structured.apply(output)
        #     return output

        # EXCLUDED_ACTIVATIONS = (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.PReLU)

        # for name, module in self.model.named_modules():
        #     if not isinstance(module, nn.ModuleList) and not list(module.children()) and "intermediate_act_fn" not in name and not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Dropout) and not any(isinstance(module, activation) for activation in EXCLUDED_ACTIVATIONS):
        #         module.register_forward_hook(activation_hook)
        # # # # PH: end

        # # PH: start (LLM.int8())
        # # Find columns that have at least an element larger than a threshold (6.0) and then keep it in high precision. Process and store in low-precision for the rest (row-wise activation and column-wise weights like zeroquant)
        # num_bit = 8
        # threshold = 0.01 # for outliers

        # # For keeping track of activations:
        # # class ReferenceCounter:
        # #     def __init__(self):
        # #         self.count = 0
        # #     def increase(self):
        # #         self.count += 1
        # #     def get_count(self):
        # #         return self.count

        # # counter = ReferenceCounter()
        # # list_output_activation = {}

        # class STEFunction_structured(torch.autograd.Function):
        #     @staticmethod
        #     def forward(ctx, input):
        #         # ctx.save_for_backward(input.clone()) # if you want to use input during backward calculation
        #         if isinstance(input, tuple):
        #             output = tuple(t.clone() for t in input)
        #             # Keep high precision for columns with elements greater than 6.0
        #             # output = tuple(torch.where(torch.any(torch.abs(t) > threshold, dim=1, keepdim=True), t, (torch.round(torch.where(t < 0, -torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=1)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=1)[0]))), min=0, max=num_bit)-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=1)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=1)[0]))), min=0, max=num_bit)-1)).unsqueeze(1)), torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=1)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=1)[0]))), min=0, max=num_bit)-1))).unsqueeze(1), max=torch.pow(2, torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=1)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=1)[0]))), min=0, max=num_bit)-1)).unsqueeze(1))) * torch.pow(2, torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=1)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=1)[0]))), min=0, max=num_bit)).unsqueeze(1))) / torch.pow(2, torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=1)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=1)[0]))), min=0, max=num_bit)).unsqueeze(1)) for t in output)
        #             # output = tuple(torch.where(torch.any(torch.abs(t) > threshold, dim=1, keepdim=True), t, (torch.round(torch.where(t < 0, -torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=2)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=2)[0]))), min=0, max=num_bit)-1))).unsqueeze(2), max=torch.pow(2, torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=2)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=2)[0]))), min=0, max=num_bit)-1)).unsqueeze(2)),  torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=2)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=2)[0]))), min=0, max=num_bit)-1))).unsqueeze(2), max=torch.pow(2, torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=2)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=2)[0]))), min=0, max=num_bit)-1)).unsqueeze(2))) * torch.pow(2, torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=2)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=2)[0]))), min=0, max=num_bit)).unsqueeze(2))) / torch.pow(2, torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=2)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=2)[0]))), min=0, max=num_bit)).unsqueeze(2)) for t in output)
        #             # output = tuple(torch.where(torch.any(torch.abs(t) > threshold, dim=1, keepdim=True), t, (torch.round(torch.where(t == 0, torch.tensor(0.0), torch.where(t < 0, -torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=2)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=2)[0]))), min=0, max=num_bit)-1))).unsqueeze(2), max=torch.pow(2, torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=2)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=2)[0]))), min=0, max=num_bit)-1)).unsqueeze(2)), torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=2)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=2)[0]))), min=0, max=num_bit)-1))).unsqueeze(2), max=torch.pow(2, torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=2)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=2)[0]))), min=0, max=num_bit)-1)).unsqueeze(2)))) * torch.pow(2, torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=2)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=2)[0]))), min=0, max=num_bit)).unsqueeze(2))) / torch.pow(2, torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=2)[0]==0, torch.tensor(0.0001), torch.max(torch.abs(t), dim=2)[0]))), min=0, max=num_bit)).unsqueeze(2)) for t in output)
        #             output = tuple(torch.where(torch.any(torch.abs(t) > threshold, dim=2, keepdim=True), t, (torch.round(torch.where(t == 0, torch.tensor(0.0), torch.where(t < 0, -torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=3)[0]==0, torch.tensor(0.000001), torch.max(torch.abs(t), dim=3)[0]))), min=0, max=num_bit)-1))).unsqueeze(3), max=torch.pow(2, torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=3)[0]==0, torch.tensor(0.000001), torch.max(torch.abs(t), dim=3)[0]))), min=0, max=num_bit)-1)).unsqueeze(3)), torch.clamp(torch.abs(t), min=torch.pow(2, -(torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=3)[0]==0, torch.tensor(0.000001), torch.max(torch.abs(t), dim=3)[0]))), min=0, max=num_bit)-1))).unsqueeze(3), max=torch.pow(2, torch.pow(2, num_bit - torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=3)[0]==0, torch.tensor(0.000001), torch.max(torch.abs(t), dim=3)[0]))), min=0, max=num_bit)-1)).unsqueeze(3)))) * torch.pow(2, torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=3)[0]==0, torch.tensor(0.000001), torch.max(torch.abs(t), dim=3)[0]))), min=0, max=num_bit)).unsqueeze(3))) / torch.pow(2, torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/torch.where(torch.max(torch.abs(t), dim=3)[0]==0, torch.tensor(0.000001), torch.max(torch.abs(t), dim=3)[0]))), min=0, max=num_bit)).unsqueeze(3)) for t in output)
        #             return output
        #         else:
        #             # IMPORTANT: INCREASE EACH DIM BY ONE. ALSO, INCREASE THE ARGUMENT OF UNSQUEEZE() BY TWO 
        #             output = input.clone()
        #             # max_values = torch.max(torch.abs(output), dim=1)[0]
        #             max_values = torch.max(torch.abs(output), dim=1)[0]
        #             max_values = torch.where(max_values==0, torch.tensor(0.0001), max_values) # VERY IMPORTANT: YOU NEED TO REPLACE ZEROS WITH ONES JUST IN CASE IF THE MAX WAS ZERO WHICH LEADS TO NAN
        #             # Identify columns with at least one element greater than 6.0
        #             mask_high_precision = torch.any(torch.abs(output) > threshold, dim=0, keepdim=True)
        #             # Quantization for columns without elements greater than 6.0
        #             # num_frac = torch.floor(torch.log2((2**(num_bit-1) - 1)/max_values))
        #             num_frac = torch.clamp(torch.floor(torch.log2((2**(num_bit-1) - 1)/max_values)), min=0, max=num_bit)
        #             num_bit_mantissa = num_bit - num_frac
        #             scale = torch.pow(2, num_frac)
        #             threshold_clamp = torch.pow(2, num_bit_mantissa-1)
        #             threshold_up = torch.pow(2, threshold_clamp)
        #             threshold_down = torch.pow(2, -(threshold_clamp))
        #             # clamped_output = torch.clamp(torch.abs(output), min=threshold_down.unsqueeze(1), max=threshold_up.unsqueeze(1))
        #             clamped_output = torch.clamp(torch.abs(output), min=threshold_down.unsqueeze(1), max=threshold_up.unsqueeze(1))
        #             output_q = torch.where(output < 0, -clamped_output, clamped_output)
        #             output_q = torch.where(output == 0, torch.tensor(0.0), output_q)
        #             # output_q = (torch.round(output_q * scale.unsqueeze(1))) / scale.unsqueeze(1)
        #             output_q = (torch.round(output_q * scale.unsqueeze(1))) / scale.unsqueeze(1)
        #             # Keep high precision for columns with elements greater than 6.0
        #             output_q = torch.where(mask_high_precision, output, output_q)
        #             return output_q

        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         grad_input = grad_output.clone()
        #         return grad_input

        # def activation_hook(module, input, output):
        #     output = STEFunction_structured.apply(output)
        #     return output

        # EXCLUDED_ACTIVATIONS = (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.PReLU)
        # for name, module in self.model.named_modules():
        #     if not isinstance(module, nn.ModuleList) and not list(module.children()) and "intermediate_act_fn" not in name and not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Dropout) and not any(isinstance(module, activation) for activation in EXCLUDED_ACTIVATIONS):
        #         module.register_forward_hook(activation_hook)
        # # # PH: end

        # PH: start (float4)
        # # convert to float8 (and any custom float)
        # # DONT FORGET TO FIRST QUANTIZE WEIGHTS TO LNS16
        # # IN YOUR HOOK, YOU HAVE TO USE .CLONE OF THE ARGUMNETS AND THEN MODIFY IT AND FINALLY RETURN IT. IT DOESNT WORK WITHOUT CLONE (IT HAS TO BE OUT OF PLACE COMPUTATION, NOT IN-PLACE)

        # # Create a 32-bit float tensor 3 bit mantissa, 4 bit exponent
        # num_bit_exponent = 2
        # num_bit_mantissa  = 1
        # offset = torch.tensor(2**(num_bit_exponent-1))
        # scale = torch.tensor(2 ** num_bit_mantissa)
        # threshold_clamp = 2**(num_bit_exponent-1)
        # threshold_up = float(2**threshold_clamp)
        # threshold_down = float(2**-(threshold_clamp))

        # # float32_tensor = torch.tensor(3.14159, dtype=torch.float32)

        # # # Extract sign, exponent, and mantissa bits from the 32-bit float
        # # # sign_bit = float32_tensor.sign()
        # # exponent_bits = torch.floor(torch.log2(torch.abs(float32_tensor))) + offset
        # # exponent = torch.pow(2, (exponent_bits - offset))
        # # mantissa_bits = torch.round(((float32_tensor / exponent) - 1) * scale)
        # # apx_float = ((mantissa_bits/scale) + 1) * exponent

        # # For keeping track of activations:
        # # class ReferenceCounter:
        # #     def __init__(self):
        # #         self.count = 0
        # #     def increase(self):
        # #         self.count += 1
        # #     def get_count(self):
        # #         return self.count

        # # counter = ReferenceCounter()
        # # list_output_activation = {}

        # class STEFunction_structured(torch.autograd.Function):
        #     """ define straight through estimator with overrided gradient (gate) """
        #     @staticmethod
        #     def forward(ctx, input):
        #         # ctx.save_for_backward(input.clone()) # if you want to use input during backward calculation
        #         # output = input.clone()
        #         if isinstance(input, tuple):
        #             # Clone each tensor in the tuple
        #             output = tuple(t.clone() for t in input)
        #             output = tuple(torch.where(t < 0, -torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up), torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up)) for t in output)
        #             output = tuple(((torch.round(((t / torch.pow(2, (torch.floor(torch.log2(torch.abs(t)))))) - 1) * scale)/scale) + 1) * torch.pow(2, (torch.floor(torch.log2(torch.abs(t))))) for t in output)
        #             return output                
        #         else:
        #             # If input is not a tuple, clone it
        #             output = input.clone()
        #             # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
        #             clamped_output = torch.clamp(torch.abs(output), min=threshold_down, max=threshold_up)
        #             output = torch.where(output<0, -clamped_output, clamped_output)

        #             exponent_bits = torch.floor(torch.log2(torch.abs(output))) + offset
        #             exponent = torch.pow(2, (exponent_bits - offset))
        #             mantissa_bits = torch.round(((output / exponent) - 1) * scale)
        #             output = ((mantissa_bits/scale) + 1) * exponent
        #             return output

        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         # # aux1 = ctx.saved_tensors # if you want to use input during backward calculation
        #         grad_input = grad_output.clone()
        #         return grad_input
        #         # # aux1 = ctx.saved_tensors # if you want to use input during backward calculation
        #         # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
        #         # grad_input = grad_output.clone()
        #         # clamped_output = torch.clamp(torch.abs(grad_input), min=threshold_down, max=threshold_up)
        #         # grad_input = torch.where(grad_input<0, -clamped_output, clamped_output)

        #         # exponent_bits = torch.floor(torch.log2(torch.abs(grad_input))) + offset
        #         # exponent = torch.pow(2, (exponent_bits - offset))
        #         # mantissa_bits = torch.round(((grad_input / exponent) - 1) * scale)
        #         # grad_input = ((mantissa_bits/scale) + 1) * exponent
        #         # return grad_input

        # def activation_hook(module, input, output):
        #     output = STEFunction_structured.apply(output)
        #     # for keeping track of activations
        #     # list_output_activation[str(module.__class__.__name__)+str("_")+str(counter.get_count())] = output #$$$
        #     # counter.increase() #$$$
        #     return output

        # EXCLUDED_ACTIVATIONS = (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.PReLU)

        # for name, module in self.model.named_modules():
        #     if not isinstance(module, nn.ModuleList) and not list(module.children()) and "intermediate_act_fn" not in name and not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Dropout) and not any(isinstance(module, activation) for activation in EXCLUDED_ACTIVATIONS):
        #         module.register_forward_hook(activation_hook)
        # model.model.layers[0].self_attn.q_proj.register_forward_hook(activation_hook)
        # # # PH: end

        # # # PH: start (LNS4)
        # num_bit_mantissa = 2 # for 8 bit repr.
        # # num_frac = 10 # fractional bits for 16 bit repr.
        # num_frac = 1 # fractional bits for 8 bit repr.

        # scale = float(2**num_frac)
        # threshold_clamp = 2**(num_bit_mantissa-1)
        # threshold_up = float(2**threshold_clamp)
        # threshold_down = float(2**-(threshold_clamp))

        # # For keeping track of activations:
        # # class ReferenceCounter:
        # #     def __init__(self):
        # #         self.count = 0
        # #     def increase(self):
        # #         self.count += 1
        # #     def get_count(self):
        # #         return self.count

        # # counter = ReferenceCounter()
        # # list_output_activation = {}

        # class STEFunction_structured(torch.autograd.Function):
        #     """ define straight through estimator with overrided gradient (gate) """
        #     @staticmethod
        #     def forward(ctx, input):
        #         # ctx.save_for_backward(input.clone()) # if you want to use input during backward calculation
        #         if isinstance(input, tuple):
        #             # Clone each tensor in the tuple
        #             output = tuple(t.clone() for t in input)
        #             output = tuple(torch.where(t<0, -torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up), torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up)) for t in output)
        #             output = tuple(torch.where(t > 0, torch.pow(2,(torch.round(torch.log2(t)*scale))/scale), torch.where(t < 0, -torch.pow(2,(torch.round(torch.log2(-t)*scale)/scale)), t)) for t in output)
        #             return output             
        #         else:
        #             output = input.clone()
        #             # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
        #             clamped_output = torch.clamp(torch.abs(output), min=threshold_down, max=threshold_up)
        #             output = torch.where(output<0, -clamped_output, clamped_output)
        #             # v1: concise
        #             output = torch.where(output > 0, torch.pow(2,(torch.round(torch.log2(output)*scale))/scale), torch.where(output < 0, -torch.pow(2,(torch.round(torch.log2(-output)*scale)/scale)), output))
        #             return output

        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         grad_input = grad_output.clone()
        #         return grad_input

        # def activation_hook(module, input, output):
        #     output = STEFunction_structured.apply(output)
        #     return output

        # EXCLUDED_ACTIVATIONS = (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.PReLU)

        # for name, module in self.model.named_modules():
        #     if not isinstance(module, nn.ModuleList) and not list(module.children()) and "intermediate_act_fn" not in name and not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Dropout) and not any(isinstance(module, activation) for activation in EXCLUDED_ACTIVATIONS):
        #         module.register_forward_hook(activation_hook)
        # # PH: end

        # # # # PH: start (modified LNS4 without pervector quant optimization) ---- BASE 4 logarithm
        # # num_bit_mantissa = 5 # for 16 bit repr.
        # num_bit_mantissa = 3 # for 8 bit repr.
        # threshold_mantissa = 2**(num_bit_mantissa-1)
        # threshold_up = float(4**threshold_mantissa)
        # threshold_down = float(4**-(threshold_mantissa))

        # # new version:
        # # max_num_bit_mantissa_needed = 1 # according to the distribution you can get this number
        # # log_domain_threshold = 2** max_num_bit_mantissa_needed # 4
        # # real_domain_threshold_up = float(2**log_domain_threshold) # 16
        # # real_domain_threshold_down = float(2**(-log_domain_threshold)) # 1/16

        # # num_frac_low_prec = 10 # number of fractional bits for 16 bit repr.
        # num_frac_low_prec = 0 # number of fractional bits for 8 bit repr.
        # # num_frac_high_prec = num_frac_low_prec + (num_bit_mantissa-max_num_bit_mantissa_needed) # 13
        # num_frac_high_prec = num_frac_low_prec + 1 # 13
        # scale_low_prec = 4**(num_frac_low_prec)
        # scale_high_prec = 4**(num_frac_high_prec)
        # # v3:
        # num_frac_highest_prec = num_frac_high_prec + 4 # for extreme outliers
        # scale_highest_prec = 4**(num_frac_highest_prec)

        # # For keeping track of activations:
        # # class ReferenceCounter:
        # #     def __init__(self):
        # #         self.count = 0
        # #     def increase(self):
        # #         self.count += 1
        # #     def get_count(self):
        # #         return self.count

        # # counter = ReferenceCounter()
        # # list_output_activation = {}

        # class STEFunction_structured(torch.autograd.Function):
        #     """ define straight through estimator with overrided gradient (gate) """
        #     @staticmethod
        #     def forward(ctx, input):
        #         # ctx.save_for_backward(input.clone()) # if you want to use input during backward calculation
        #         if isinstance(input, tuple):
        #             output = tuple(t.clone() for t in input)
        #             output = tuple(torch.where(t<0, -torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up), torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up)) for t in output)
        #             # output = tuple(torch.where(t > 0, torch.pow(2, torch.where(torch.log2(t)>torch.max(torch.log2(t))-5, torch.where(torch.log2(t)>torch.max(torch.log2(t))-3, torch.round(torch.log2(t) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.log2(t) * scale_high_prec)/ scale_high_prec), torch.round(torch.log2(t) * scale_low_prec)/ scale_low_prec)), torch.where(t < 0, -torch.pow(2, torch.where(torch.log2(-t)>torch.max(torch.log2(-t))-5, torch.where(torch.log2(-t)>torch.max(torch.log2(-t))-3, torch.round(torch.log2(-t) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.log2(-t) * scale_high_prec)/ scale_high_prec), torch.round(torch.log2(-t) * scale_low_prec)/ scale_low_prec)), t)) for t in output)
        #             # output = tuple(torch.where(t > 0, torch.pow(4, torch.where((torch.log2(t)/2)>torch.max((torch.log2(t)/2))-4, torch.where((torch.log2(t)/2)>torch.max((torch.log2(t)/2))-3, torch.round((torch.log2(t)/2) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(t)/2) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(t)/2) * scale_low_prec)/ scale_low_prec)), torch.where(t < 0, -torch.pow(4, torch.where((torch.log2(-t)/2)>torch.max((torch.log2(-t)/2))-4, torch.where((torch.log2(-t)/2)>torch.max((torch.log2(-t)/2))-3, torch.round((torch.log2(-t)/2) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(-t)/2) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(-t)/2) * scale_low_prec)/ scale_low_prec)), t)) for t in output)                    
        #             # output = tuple(torch.where(t > 0, torch.pow(2, torch.where((torch.log2(t))>torch.max((torch.log2(t)))-4, torch.where((torch.log2(t))>torch.max((torch.log2(t)))-3, torch.round((torch.log2(t)) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(t)) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(t)) * scale_low_prec)/ scale_low_prec)), torch.where(t < 0, -torch.pow(2, torch.where((torch.log2(-t))>torch.max((torch.log2(-t)))-4, torch.where((torch.log2(-t))>torch.max((torch.log2(-t)))-3, torch.round((torch.log2(-t)) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(-t)) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(-t)) * scale_low_prec)/ scale_low_prec)), t)) for t in output)                    
        #             output = tuple(torch.where(t<0, -(torch.pow(4, torch.where(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))))-5, torch.where(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))))-3, torch.round(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))) * scale_high_prec)/ scale_high_prec), torch.round(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))) * scale_low_prec)/ scale_low_prec))), torch.where(t>0, torch.pow(4, torch.where(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))))-5, torch.where(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))))-3, torch.round(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))) * scale_high_prec)/ scale_high_prec), torch.round(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))) * scale_low_prec)/ scale_low_prec)), t)) for t in output)
        #             return output
        #         else:
        #             output = input.clone()
        #             # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
        #             clamped_output = torch.clamp(torch.abs(output), min=threshold_down, max=threshold_up)
        #             output = torch.where(output<0, -clamped_output, clamped_output)
        #             # v3:
        #             log_x = torch.where(output<0, torch.log2(-output)/2, torch.where(output > 0, torch.log2(output)/2, torch.tensor(-64000.0)))
        #             quant_exponent_low_prec = torch.round(log_x * scale_low_prec)/ scale_low_prec # 2**3 - round(+ 0.5)
        #             quant_exponent_high_prec = torch.round(log_x * scale_high_prec)/ scale_high_prec # 2**3 - round(+ 0.5)
        #             quant_exponent_highest_prec = torch.round(log_x * scale_highest_prec)/ scale_highest_prec # 2**3 - round(+ 0.5)
        #             max_val = torch.max(log_x)
        #             quant_exponent = torch.where(log_x>max_val-5, torch.where(log_x>max_val-3, quant_exponent_highest_prec, quant_exponent_high_prec), quant_exponent_low_prec) # max_val-3 and max_val-5 are thresholds for extreme and moderate outliers (beta nd gamma)
        #             output = torch.where(output<0, -(torch.pow(4, quant_exponent)), torch.where(output>0, torch.pow(4, quant_exponent), output))

        #             # output = input.clone()
        #             # # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
        #             # clamped_output = torch.clamp(torch.abs(output), min=threshold_down, max=threshold_up)
        #             # output = torch.where(output<0, -clamped_output, clamped_output)

        #             # # v3:
        #             # if len(output.shape) == 3: # 3D
        #             #   non_zero_indices = output.nonzero()
        #             #   non_zero_values = output[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]] # 0 because the first dimension is batch, 1 b/c next one is first dimension of feature, 2 b/c it is second dimension of features
        #             #   # if 1D: non_zero_indices
        #             #   if len(non_zero_values) > 0: # any nonzero avail
        #             #     log_x = torch.where(non_zero_values > 0, torch.log2(non_zero_values), torch.log2(-non_zero_values))
        #             #     quant_exponent_low_prec = torch.round(log_x * scale_low_prec)/ scale_low_prec # 2**3 - round(+ 0.5)
        #             #     quant_exponent_high_prec = torch.round(log_x * scale_high_prec)/ scale_high_prec # 2**3 - round(+ 0.5)
        #             #     # --------  v3 (including extreme outliers) ---------
        #             #     quant_exponent_highest_prec = torch.round(log_x * scale_highest_prec)/ scale_highest_prec # 2**3 - round(+ 0.5)
        #             #     max_val = torch.max(log_x)
        #             #     quant_exponent = torch.where(log_x>max_val-5, torch.where(log_x>max_val-3, quant_exponent_highest_prec, quant_exponent_high_prec), quant_exponent_low_prec) # max_val-3 and max_val-5 are thresholds for extreme and moderate outliers (beta nd gamma)
        #             #     # ------- end v3 ---------
        #             #     quantized_values = torch.where(non_zero_values > 0, torch.pow(2, quant_exponent), -(torch.pow(2, quant_exponent)))
        #             #     output[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]] = quantized_values
        #             # elif len(output.shape) == 2: # 2D
        #             #   non_zero_indices = output.nonzero()
        #             #   non_zero_values = output[non_zero_indices[:, 0], non_zero_indices[:, 1]] # 0 because the first dimension is batch, 1 b/c next one is first dimension of feature, 2 b/c it is second dimension of features
        #             #   if len(non_zero_values) > 0:
        #             #     log_x = torch.where(non_zero_values > 0, torch.log2(non_zero_values), torch.log2(-non_zero_values))
        #             #     quant_exponent_low_prec = torch.round(log_x * scale_low_prec) / scale_low_prec # 2**3 - round(+ 0.5)
        #             #     quant_exponent_high_prec = torch.round(log_x * scale_high_prec) / scale_high_prec # 2**3 - round(+ 0.5)
        #             #     # --------  v3 (including extreme outliers) ---------
        #             #     quant_exponent_highest_prec = torch.round(log_x * scale_highest_prec)/ scale_highest_prec # 2**3 - round(+ 0.5)
        #             #     max_val = torch.max(log_x)
        #             #     quant_exponent = torch.where(log_x>max_val-5, torch.where(log_x>max_val-3, quant_exponent_highest_prec, quant_exponent_high_prec), quant_exponent_low_prec) # max_val-3 and max_val-5 are thresholds for extreme and moderate outliers (beta nd gamma)
        #             #     # ------- end v3 ---------
        #             #     quantized_values = torch.where(non_zero_values > 0, torch.pow(2, quant_exponent), -(torch.pow(2, quant_exponent)))
        #             #     output[non_zero_indices[:, 0], non_zero_indices[:, 1]] = quantized_values
        #             # else:
        #             #   print("Out of shape")
        #             return output

        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         # aux1 = ctx.saved_tensors # if you want to use input during backward calculation
        #         grad_input = grad_output.clone()
        #         return grad_input

        # def activation_hook(module, input, output):
        #     output = STEFunction_structured.apply(output)
        #     return output

        # EXCLUDED_ACTIVATIONS = (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.PReLU)

        # for name, module in self.model.named_modules():
        #     if not isinstance(module, nn.ModuleList) and not list(module.children()) and "intermediate_act_fn" not in name and not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Dropout) and not any(isinstance(module, activation) for activation in EXCLUDED_ACTIVATIONS):
                # module.register_forward_hook(activation_hook)
        # # PH: end

        # # # PH: start (modified LNS4) ---- BASE 4 logarithm
        # # num_bit_mantissa = 5 # for 16 bit repr.
        # num_bit_mantissa = 3 # for 8 bit repr.
        # threshold_mantissa = 2**(num_bit_mantissa-1)
        # threshold_up = float(4**threshold_mantissa)
        # threshold_down = float(4**-(threshold_mantissa))

        # # new version:
        # # max_num_bit_mantissa_needed = 1 # according to the distribution you can get this number
        # # log_domain_threshold = 2** max_num_bit_mantissa_needed # 4
        # # real_domain_threshold_up = float(2**log_domain_threshold) # 16
        # # real_domain_threshold_down = float(2**(-log_domain_threshold)) # 1/16

        # # num_frac_low_prec = 10 # number of fractional bits for 16 bit repr.
        # num_frac_low_prec = 0 # number of fractional bits for 8 bit repr.
        # # num_frac_high_prec = num_frac_low_prec + (num_bit_mantissa-max_num_bit_mantissa_needed) # 13
        # num_frac_high_prec = num_frac_low_prec + 1 # 13
        # scale_low_prec = 4**(num_frac_low_prec)
        # scale_high_prec = 4**(num_frac_high_prec)
        # # v3:
        # num_frac_highest_prec = num_frac_high_prec + 4 # for extreme outliers
        # scale_highest_prec = 4**(num_frac_highest_prec)

        # # For keeping track of activations:
        # # class ReferenceCounter:
        # #     def __init__(self):
        # #         self.count = 0
        # #     def increase(self):
        # #         self.count += 1
        # #     def get_count(self):
        # #         return self.count

        # # counter = ReferenceCounter()
        # # list_output_activation = {}

        # class STEFunction_structured(torch.autograd.Function):
        #     """ define straight through estimator with overrided gradient (gate) """
        #     @staticmethod
        #     def forward(ctx, input):
        #         ctx.save_for_backward(input.clone()) # if you want to use input during backward calculation
        #         if isinstance(input, tuple):
        #             output = tuple(t.clone() for t in input)
        #             output = tuple(torch.where(t<0, -torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up), torch.clamp(torch.abs(t), min=threshold_down, max=threshold_up)) for t in output)
        #             # output = tuple(torch.where(t > 0, torch.pow(2, torch.where(torch.log2(t)>torch.max(torch.log2(t))-5, torch.where(torch.log2(t)>torch.max(torch.log2(t))-3, torch.round(torch.log2(t) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.log2(t) * scale_high_prec)/ scale_high_prec), torch.round(torch.log2(t) * scale_low_prec)/ scale_low_prec)), torch.where(t < 0, -torch.pow(2, torch.where(torch.log2(-t)>torch.max(torch.log2(-t))-5, torch.where(torch.log2(-t)>torch.max(torch.log2(-t))-3, torch.round(torch.log2(-t) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.log2(-t) * scale_high_prec)/ scale_high_prec), torch.round(torch.log2(-t) * scale_low_prec)/ scale_low_prec)), t)) for t in output)
        #             # output = tuple(torch.where(t > 0, torch.pow(4, torch.where((torch.log2(t)/2)>torch.max((torch.log2(t)/2))-4, torch.where((torch.log2(t)/2)>torch.max((torch.log2(t)/2))-3, torch.round((torch.log2(t)/2) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(t)/2) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(t)/2) * scale_low_prec)/ scale_low_prec)), torch.where(t < 0, -torch.pow(4, torch.where((torch.log2(-t)/2)>torch.max((torch.log2(-t)/2))-4, torch.where((torch.log2(-t)/2)>torch.max((torch.log2(-t)/2))-3, torch.round((torch.log2(-t)/2) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(-t)/2) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(-t)/2) * scale_low_prec)/ scale_low_prec)), t)) for t in output)                    
        #             # output = tuple(torch.where(t > 0, torch.pow(2, torch.where((torch.log2(t))>torch.max(t, dim=0).values.unsqueeze(0).expand_as(t)-4, torch.where((torch.log2(t))>torch.max(t, dim=0).values.unsqueeze(0).expand_as(t)-3, torch.round((torch.log2(t)) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(t)) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(t)) * scale_low_prec)/ scale_low_prec)), torch.where(t < 0, -torch.pow(2, torch.where((torch.log2(-t))>torch.max(t, dim=0).values.unsqueeze(0).expand_as(t)-4, torch.where((torch.log2(-t))>torch.max(t, dim=0).values.unsqueeze(0).expand_as(t)-3, torch.round((torch.log2(-t)) * scale_highest_prec)/ scale_highest_prec, torch.round((torch.log2(-t)) * scale_high_prec)/ scale_high_prec), torch.round((torch.log2(-t)) * scale_low_prec)/ scale_low_prec)), t)) for t in output)              
        #             output = tuple(torch.where(t<0, -(torch.pow(4, torch.where(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))), dim=0).values.unsqueeze(0).expand_as(t)-5, torch.where(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))), dim=0).values.unsqueeze(0).expand_as(t)-3, torch.round(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))) * scale_high_prec)/ scale_high_prec), torch.round(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))) * scale_low_prec)/ scale_low_prec))), torch.where(t>0, torch.pow(4, torch.where(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))), dim=0).values.unsqueeze(0).expand_as(t)-5, torch.where(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0)))>torch.max(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))), dim=0).values.unsqueeze(0).expand_as(t)-3, torch.round(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))) * scale_highest_prec)/ scale_highest_prec, torch.round(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))) * scale_high_prec)/ scale_high_prec), torch.round(torch.where(t<0, torch.log2(-t)/2, torch.where(t > 0, torch.log2(t)/2, torch.tensor(-64000.0))) * scale_low_prec)/ scale_low_prec)), t)) for t in output)
        #             return output
        #         else:
        #             output = input.clone()
        #             # handling overflow/underflow (b/c of limited # of bits for mantissa) -> sparsify if less than a threshold and report an error message if larger thana threshold
        #             clamped_output = torch.clamp(torch.abs(output), min=threshold_down, max=threshold_up)
        #             output = torch.where(output<0, -clamped_output, clamped_output)
        #             # v3:
        #             log_x = torch.where(output<0, torch.log2(-output)/2, torch.where(output > 0, torch.log2(output)/2, torch.tensor(-64000.0)))
        #             quant_exponent_low_prec = torch.round(log_x * scale_low_prec)/ scale_low_prec # 2**3 - round(+ 0.5)
        #             quant_exponent_high_prec = torch.round(log_x * scale_high_prec)/ scale_high_prec # 2**3 - round(+ 0.5)
        #             quant_exponent_highest_prec = torch.round(log_x * scale_highest_prec)/ scale_highest_prec # 2**3 - round(+ 0.5)
        #             if len(output.shape) == 3: # 3D
        #                 max_val = torch.max(log_x, dim=1).values.unsqueeze(1).expand_as(log_x)
        #             elif len(output.shape) == 2: # 2D
        #                 max_val = torch.max(log_x, dim=0).values.unsqueeze(0).expand_as(log_x)
        #             else:
        #                 print("Out of shape")
        #             quant_exponent = torch.where(log_x>max_val-5, torch.where(log_x>max_val-3, quant_exponent_highest_prec, quant_exponent_high_prec), quant_exponent_low_prec) # max_val-3 and max_val-5 are thresholds for extreme and moderate outliers (beta nd gamma)
        #             output = torch.where(output<0, -(torch.pow(4, quant_exponent)), torch.where(output>0, torch.pow(4, quant_exponent), output))


        #             # v3:
        #             # if len(output.shape) == 3: # 3D
        #             #   non_zero_indices = output.nonzero()
        #             #   non_zero_values = output[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]] # 0 because the first dimension is batch, 1 b/c next one is first dimension of feature, 2 b/c it is second dimension of features
        #             #   # if 1D: non_zero_indices
        #             #   if len(non_zero_values) > 0: # any nonzero avail
        #             #     log_x = torch.where(non_zero_values > 0, torch.log2(non_zero_values), torch.log2(-non_zero_values))
        #             #     quant_exponent_low_prec = torch.round(log_x * scale_low_prec)/ scale_low_prec # 2**3 - round(+ 0.5)
        #             #     quant_exponent_high_prec = torch.round(log_x * scale_high_prec)/ scale_high_prec # 2**3 - round(+ 0.5)
        #             #     # --------  v3 (including extreme outliers) ---------
        #             #     quant_exponent_highest_prec = torch.round(log_x * scale_highest_prec)/ scale_highest_prec # 2**3 - round(+ 0.5)
        #             #     print(log_x.shape)
        #             #     max_val = torch.max(log_x, dim=1).values.unsqueeze(1).expand_as(log_x)
        #             #     quant_exponent = torch.where(log_x>max_val-5, torch.where(log_x>max_val-3, quant_exponent_highest_prec, quant_exponent_high_prec), quant_exponent_low_prec) # max_val-3 and max_val-5 are thresholds for extreme and moderate outliers (beta nd gamma)
        #             #     # ------- end v3 ---------
        #             #     quantized_values = torch.where(non_zero_values > 0, torch.pow(2, quant_exponent), -(torch.pow(2, quant_exponent)))
        #             #     output[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]] = quantized_values
        #             # elif len(output.shape) == 2: # 2D
        #             #   non_zero_indices = output.nonzero()
        #             #   non_zero_values = output[non_zero_indices[:, 0], non_zero_indices[:, 1]] # 0 because the first dimension is batch, 1 b/c next one is first dimension of feature, 2 b/c it is second dimension of features
        #             #   if len(non_zero_values) > 0:
        #             #     log_x = torch.where(non_zero_values > 0, torch.log2(non_zero_values), torch.log2(-non_zero_values))
        #             #     quant_exponent_low_prec = torch.round(log_x * scale_low_prec) / scale_low_prec # 2**3 - round(+ 0.5)
        #             #     quant_exponent_high_prec = torch.round(log_x * scale_high_prec) / scale_high_prec # 2**3 - round(+ 0.5)
        #             #     # --------  v3 (including extreme outliers) ---------
        #             #     quant_exponent_highest_prec = torch.round(log_x * scale_highest_prec)/ scale_highest_prec # 2**3 - round(+ 0.5)
        #             #     max_val = torch.max(log_x, dim=0).values.unsqueeze(0).expand_as(log_x)
        #             #     quant_exponent = torch.where(log_x>max_val-5, torch.where(log_x>max_val-3, quant_exponent_highest_prec, quant_exponent_high_prec), quant_exponent_low_prec) # max_val-3 and max_val-5 are thresholds for extreme and moderate outliers (beta nd gamma)
        #             #     # ------- end v3 ---------
        #             #     quantized_values = torch.where(non_zero_values > 0, torch.pow(2, quant_exponent), -(torch.pow(2, quant_exponent)))
        #             #     output[non_zero_indices[:, 0], non_zero_indices[:, 1]] = quantized_values
        #             # else:
        #             #   print("Out of shape")
        #             return output

        #     @staticmethod
        #     def backward(ctx, grad_output):
        #         # aux1 = ctx.saved_tensors # if you want to use input during backward calculation
        #         grad_input = grad_output.clone()
        #         return grad_input

        # def activation_hook(module, input, output):
        #     output = STEFunction_structured.apply(output)
        #     return output

        # EXCLUDED_ACTIVATIONS = (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.PReLU)

        # for name, module in self.model.named_modules():
        #     if not isinstance(module, nn.ModuleList) and not list(module.children()) and "intermediate_act_fn" not in name and not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Dropout) and not any(isinstance(module, activation) for activation in EXCLUDED_ACTIVATIONS):
        #         module.register_forward_hook(activation_hook)
        # PH: end

        self.model.eval()
        torch.set_grad_enabled(False)

        self._device = device
        if use_accelerate and "lm_head" in self.model.hf_device_map:
            # `accelerate` can place `lm_head` weights on a different device than
            # the user specified one so we force `self._device` to be the same as
            # `lm_head`'s.
            self._device = self.model.hf_device_map["lm_head"]
        if not use_accelerate and not (load_in_4bit or load_in_8bit):
            try:
                self.model.to(self._device)
            except:
                print(
                    "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes`. If the desired GPU is being used, this message is safe to ignore."
                )

    def _create_auto_model(
        self,
        *,
        pretrained: str,
        quantized: Optional[Union[bool, str]] = False,
        revision: str,
        subfolder: str,
        low_cpu_mem_usage: Optional[bool] = True,
        device_map: Optional[Union[str, _DeviceMapping]] = None,
        max_memory: Optional[dict] = None,
        offload_folder: Optional[str] = None,
        load_in_8bit: Optional[bool] = False,
        load_in_4bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        gptq_use_triton: Optional[bool] = False,
        inject_fused_attention: Optional[bool] = True,
        bnb_4bit_quant_type: Optional[str] = None,
        bnb_4bit_compute_dtype: Optional[Union[str, torch.dtype]] = None,
        bnb_4bit_use_double_quant: Optional[bool] = False,
    ) -> transformers.AutoModel:
        """Returns a pre-trained pytorch model from a pre-trained model configuration."""
        if not quantized:
            if load_in_4bit:
                assert (
                    transformers.__version__ >= "4.30.0"
                ), "load_in_4bit requires transformers >= 4.30.0"
            model_kwargs = {}
            if transformers.__version__ >= "4.30.0":
                model_kwargs["load_in_4bit"] = load_in_4bit
                if load_in_4bit:
                    if bnb_4bit_quant_type:
                        model_kwargs["bnb_4bit_quant_type"] = bnb_4bit_quant_type
                    if bnb_4bit_compute_dtype:
                        model_kwargs["bnb_4bit_compute_dtype"] = _get_dtype(
                            bnb_4bit_compute_dtype
                        )
                    if bnb_4bit_use_double_quant:
                        model_kwargs[
                            "bnb_4bit_use_double_quant"
                        ] = bnb_4bit_use_double_quant
            
            # PH: start (LLM.int8() performance)
            # print("Hi")
            # quantization_config = BitsAndBytesConfig(bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_threshold=6.0)
            
            # model = self.AUTO_MODEL_CLASS.from_pretrained(
            #     pretrained,
            #     revision=revision + ("/" + subfolder if subfolder is not None else ""),
            #     low_cpu_mem_usage=low_cpu_mem_usage,
            #     device_map="auto",
            #     max_memory=max_memory,
            #     offload_folder=offload_folder,
            #     load_in_8bit=True,
            #     trust_remote_code=trust_remote_code,
            #     # torch_dtype=torch_dtype,
            #     quantization_config=quantization_config,
            #     **model_kwargs,
            # )

            # print("Hi")

            model = self.AUTO_MODEL_CLASS.from_pretrained(
                pretrained,
                revision=revision + ("/" + subfolder if subfolder is not None else ""),
                low_cpu_mem_usage=low_cpu_mem_usage,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                load_in_8bit=load_in_8bit,
                # load_in_8bit=True,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                # torch_dtype=torch.float16,
                **model_kwargs,
            )
            # PH: end
        else:
            from auto_gptq import AutoGPTQForCausalLM

            model = AutoGPTQForCausalLM.from_quantized(
                pretrained,
                model_basename=None if quantized == True else Path(quantized).stem,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=trust_remote_code,
                use_safetensors=True
                if quantized == True
                else quantized.endswith(".safetensors"),
                use_triton=gptq_use_triton,
                warmup_triton=gptq_use_triton,
                inject_fused_attention=inject_fused_attention,
            )
        return model

    def _create_auto_model_peft(
        self,
        *,
        model: transformers.PreTrainedModel,
        peft: str,
        revision: str,
        subfolder: str,
        load_in_4bit: Optional[bool] = False,
    ):
        if load_in_4bit:
            assert PEFT_VERSION >= "0.4.0", "load_in_4bit requires peft >= 0.4.0"
        model = self.AUTO_PEFT_CLASS.from_pretrained(
            model,
            peft,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
        )
        return model

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
    ) -> transformers.PreTrainedTokenizer:
        """Returns a pre-trained tokenizer from a pre-trained tokenizer configuration."""
        tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            trust_remote_code=trust_remote_code,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForCausalLM:
            return False
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM:
            return True
        else:
            raise ValueError(
                "Could not determine `add_special_tokens` value from the model "
                "class. Set to `True` or `False` depending on whether the model "
                "was pre-trained with special tokens."
            )

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig, T5Config)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        if self._max_length is not None:
            return self._max_length
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self._config, attr):
                return getattr(self._config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def batch_size(self) -> int:
        # TODO: Add adaptive batch size.
        return self._batch_size  # * gpus

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    def tok_encode(self, string: str) -> TokenSequence:
        # TODO: Merge `tok_encode_batch` here.
        return self.tokenizer.encode(string, add_special_tokens=self.add_special_tokens)

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def greedy_until(
        self, requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)

        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        for chunk in utils.chunks(
            tqdm(reorder.get_reordered(), disable=False),
            self.batch_size if self.batch_size != "auto" else adaptive_batch_size,
        ):
            context = [c[0] for c in chunk]
            request_args = chunk[0][1]
            stop = request_args.get("until", None)
            stop_sequences = stop if isinstance(stop, list) else [stop]
            max_generation_length = request_args.get("max_length", None)

            assert (
                isinstance(max_generation_length, int) or max_generation_length is None
            )
            assert isinstance(stop_sequences, list) or stop_sequences is None

            # TODO: Find a better way to handle stop sequences for 0-shot.
            if stop_sequences is None:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length

            # PH: start
            # print("Bye1")
            # print("max_token:", max_tokens)
            # PH: end
            token_context = self.tok_encode_batch(context)

            responses = self._model_generate(
                inputs=token_context,
                max_tokens=max_tokens,
                stop=until,
            )
            responses = self.tok_decode(responses.tolist())

            for response in responses:
                # Ensure the generated responses do not contain the stop sequences.
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), response)
                results.append(response)
        return reorder.get_original(results)


class AutoCausalLM(HuggingFaceAutoLM):
    """Causal language modeling.
    You can find a set of supported models in the HF documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForCausalLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
    AUTO_PEFT_CLASS = peft.PeftModel

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
    ) -> transformers.PreTrainedTokenizer:
        tokenizer = super()._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
        )
        tokenizer.padding_side = "left"
        return tokenizer

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(inputs)["logits"]

    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> TokenSequence:
        # Ensure that the context does not encroach into the `space`
        # for the generation.
        input_ids = inputs["input_ids"][:, self.max_gen_toks - self.max_length :]
        attention_mask = inputs["attention_mask"][
            :, self.max_gen_toks - self.max_length :
        ]
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, input_ids.shape[1], input_ids.shape[0]
        )

        # PH: start
        # print("Bye2")
        # print("max_token:", max_tokens)
        # PH: end
        generations = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # GPT style models require the `generate` `max_length` arg to include the
            # context length, so we instead set `max_new_tokens` which is the number
            # of new tokens to generate, excluding the current number of tokens.
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return utils.select_continuation_from_batch_left_padding(
            generations, max_context_size=inputs["input_ids"].size(1)
        )


class AutoSeq2SeqLM(HuggingFaceAutoLM):
    """Seq2Seq language modeling.
    You can find a set of supported models in the following documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForSeq2SeqLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM
    AUTO_PEFT_CLASS = peft.PeftModel

    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        new_requests = []
        for chunk in utils.chunks(requests, self.batch_size):
            context, continuation = zip(*chunk)

            # Fill empty contexts with the EOT token.
            context = [
                f"{self.eot_token}" if len(text) == 0 else text for text in context
            ]
            context_enc = self.tok_encode_batch(context)
            for key in context_enc:
                context_enc[key] = context_enc[key][:, -self.max_length :]

            # Remove leading whitespace introduced by the default
            # `text_target_separator` since the context and continuation
            # will not be concatenated as a single (decoder) input.
            continuation = [text.lstrip() for text in continuation]
            continuation_enc = self.tok_encode_batch(list(continuation))
            for key in continuation_enc:
                continuation_enc[key] = continuation_enc[key][:, -self.max_length :]

            new_requests.append(
                ((context, continuation), context_enc, continuation_enc)
            )
        return self._loglikelihood_tokens(new_requests)

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        loglikelihoods = []
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            contexts, conts = utils.split_and_pad_windows(
                rolling_token_windows,
                pad_token_id=self.eot_token_id,
                max_seq_len=self.max_length,
            )
            # Manually create BatchEncoding tensors with attention masks as
            # expected by `self._model_call` in `self._loglikelihood_tokens`.
            contexts_enc = torch.Tensor(contexts).long()
            contexts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": contexts_enc,
                    "attention_mask": (contexts_enc != self.eot_token_id).long(),
                }
            )
            conts_enc = torch.Tensor(conts).long()
            conts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": conts_enc,
                    "attention_mask": (conts_enc != self.eot_token_id).long(),
                }
            )
            # TODO: Extract out this call so it only gets called once and also
            # somehow figure out partial caching for.
            rolling_token_windows_request = [
                ((contexts, conts), contexts_enc, conts_enc)
            ]
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows_request, disable_tqdm=True
            )
            string_nll = [x[0] for x in string_nll]  # discard is_greedy
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], TokenSequence, TokenSequence]],
        disable_tqdm: Optional[bool] = False,
    ) -> List[Tuple[float, bool]]:
        results = []
        for chunk in tqdm(
            requests, total=math.ceil(len(requests)), disable=disable_tqdm
        ):
            cache_keys, inputs_tokens, targets_tokens = chunk
            inputs_tokens = inputs_tokens.to(self.device)
            targets_tokens = targets_tokens.to(self.device)
            outputs = self._model_call(inputs=inputs_tokens, labels=targets_tokens)
            log_softmaxes = F.log_softmax(outputs.logits, dim=-1)

            output_iterator = zip(
                zip(cache_keys[0], cache_keys[1]),
                log_softmaxes,
                targets_tokens["input_ids"],
                targets_tokens["attention_mask"],
            )
            for cache_key, log_softmax, target_tokens, target_mask in output_iterator:
                length = target_mask.sum()
                log_softmax = log_softmax[:length]
                target_tokens = target_tokens[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tokens).all()
                target_logits = torch.gather(
                    log_softmax, 1, target_tokens.unsqueeze(-1)
                ).squeeze(-1)
                answer = (float(target_logits.sum()), bool(max_equal))
                results.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return results

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(**inputs, labels=labels["input_ids"])

    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> TokenSequence:
        input_ids = inputs["input_ids"][:, -self.max_length :].to(self.device)
        attention_mask = inputs["attention_mask"][:, -self.max_length :].to(self.device)

        # Generate one token to calculate the number of start tokens prepended to decoder_input_ids
        # (leaving this here in case the below assumption is violated in the future)
        # one_tok_gen = self.model.generate(
        #    input_ids=torch.zeros((1, 1), dtype=torch.int),
        #    min_length=2,
        #    max_new_tokens=1,
        # ).squeeze()
        # initial_decoder_input_length = len(one_tok_gen) - 1

        # Assume that there will always only be one token in the decoder inputs, assumption holds for existing HF models
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, 1, input_ids.shape[0]
        )

        # PH:
        # print("Bye3")
        # PH:

        generations = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return generations


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :][
            :, -self.sequence_id_len :
        ]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )