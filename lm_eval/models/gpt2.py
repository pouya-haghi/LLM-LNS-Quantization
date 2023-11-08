import torch
import transformers
from typing import Optional, Union
from lm_eval.base import BaseLM


def _get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class HFLM(BaseLM):

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        max_batch_size=512,
        max_length=None,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
    ):
        super().__init__()

        # Initialize model
        if isinstance(pretrained, transformers.PreTrainedModel):
            self.model = pretrained
            self._device = self.model.device

            if tokenizer:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
            else:
                # Get tokenizer
                model_name = self.model.name_or_path
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                )

        elif isinstance(pretrained, str):

            # Initialize device
            assert isinstance(device, str)
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                print(f"Using device '{device}'")
            else:
                print("Device not specified")
                print(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            revision = revision + ("/" + subfolder if subfolder is not None else "")


            # PH: start
            # Initialize new model and tokenizer instances
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained,
                load_in_8bit=load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage,
                revision=revision,
                torch_dtype=_get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            ).to(self.device)

            # self.model = transformers.AutoModelForCausalLM.from_pretrained(
            #     pretrained,
            #     load_in_8bit=load_in_8bit,
            #     low_cpu_mem_usage=low_cpu_mem_usage,
            #     revision=revision,
            #     torch_dtype=_get_dtype(dtype),
            #     trust_remote_code=trust_remote_code,
            # )

            # PH: end

            # # PH: start (pre-processing) finding # of times that inference is called.
            # For keeping track of activations:
            class ReferenceCounter:
                def __init__(self):
                    self.count = 0
                    self.count_shape = 0
                def increase(self):
                    self.count += 1
                def add_shape(self, num):
                    self.count_shape += num
                def get_count(self):
                    return self.count

            counter = ReferenceCounter()
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
                        print("count", counter.count)
                        return output                
                    else:
                        # If input is not a tuple, clone it
                        output = input.clone()
                        print("count", counter.count)
                        if len(output.shape) == 3: # 3D
                            counter.add_shape(output.shape[1])
                        elif len(output.shape) == 2: # 2D
                            counter.add_shape(output.shape[0])
                        else:
                            print("Out of shape")
                        print("shape", counter.count_shape)
                        print("avg", counter.count_shape/counter.count)
                        return output

                @staticmethod
                def backward(ctx, grad_output):
                    # # aux1 = ctx.saved_tensors # if you want to use input during backward calculation
                    grad_input = grad_output.clone()
                    return grad_input

            def activation_hook(module, input, output):
                counter.increase() #$$$
                output = STEFunction_structured.apply(output)
                # for keeping track of activations
                # list_output_activation[str(module.__class__.__name__)+str("_")+str(counter.get_count())] = output #$$$
                return output

            EXCLUDED_ACTIVATIONS = (nn.ReLU, nn.Tanh, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.PReLU)
            for name, module in self.model.named_modules():
                if not isinstance(module, nn.ModuleList) and not list(module.children()) and "intermediate_act_fn" not in name and not isinstance(module, nn.LayerNorm) and not isinstance(module, nn.Dropout) and not any(isinstance(module, activation) for activation in EXCLUDED_ACTIVATIONS):
                    module.register_forward_hook(activation_hook)
                    break
            # self.model.model.layers[0].self_attn.q_proj.register_forward_hook(activation_hook)
            # # # PH: end

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer if tokenizer else pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )

        else:
            raise TypeError(
                "Parameter pretrained should be of type str or transformers.PreTrainedModel"
            )

        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size

        # Validate batch_size
        assert isinstance(batch_size, (int, str))

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_length = max_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
            generation_kwargs[
                "pad_token_id"
            ] = eos_token_id  # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)


# for backwards compatibility
GPT2LM = HFLM
