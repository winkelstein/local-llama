from llama_cpp import Llama
from typing import List, Dict


class LlamaModel:
    def __init__(
        self,
        model_path: str,
        use_gpu: bool = True,
        verbose: bool = False,
        context_window: int = 2048,
    ):
        self.use_gpu = use_gpu
        self.context_window = context_window

        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1 if use_gpu else 0,
            verbose=verbose,
            n_ctx=context_window,
        )

        self.model_name = self.llm.metadata["general.name"]

    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        response = self.llm.create_chat_completion(messages)
        response = response["choices"][0]["message"]["content"]
        return response


llama: LlamaModel = LlamaModel("./llama/model.gguf")
