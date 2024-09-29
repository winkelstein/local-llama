from llama_cpp import Llama, CreateChatCompletionResponse
from typing import List, Dict


class LlamaModel:
    def __init__(
        self,
        model_path: str,
        use_gpu: bool = True,
        verbose: bool = False,
        context_window: int = 2048,
    ) -> None:
        self.use_gpu: bool = use_gpu
        self.context_window: int = context_window

        self.llm: Llama = Llama(
            model_path=model_path,
            n_gpu_layers=-1 if use_gpu else 0,
            verbose=verbose,
            n_ctx=context_window,
        )

        self.model_name: str = self.llm.metadata["general.name"]

    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        response: CreateChatCompletionResponse = self.llm.create_chat_completion(
            messages
        )
        response: str = response["choices"][0]["message"]["content"]
        return response


llama: LlamaModel = LlamaModel("./llama/model.gguf")
