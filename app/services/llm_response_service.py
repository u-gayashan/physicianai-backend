class LLMService:
    def __init__(self, llm_repository):
        self.llm_repository = llm_repository

    def get_response(self, chain, id, mode):
        return self.llm_repository.get_response(chain, id, mode)

