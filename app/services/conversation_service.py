# from repositories import ConversationRepository
# from services.prompt_engineer_service import PromptEngineerService
# from services.language_model_service import LanguageModelService

# async def respond(text_input: str, language: str, conversation_history: dict = None):
#     conversation_history = conversation_history or await ConversationRepository.get_conversation_history(user_id)  # Get history if available
#     prompt = PromptEngineerService.generate_prompt(text_input, conversation_history, context)
#     response = await LanguageModelService.generate_response(prompt)
#     await ConversationRepository.save_conversation_history(user_id, response)  # Updateing history
#     return response