# class ResponseLogicHandler:
#     def __init__(self):
#         self.id_to_logic = {
#             8: self.final_question_logic,
#             13: self.id_13_logic,
#         }

#     def handle_logic(self, chain, id, mode):
#         logic_function = self.id_to_logic.get(id, self.default_logic)
#         return logic_function(chain, id, mode)

#     def final_question_logic(self, chain, id, mode):
#         # ...

#     def id_13_logic(self, chain, id, mode):
#         # ...

#     def default_logic(self, chain, id, mode):
#         # ...