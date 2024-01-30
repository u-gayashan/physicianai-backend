# llm_response_repository.py
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, TextStreamer, pipeline

class LLMRepository:
    def __init__(self):
        self.sys = ""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large", model_kwargs={"device": self.device}
        )
        
        self.new_db = FAISS.load_local("../utils/faiss_index", self.embeddings)

        self.model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
        self.model_basename = "model"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

        self.model = AutoGPTQForCausalLM.from_quantized(
            self.model_name_or_path,
            revision="gptq-4bit-128g-actorder_True",
            model_basename=self.model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device=self.device,
            inject_fused_attention=False,
            quantize_config=None,
        )

        self.DEFAULT_SYSTEM_PROMPT = """
        You are a helpful, respectful and honest assistant. give answer for any questions.
        """.strip()

        self.generate_prompt()

        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=4096,
            temperature=2,
            top_p=0.95,
            repetition_penalty=1.15,
            streamer=self.streamer,
        )

        self.llm = HuggingFacePipeline(pipeline=self.text_pipeline, model_kwargs={"temperature": 2})
        self.llm2 = HuggingFacePipeline(pipeline=self.text_pipeline, model_kwargs={"temperature": 2})

        self.SYSTEM_PROMPT = "give answer from external data's. don't use the provided context"

        self.template = self.generate_prompt(
            """
        {context}
        Question: {question}
        """,
            system_prompt=self.SYSTEM_PROMPT,
        )
        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.new_db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )

        self.qa_chain_a = RetrievalQA.from_chain_type(
            llm=self.llm2,
            chain_type="stuff",
            retriever=self.new_db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )

    def generate_prompt(self, prompt=None, system_prompt=None):
        if prompt is None:
            prompt = """
            {context}
            Question: {question}
            """

        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT

        return f"""
        [INST] <<SYS>>
        {system_prompt}
        <</SYS>>
        {prompt} [/INST]
        """.strip()

    def refresh_model(self):
        self.llm = HuggingFacePipeline(pipeline=self.text_pipeline, model_kwargs={"temperature": 2})
        self.llm2 = HuggingFacePipeline(pipeline=self.text_pipeline, model_kwargs={"temperature": 2})

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.new_db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )

        self.qa_chain_a = RetrievalQA.from_chain_type(
            llm=self.llm2,
            chain_type="stuff",
            retriever=self.new_db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )

        print("Model refreshed")
        
    def get_response(self, chain, id, mode):
        if id < 13:
            if id >= 8:
                return self.final_question[id]
            else:
                if id < 5:
                    question = self.qa_chain(
                        self.sys + chain + """
                    \n\nask single small queston to get details based on the patient response,and don't ask 
                    same question again, and don't provide treatment and diagnosis ask next small and short question , 
                    always don't ask same question again and again , always only ask next single small question"""
                    )
                else:
                    question = self.qa_chain(
                        self.sys + chain + """
                    \n\nask single small queston to get details based on the patient response,and don't ask 
                    same question again, and don't provide treatment and diagnosis ask next small and short question with yes or no format , 
                    always don't ask same question again and again , always only ask next single small question"""
                    )
                try:
                    if (
                        "Patient:" in str(question["result"])
                        or "Patient response:" in str(question["result"])
                        or "Patient Response" in str(question["result"])
                    ):
                        return str((str(question["result"]).split("\n\n")[-1]).split(":")[-1])
                    else:
                        return str(question["result"]).split("\n\n")[1]

                except:
                    if (
                        "Patient:" in str(question["result"])
                        or "Patient response:" in str(question["result"])
                        or "Patient Response" in str(question["result"])
                    ):
                        return str(question["result"]).split(":")[-1]
                    else:
                        return str(question["result"])

        if id == 13:
            end_sys_prompts = "\n\ngive correct treatment and most related diagnosis with ICD code don't ask any questions. if question is not related to provided data don't give answer from this provided data's"
            diagnosis_and_treatment = self.qa_chain(self.sys + chain + self.end_sys_prompts)
            diagnosis_and_treatment = str(diagnosis_and_treatment["result"])
            print(mode)
            if str(mode) == "dirrect":
                print(diagnosis_and_treatment)
                print("dirrect answer")
                return diagnosis_and_treatment

            else:
                report = self.qa_chain_a(
                    self.report_prompt_template
                    + self.sys
                    + chain
                    + "\n\ntreatment & diagnosis with ICD code below\n"
                    + diagnosis_and_treatment
                )
                print(str(report["result"]))
                print("h&P")
                return str(report["result"])

        result_ex = self.qa_chain(
            self.sys + chain + """\n\n\nalways give small and single response based on the patient 
            response. don't give multiline response always give response based on last patient response"""
        )
        if (
            "Patient:" in str(result_ex["result"])
            or "Patient response:" in str(result_ex["result"])
            or "Patient Response" in str(result_ex["result"])
        ):
            return str((str(result_ex["result"]).split("\n\n")[-1]).split(":")[-1])
        else:
            return str(result_ex["result"]).split("\n\n")[1]
