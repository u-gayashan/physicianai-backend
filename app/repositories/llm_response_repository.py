# llm_response_repository.py
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, TextStreamer, pipeline

import psycopg2
import pgvector
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from langchain.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import google.generativeai as genai
import googletrans
from googletrans import Translator

class LLMRepository:
    def __init__(self):
        self.sys = ""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}

        embeddings = HuggingFaceBgeEmbeddings(
            model_name="physician-ai/Model81", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        
        CONNECTION_STRING = "postgresql+psycopg2://postgres:ElonMusk123@physician-ai.cbtexq15uzag.us-east-1.rds.amazonaws.com:5432/vector_db" 

        store = PGVector(
            connection_string=CONNECTION_STRING,
            embedding_function=embeddings,
            collection_name= "Model 81"
        )
        # self.embeddings = HuggingFaceInstructEmbeddings(
        #     model_name="hkunlp/instructor-large", model_kwargs={"device": self.device}
        # )
        
        genai.configure(api_key="AIzaSyAjG_p_DA8rSsTNUt1w4zQ_7MIZ9ADqvqk")
        gemini_model = genai.GenerativeModel('gemini-pro')

        self.new_db = FAISS.load_local("../utils", embeddings)

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
            retriever=store.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )

        self.qa_chain_a = RetrievalQA.from_chain_type(
            llm=self.llm2,
            chain_type="stuff",
            retriever=store.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )
        
    def translate(text,language):
        return str(translator.translate(text,dest=keys[vals.index(language)]).text)
        
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

    def QA():
        print("\nopen QA mode running ========================================\n")

        try:
            print("\n google gemini===================\n")
            gemini_chat = gemini_model.start_chat(history=[])
            
            gemini_response = gemini_chat.send_message('give next small response for laste patient response like a doctor.  '+str(chain))
            return gemini_response.text
        except:
            print("\n llmmaa ===================\n")
            result_ex = qa_chain(sys+chain+"""\n\n\nalways give small and single response based on the patient 
                response. don't ask any question give simple response""")
            if "Patient:" in str(result_ex['result']) or "Patient response:" in str(result_ex['result']) or "Patient Response" in str(result_ex['result']):
                return str((str(result_ex['result']).split("\n\n")[-1]).split(":")[-1])
            else:
                try:
                   return str(result_ex['result']).split("\n\n")[1]
                except:
                   return str(result_ex['result']).split("\n\n")
        if str(mode)=="dirrect_QA" and id==3:
            diagnosis_and_treatment = qa_chain(sys+chain+end_sys_prompts)
            diagnosis_and_treatment = str(diagnosis_and_treatment['result'])
            print(diagnosis_and_treatment)
            print("dirrect answer")
            return {"english":diagnosis_and_treatment,"translated":translate(diagnosis_and_treatment,language)}
            
        if str(mode)=="dirrect_QA" and id>3:
            qa_text =  str(QA())
            return {"english":qa_text,"translated":translate(qa_text,language)}
    
    def refresh_model(self):
        self.llm = HuggingFacePipeline(pipeline=self.text_pipeline, model_kwargs={"temperature": 2})
        self.llm2 = HuggingFacePipeline(pipeline=self.text_pipeline, model_kwargs={"temperature": 2})

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=store.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )

        self.qa_chain_a = RetrievalQA.from_chain_type(
            llm=self.llm2,
            chain_type="stuff",
            retriever=store.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )

        print("Model refreshed")
        
    def get_response(self, chain, id, mode):
        if id<13:

            if id>=8:
                return {"english":final_question[id],"translated":translate(final_question[id],language)}
            else:
                if id<5:
                    # 2 . prompmt control the natural way on question asking based on patient response,symptomps type
                    question = qa_chain(sys+chain +"""\n\nask single small queston to get details based on the patient response,and don't ask 
                    same question again, and don't provide treatment and diagnosis ask next small and short question , 
                    always don't ask same question again and again , always only ask next single small question""")

                else:
                    # 3. prompt will guide the model to ask yes or no questions based on patient response,symptomps type
                    question = qa_chain(sys+chain +"""\n\nask single small queston to get details based on the patient response,and don't ask 
                    same question again, and don't provide treatment and diagnosis ask next small and short question with yes or no format , 
                    always don't ask same question again and again , always only ask next single small question""")
                try:
                    if "Patient:" in str(question['result']) or "Patient response:" in str(question['result']) or "Patient Response" in str(question['result']):

                        return {"english":str((str(question['result']).split("\n\n")[-1]).split(":")[-1]),"translated":translate(str((str(question['result']).split("\n\n")[-1]).split(":")[-1]),language)}
                    else:

                        return {"english":str(question['result']).split("\n\n")[1],"translated":translate(str(question['result']).split("\n\n")[1],language)}

                except:
                    if "Patient:" in str(question['result']) or "Patient response:" in str(question['result']) or "Patient Response" in str(question['result']):

                       return {"english":str(question['result']).split(":")[-1],"translated":translate(str(question['result']).split(":")[-1],language)}
                    else:

                       return {"english":str(question['result']),"translated":translate(str(question['result']),language)}

        if id==13:
            diagnosis_and_treatment = qa_chain(sys+chain+end_sys_prompts)
            diagnosis_and_treatment = str(diagnosis_and_treatment['result'])
            print(mode,diagnosis_and_treatment)
            report = qa_chain_a(report_prompt_template+sys+chain+"\n\ntreatment & diagnosis with ICD code below\n"+diagnosis_and_treatment)
            print(str(report['result']))
            print("h&P")
            return {"english":str(report['result']),"translated":translate(str(report['result']),language)}

        qa_text =  str(QA())        
        return {"english":qa_text,"translated":translate(qa_text,language)}
