import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
import os

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

st.title("Youtube Script Genereator with Langchain")
prompt = st.text_input("Enter the Input")

title_template = PromptTemplate(input_variables = ['topic'], template = 'Write me a Youtube video title about {topic}')
script_template = PromptTemplate(input_variables = ['title', 'wikipedia_research'],
                                 template = 'Write me a Youtube script on the title {title} while leveraging the Wikipedia research {wikipedia_research}')

title_memory = ConversationBufferMemory(input_key = 'topic', memory_key = 'chat_history')
script_memory = ConversationBufferMemory(input_key = 'title', memory_key = 'chat_history')


llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl",
                    model_kwargs = {"temperature":0.9})
title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key = 'title', memory = title_memory)
script_chain = LLMChain(llm = llm, prompt = script_template, verbose = True, output_key = 'script', memory = script_memory)

wiki = WikipediaAPIWrapper()

if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title = title, wikipedia_research = wiki_research)
    st.write(title)
    st.write(script)
    with st.expander('Title History'):
        st.info(title_memory.buffer)
    with st.expander('Script History'):
        st.info(script_memory.buffer)
    with st.expander('Wikipedia Research'):
        st.info(wiki_research)