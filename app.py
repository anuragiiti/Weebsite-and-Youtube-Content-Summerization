import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Streamlit App
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and URL (YT or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    # Optionally show a part of the key for validation
    if groq_api_key:
        st.write(f"Entered API Key: {groq_api_key[:4]}******")

generic_url = st.text_input("URL", label_visibility="collapsed")

## Prompt template for the summarization task
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

## Button to trigger the summarization process
if st.button("Summarize the Content from YT or Website"):
    
    # Validate inputs: check for API key and URL
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the API key and the URL to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video or a website URL.")
    else:
        try:
            # Initialize the Groq client only after the API key has been entered
            try:
                llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)
            except Exception as e:
                st.error(f"Error initializing Groq API Client: {e}")
                st.stop()  # Stop further execution if API client fails

            with st.spinner("Fetching and summarizing content..."):
                # Loading the content from the URL (YouTube or website)
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                docs = loader.load()

                # Chain for summarization using the LLM and prompt
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                # Display the summarized output
                st.success(output_summary)

        except Exception as e:
            st.exception(f"An error occurred: {e}")
