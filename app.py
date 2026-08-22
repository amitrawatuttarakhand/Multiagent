import os
import sys

# Patch legacy pydantic v1 imports before importing CrewAI
import pydantic
if "pydantic.v1" not in sys.modules:
    sys.modules["pydantic.v1"] = pydantic

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import YoutubeChannelSearchTool

st.set_page_config(
    page_title="Multi-Agent YouTube Content Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 CrewAI Multi-Agent YouTube Blog Post Generator (OpenRouter)")

# Secret Management: Fetch from Streamlit Cloud Secrets or Sidebar
if "OPENROUTER_API_KEY" in st.secrets:
    openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
else:
    openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

model_name = st.sidebar.text_input(
    "OpenRouter Model", 
    value="openrouter/openai/gpt-4o-mini"
)
channel_handle = st.sidebar.text_input("YouTube Channel Handle", value="@krishnaik06")
topic_query = st.text_input("Enter Topic / Query to Research", value="AI vs ML vs Data Science")

if st.button("🚀 Run Crew Workflow"):
    if not openrouter_api_key:
        st.error("Please enter your OpenRouter API Key in the sidebar or Streamlit Secrets.")
        st.stop()

    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key

    with st.spinner("Executing Task..."):
        try:
            # 1. Instantiate OpenRouter LLM via CrewAI LLM Class
            llm = LLM(
                model=model_name,
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key
            )

            # 2. Configure YouTube Search Tool without requiring default OpenAI embeddings
            yt_tool = YoutubeChannelSearchTool(
                youtube_channel_handle=channel_handle,
                config=dict(
                    llm=dict(
                        provider="openrouter",
                        config=dict(
                            model=model_name,
                            api_key=openrouter_api_key,
                        ),
                    ),
                )
            )

            # 3. Create Agents with max_iter=2 and memory disabled
            blog_researcher = Agent(
                role="Blog Researcher from YouTube Videos",
                goal=f"Get relevant video content for '{topic_query}' from YT Channel.",
                verbose=True,
                memory=False,
                backstory="An expert in understanding videos in AI and data science.",
                tools=[yt_tool],
                allow_delegation=True,
                llm=llm,
                max_iter=2
            )

            blog_writer = Agent(
                role="Blog Writer",
                goal=f"Narrate compelling stories on '{topic_query}'.",
                verbose=True,
                memory=False,
                backstory="Crafts engaging narratives that simplify complex topics.",
                tools=[yt_tool],
                allow_delegation=False,
                llm=llm,
                max_iter=2
            )

            # 4. Define Tasks
            research_task = Task(
                description=f"Identify videos on '{topic_query}' and extract key content.",
                expected_output="Detailed summary report based on video content.",
                tools=[yt_tool],
                agent=blog_researcher
            )

            write_task = Task(
                description=f"Draft a markdown blog post on '{topic_query}'.",
                expected_output="A full blog post in markdown format.",
                tools=[yt_tool],
                agent=blog_writer,
                output_file="blog_post.md"
            )

            # 5. Assemble Crew with max_iter=2 and memory disabled
            crew = Crew(
                agents=[blog_researcher, blog_writer],
                tasks=[research_task, write_task],
                process=Process.sequential,
                verbose=True,
                memory=False,
                max_iter=2
            )

            result = crew.kickoff(inputs={"topic": topic_query})

            st.success("Workflow Executed Successfully!")
            st.markdown(result.raw)

            st.download_button(
                label="📥 Download Markdown Post",
                data=str(result.raw),
                file_name="generated_blog_post.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
