import os
import sys


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

openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password")
model_name = st.sidebar.text_input(
    "OpenRouter Model", 
    value="openrouter/openai/gpt-4o-mini"
)
channel_handle = st.sidebar.text_input("YouTube Channel Handle", value="@krishnaik06")
topic_query = st.text_input("Enter Topic / Query to Research", value="AI vs ML vs Data Science")

if st.button("🚀 Run Crew Workflow"):
    if not openrouter_api_key:
        st.error("Please enter your OpenRouter API Key in the sidebar.")
        st.stop()

    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key

    with st.spinner("Executing Task..."):
        try:
            llm = LLM(
                model=model_name,
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key
            )

            yt_tool = YoutubeChannelSearchTool(youtube_channel_handle=channel_handle)

            # memory=False prevents CrewAI from loading ChromaDB vector storage
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

            research_task = Task(
                description=f"Identify videos on '{topic_query}' and extract info.",
                expected_output="Summary report based on video content.",
                tools=[yt_tool],
                agent=blog_researcher
            )

            write_task = Task(
                description=f"Draft a markdown blog post on '{topic_query}'.",
                expected_output="A blog post in markdown format.",
                tools=[yt_tool],
                agent=blog_writer,
                output_file="blog_post.md"
            )

            crew = Crew(
                agents=[blog_researcher, blog_writer],
                tasks=[research_task, write_task],
                process=Process.sequential,
                verbose=True,
                memory=False,
                max_iter=2
            )

            result = crew.kickoff(inputs={"topic": topic_query})

            st.success("Completed!")
            st.markdown(result.raw)

        except Exception as e:
            st.error(f"Error: {str(e)}")
