import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import YoutubeChannelSearchTool

# 1. Page Configuration
st.set_page_config(
    page_title="Multi-Agent YouTube Content Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 CrewAI Multi-Agent YouTube Blog Post Generator (OpenRouter)")

# 2. Secret & Input Management
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

# 3. Execution Logic
if st.button("🚀 Run Crew Workflow"):
    if not openrouter_api_key:
        st.error("Please enter your OpenRouter API Key in the sidebar or Streamlit Secrets.")
        st.stop()

    # Set environment variables for OpenRouter / LiteLLM integration
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key

    with st.spinner("Executing Task..."):
        try:
            # Instantiate Modern CrewAI LLM Class
            llm = LLM(
                model=model_name,
                api_key=openrouter_api_key
            )

            # Initialize YouTube Search Tool
            yt_tool = YoutubeChannelSearchTool(
                youtube_channel_handle=channel_handle
            )

            # Define Agents
            blog_researcher = Agent(
                role="Blog Researcher from YouTube Videos",
                goal=f"Extract key video content for topic '{topic_query}' from YouTube channel {channel_handle}.",
                backstory="An expert researcher specializing in analyzing YouTube video transcripts and technical content.",
                tools=[yt_tool],
                allow_delegation=False,
                llm=llm,
                verbose=True,
                memory=False,
                max_iter=2
            )

            blog_writer = Agent(
                role="Blog Writer",
                goal=f"Draft a compelling, well-formatted blog post on '{topic_query}' based on research.",
                backstory="A skilled writer who crafts engaging technical blog posts from raw research summaries.",
                tools=[],
                allow_delegation=False,
                llm=llm,
                verbose=True,
                memory=False,
                max_iter=2
            )

            # Define Tasks
            research_task = Task(
                description=f"Search the YouTube channel {channel_handle} for videos related to '{topic_query}' and synthesize main points.",
                expected_output="A detailed summary of key points and facts found in video content.",
                tools=[yt_tool],
                agent=blog_researcher
            )

            write_task = Task(
                description=f"Using the research provided, write a comprehensive markdown blog post about '{topic_query}'.",
                expected_output="A complete, publication-ready blog post in markdown format.",
                agent=blog_writer,
                output_file="blog_post.md"
            )

            # Assemble and Run Crew
            crew = Crew(
                agents=[blog_researcher, blog_writer],
                tasks=[research_task, write_task],
                process=Process.sequential,
                verbose=True,
                memory=False
            )

            result = crew.kickoff(inputs={"topic": topic_query})

            # Render Results
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
