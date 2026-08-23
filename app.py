import os
import re
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi

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

MODEL_NAME = "openrouter/openai/gpt-4o-mini"

channel_input = st.sidebar.text_input("YouTube Channel Handle or URL (e.g., @paurigarhwal)").strip()
topic_query = st.text_input("Enter Topic / Query to Research", value="Enter your query")

# 3. Custom Native CrewAI Tool for YouTube Transcripts
@tool("Fetch YouTube Transcript")
def fetch_youtube_transcript(video_id_or_url: str) -> str:
    """
    Extracts text transcript given a YouTube video ID or full YouTube video URL.
    Use this tool to get transcript text from a video for analysis.
    """
    try:
        # Extract 11-character video ID if a full URL is passed
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", video_id_or_url)
        video_id = video_id_match.group(1) if video_id_match else video_id_or_url

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry['text'] for entry in transcript_list])
        
        # Limit length to avoid exceeding context window
        return full_text[:4000]
    except Exception as e:
        return f"Could not fetch transcript for {video_id_or_url}: {str(e)}"


# 4. Alternative Standard YoutubeChannelSearchTool with Full URL Formatting
def get_channel_url(user_input: str) -> str:
    """Ensures input is formatted as a full YouTube URL for Embedchain compatibility."""
    user_input = user_input.strip()
    if user_input.startswith("http://") or user_input.startswith("https://"):
        return user_input
    
    handle = user_input if user_input.startswith("@") else f"@{user_input}"
    return f"https://www.youtube.com/{handle}"


# 5. Execution Logic
if st.button("🚀 Run Crew Workflow"):
    if not openrouter_api_key:
        st.error("Please enter your OpenRouter API Key in the sidebar or Streamlit Secrets.")
        st.stop()

    if not channel_input:
        st.error("Please enter a YouTube Channel Handle or URL in the sidebar.")
        st.stop()

    formatted_channel_url = get_channel_url(channel_input)
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
    os.environ["OPENAI_API_KEY"] = "NA"

    with st.spinner("Executing Task..."):
        try:
            # Instantiate CrewAI LLM
            llm = LLM(
                model=MODEL_NAME,
                api_key=openrouter_api_key
            )

            # Initialize YoutubeChannelSearchTool with full valid URL
            from crewai_tools import YoutubeChannelSearchTool
            
            yt_tool = YoutubeChannelSearchTool(
                youtube_channel_handle=formatted_channel_url,
                config=dict(
                    llm=dict(
                        provider="openrouter",
                        config=dict(
                            model=MODEL_NAME,
                            api_key=openrouter_api_key,
                        ),
                    ),
                    embedder=dict(
                        provider="huggingface",
                        config=dict(
                            model="sentence-transformers/all-MiniLM-L6-v2",
                        ),
                    ),
                )
            )

            # Define Agents
            blog_researcher = Agent(
                role="Blog Researcher from YouTube Videos",
                goal=f"Extract key video content for topic '{topic_query}' from channel '{formatted_channel_url}'.",
                backstory="An expert researcher specializing in analyzing YouTube content and technical transcripts.",
                tools=[yt_tool, fetch_youtube_transcript],
                allow_delegation=False,
                llm=llm,
                verbose=True,
                memory=False,
                max_iter=2
            )

            blog_writer = Agent(
                role="Blog Writer",
                goal=f"Draft a compelling, well-formatted blog post on '{topic_query}' based on research.",
                backstory="A skilled writer who crafts engaging blog posts from raw research summaries.",
                tools=[],
                allow_delegation=False,
                llm=llm,
                verbose=True,
                memory=False,
                max_iter=2
            )

            # Define Tasks
            research_task = Task(
                description=f"Search YouTube channel at {formatted_channel_url} for videos about '{topic_query}' and summarize main insights.",
                expected_output="A detailed summary of key points and facts found in video content.",
                tools=[yt_tool, fetch_youtube_transcript],
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
