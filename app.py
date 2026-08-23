import os
import re
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi

# 1. Page Configuration
st.set_page_config(
    page_title="▶️ YouTube Content Generator",
    page_icon="▶️",
    layout="wide"
)

st.title("YouTube Blog Post Generator")

# 2. Secret & Input Management
if "OPENROUTER_API_KEY" in st.secrets:
    openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
else:
    openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

MODEL_NAME = "openrouter/openai/gpt-4o-mini"

# User inputs full YouTube Channel URL directly
channel_url_input = st.sidebar.text_input(
    "Full YouTube Channel URL", 
    placeholder="https://www.youtube.com/"
).strip()

topic_query = st.text_input("Enter Topic / Query to Research", value="")

# 3. Native Custom Tool for YouTube Transcripts (Fallback)
@tool("Fetch YouTube Transcript")
def fetch_youtube_transcript(video_id_or_url: str) -> str:
    """
    Extracts text transcript given a YouTube video ID or full YouTube video URL.
    Use this tool to get transcript text from a video for analysis.
    """
    try:
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", video_id_or_url)
        video_id = video_id_match.group(1) if video_id_match else video_id_or_url

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry['text'] for entry in transcript_list])
        
        return full_text[:4000]
    except Exception as e:
        return f"Could not fetch transcript for {video_id_or_url}: {str(e)}"


# 4. Helper Function to Format Full YouTube Channel URL
def clean_channel_url(url_input: str) -> str:
    """Ensures input is a clean, properly formatted YouTube channel URL."""
    url = url_input.strip()
    if url.startswith("http://") or url.startswith("https://"):
        return url
    
    if url.startswith("@"):
        return f"https://www.youtube.com/{url}"
    
    return f"https://www.youtube.com/@{url}"


# 5. Execution Logic
if st.button("🚀 Run Crew Workflow"):
    if not openrouter_api_key:
        st.error("Please enter your OpenRouter API Key in the sidebar or Streamlit Secrets.")
        st.stop()

    if not channel_url_input:
        st.error("Please enter a full YouTube Channel URL in the sidebar.")
        st.stop()

    full_channel_url = clean_channel_url(channel_url_input)
    
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
    os.environ["OPENAI_API_KEY"] = "NA"

    with st.spinner("Executing Task..."):
        try:
            # Instantiate Modern CrewAI LLM Class
            llm = LLM(
                model=MODEL_NAME,
                api_key=openrouter_api_key
            )

            # Initialize YoutubeChannelSearchTool with youtube_channel_url parameter
            from crewai_tools import YoutubeChannelSearchTool
            
            yt_tool = YoutubeChannelSearchTool(
                youtube_channel_url=full_channel_url,
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
                goal=f"Extract key video content for topic '{topic_query}' from channel URL '{full_channel_url}'.",
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
                description=f"Search YouTube channel at {full_channel_url} for videos about '{topic_query}' and summarize main insights.",
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
