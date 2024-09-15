import streamlit as st
from crewai import Agent, Task, Crew, Process
import os
from crewai_tools import SerperDevTool
import dotenv
from langchain_groq import ChatGroq
dotenv.load_dotenv()
api_key = os.getenv("SERPER_API_KEY")

search_tool = SerperDevTool()

llama_model = ChatGroq(
    api_key='gsk_mPmesto6dMZI2yWw4g0kWGdyb3FYFnyIyj4KeyL9P71rVAQtsN7',
    model='llama3-groq-70b-8192-tool-use-preview'
)

if 'history' not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Query and Response History")
    for entry in st.session_state.history:
        st.write(f"**Query:** {entry['query']}")
        st.write(f"**Response:** {entry['response']}")
        st.write("---")

with col2:
    st.subheader("Enter your query")
    user_query = st.text_input("Ask about books, papers, or topics:")

    if st.button("Submit Query"):
        if user_query:
            recommender = Agent(
                role = "Book Recommendation Agent",
                goal = """Recommend books, papers, or articles based on the student's query.Give links that you find necessary.
                        Provide any suggestion that is relevant and helpful.""",
                backstory = "You are an AI assistant focused on providing book recommendations to students.",
                verbose=True,
                allow_delegation=False,
                tools=[search_tool],
                llm=llama_model
            )

            follow_up = Agent(
                role="Follow-up Agent",
                goal = """Maintain conversation context to answer follow-up questions based on the student's original query.
                        Make sure the answers are relevant to the previous discussion.""",
                backstory = "You are an AI assistant who work on follow-up question based on context.",
                verbose = True,
                allow_delegation=False,
                llm=llama_model
            )

            recommendation_task = Task(
                description=f"Recommend books or papers along with their links based on the student's query: '{user_query}'",
                agent=book_recommendation_agent,
                expected_output=" A recommended book, paper, or article along with their links."
            )

            follow_up_task = Task(
                description= f"Handle follow-up query based on previous context from: '{user_query}'",
                agent=context_follow_up_agent,
                expected_output="A work on follow-up response."
            )

            crew = Crew(
                agents=[recommender,follow_up],
                tasks=[recommendation_task, follow_up_task],
                verbose=True,
                process=Process.sequential
            )

            output = crew.kickoff()

            recommendation = output.raw

            st.session_state.history.append({
                'query': user_query,
                'response': recommendation
            })

            st.write(output.raw)
        else:
            st.write("Please enter a query about books or topics.")
