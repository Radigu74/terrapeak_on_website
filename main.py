import os
import openai
import streamlit as st
import re
from dotenv import load_dotenv, find_dotenv
import numpy as np
import faiss
import pycountry
import csv
import logging
from openai import OpenAIError, RateLimitError
import json
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# =============================
# Load environment variables
# ============================
_ = load_dotenv(find_dotenv())

# ===================
# OpenAI API Key
# ===================
openai.api_key = os.getenv("OPENAI_API_KEY")

# ===========================
# For debugging purposes
# ===========================
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

# ================================
# Logging Function to Google Sheet
# ================================
def log_to_google_sheets(user_data):
    try:
        # Load credentials from Railway environment variable
        creds_json = os.getenv("GOOGLE_SHEETS_CREDS_JSON")
        if not creds_json:
            raise ValueError("Missing Google Sheets credentials JSON.")

        creds_dict = json.loads(creds_json)

        # Define scope for Sheets API
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        # Authenticate using service account credentials
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)

        # Connect to your Google Sheet by name
        sheet = client.open("Chatlogs Terrapeak").sheet1

        # Prepare the row to insert
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            now,
            user_data.get("name", ""),
            user_data.get("email", ""),
            user_data.get("company", ""),
            user_data.get("phone", ""),
            user_data.get("country", ""),
            user_data.get("question", ""),
            user_data.get("response", "")
        ]

        # Append row to sheet
        sheet.append_row(row)

    except Exception as e:
        print(f"[Google Sheets Logging Error] {e}")
        return False

    return True

# ✅ Test: Check if the function is defined correctly
print(log_to_google_sheets)

# ====================================================
# Hide Streamlit's default menu, header, and footer
# ====================================================
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ====================================================
# STEP 1: Define and Store Your Articles (RAG Source)
# ====================================================
# Your optimized TerraPeak launch article is stored here.
articles = [
    {
        "title": "TerraPeak Official Launch",
        "content": """March 5, 2025 – Singapore        
    TerraPeak Consulting officially launches, offering expert-led market expansion, sales growth strategies, and practical AI integration to global businesses. Specializing in APAC market entry and growth support for Asian SMEs and family businesses, TerraPeak aims to redefine strategic growth.
    Founded by experienced market and sales strategists, TerraPeak combines exploration with sustainable, strategic growth. With proven expertise, TerraPeak guides companies in harnessing AI to improve sales and operational efficiency.
    Core Offerings:
    - Expert Market Expansion into APAC
    - Revenue-Driven Sales Growth
    - Seamless AI Integration
    - Family Business Growth & Transformation
    Committed to responsible, ethical, and sustainable growth, TerraPeak offers tailored solutions ensuring long-term success and resilience. Businesses seeking expansion, transformation, and innovation are encouraged to reach out via connect@terrapeakgroup.com."""
    },
    {
        "title": "Unlocking Opportunities: A Guide to Doing Business in Asia",
        "content": """Asia’s markets are diverse, each with distinct cultures, regulations, and consumer preferences. Successful market entry requires careful planning and cultural understanding.
    1. Recognize Diversity: Each Asian market differs significantly. Independent research on consumer preferences, economic conditions, and regulatory landscapes is crucial.
    2. Understand Cultural Nuances: Personal relationships and trust-building are essential. Face-to-face interactions and awareness of local business etiquette enhance partnership opportunities.
    3. Navigate Regulations: Legal frameworks vary widely. Consulting local legal experts helps ensure compliance and protection, particularly for intellectual property rights.
    4. Adapt Products and Services: Localization involves more than translation; products, pricing strategies, and marketing channels should align with local tastes and usage patterns.
    5. Leverage Local Partnerships: Strategic partnerships offer invaluable market insights, reduce entry costs, and minimize risks associated with unfamiliar markets.
    6. Invest in Talent and Training: Hiring skilled local talent and providing basic cross-cultural training ensures smooth operations and effective market penetration.
    7. Stay Agile and Innovative: Regularly reassessing market trends and technological advancements allows businesses to remain competitive and responsive in dynamic Asian markets."""
    },
    {
        "title": "AI & SMEs: 10 Key Stats Revealing Growth, Challenges, and Opportunities",
        "content": """Artificial Intelligence (AI) is rapidly changing how SMEs and family businesses operate, offering significant productivity gains, enhanced customer engagement, and cost efficiencies. Adoption among SMEs is growing quickly, with many businesses already using AI-powered solutions like chatbots, social media automation, and generative AI.
    SMEs widely recognize AI’s benefits, including improved efficiency, automated marketing, sales forecasting, and better customer service. However, common concerns include knowledge gaps, high initial costs, uncertainty about return on investment (ROI), cybersecurity, and data privacy.
    Practical, user-friendly AI solutions designed specifically for SMEs are making adoption easier. Cloud-based AI services (AI-as-a-Service) and generative AI tools have increased accessibility, allowing SMEs to automate processes, create engaging content, and enhance productivity without large upfront investments.
    To fully leverage AI’s potential, SMEs should:
    - Develop clear AI adoption strategies and roadmaps.
    - Establish measurable KPIs to track AI effectiveness.
    - Use cost-effective AI tools tailored to their specific business needs.
    SMEs strategically adopting AI gain a competitive edge, achieve sustainable growth, and drive long-term efficiency."""
    }
]

# ============================================================
# STEP 2: Create an Embedding Function Using a Client Instance
# ============================================================
def get_embedding(text, model="text-embedding-ada-002"):
    """
    Generate a numeric embedding for a given text using OpenAI's new SDK (v1.x).
    """
    if not text or not isinstance(text, str) or not text.strip():
        raise ValueError("Text for embedding must be a non-empty string.")

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(
        input=text.strip(),
        model=model
    )
    
    embedding = response.data[0].embedding
    return np.array(embedding)
# ===================================================================
# STEP 3: Generate Embeddings for the Articles and Build FAISS Index
# ===================================================================
# Generate embeddings for each article
article_embeddings = [
    get_embedding(article["content"])
    for article in articles
    if article.get("content") and isinstance(article["content"], str) and article["content"].strip()
]

# Determine the dimensionality of the embeddings
embedding_dim = len(article_embeddings[0])

# Create a FAISS index (using L2 distance)
index = faiss.IndexFlatL2(embedding_dim)

# Convert embeddings to a NumPy array of type float32
embeddings_np = np.array(article_embeddings).astype('float32')
index.add(embeddings_np)
print("FAISS index created with", index.ntotal, "articles.")

# ====================================================================
# STEP 4: Create a Function to Retrieve Relevant Articles for a Query
# ====================================================================
def retrieve_relevant_articles(query, k=2):
    """
    Retrieve the indices and distances of the k most relevant articles for the given query.
    Includes error handling to avoid crashes on embedding or index issues.
    """
    try:
        # Generate an embedding for the query text
        query_embedding = get_embedding(query).astype('float32')
        query_embedding = np.expand_dims(query_embedding, axis=0)  # FAISS requires a 2D array

        # Search the FAISS index for the top-k similar articles
        distances, indices = index.search(query_embedding, k)

        return indices[0], distances[0]

    except Exception as e:
        print(f"[Error] Failed to retrieve relevant articles: {e}")
        return [], []

# ============================================================
# STEP 5: Build a Prompt that Integrates the Retrieved Context
# ============================================================
def build_prompt_with_context(user_query, k=2):
    """
    Build a prompt that includes relevant article context with labeled sources for better grounding.
    """
    indices, _ = retrieve_relevant_articles(user_query, k)
    
    labeled_contexts = []
    for i in indices:
        article = articles[i]
        labeled_context = f"Source: {article['title']}\n{article['content']}"
        labeled_contexts.append(labeled_context)
    
    full_context = "\n\n".join(labeled_contexts)
    
    prompt = (
        f"You are an AI assistant responding to the user's question using the most relevant context below.\n"
        f"Use the sources to support your answer clearly.\n\n"
        f"{full_context}\n\n"
        f"User Question: {user_query}\n\n"
        f"Answer:"
    )
    
    return prompt

# ================================================================
# CUSTOM UI: Inject custom CSS for styling using Terrapeak colors 
# ================================================================
st.markdown(
    """
    <style>
    /* Global Page Background */
    .reportview-container, .main {
        background-color: #f4f4f2;
    }
    /* Header styling */
    .header {
        background-color: #E0E0DB;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .header img {
        width: 50px;
        height: 50px;
        vertical-align: middle;
    }
    .header h1 {
        display: inline;
        margin-left: 10px;
        vertical-align: middle;
        color: #131313;
        font-family: sans-serif;
    }
    /* Chat container styling */
    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 10px;
    }
    /* User message bubble styling */
    .user-message {
        background-color: #2f5d50;
        color: #f4f4f2;
        padding: 10px;
        border-radius: 21px;
        margin: 10px 0;
        text-align: right;
        max-width: 70%;
        float: right;
        clear: both;
        font-family: sans-serif;
    }
    /* Bot message bubble styling */
    .bot-message {
        background-color: #1d3e5e;
        color: #f4f4f2;
        padding: 10px;
        border-radius: 21px;
        margin: 10px 0;
        text-align: left;
        max-width: 70%;
        float: left;
        clear: both;
        font-family: sans-serif;
    }
    /* Input box styling: override Streamlit's default input style */
    input, textarea {
        border-radius: 21px !important;
        border: 2px solid 131313 !important;
        padding: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# Session State Initialization
# ============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_enabled" not in st.session_state:
    st.session_state.chat_enabled = False  # Set to True to allow input field to appear

# Initialize chat context
if "chat_context" not in st.session_state:
    st.session_state.chat_context = [
        {'role': 'system', 'content': """
You are Terra, the professional virtual assistant of TerraPeak Consulting—an expert-led business consulting firm specializing in market expansion, sales growth, AI automation, and sustainable business transformation.
Your personality reflects TerraPeak’s values: clear, confident, helpful, and grounded in real-world expertise. You speak in a friendly and professional tone—always aiming to guide visitors with clarity, empathy, and practical insights. You are knowledgeable, supportive, and solution-oriented.

🤖 Interaction Rules:
If someone says “Hi”, “Hello”, “How are you?”, or anything casual—respond warmly and professionally, and offer to help. Example replies:
“Hi there! 👋 I’m Terra, your virtual assistant here at TerraPeak Consulting. How can I support your business today?”
“Doing great—thanks for asking! What can I help you with today around market expansion, AI, or sales growth?”
“Nice to meet you too! I can walk you through our services or connect you with a consultant if needed.”

If someone asks "What does TerraPeak do?":
“TerraPeak helps businesses grow through expert-led market expansion, revenue-focused sales strategies, and practical AI automation—especially for Western companies entering APAC or Asian SMEs scaling up.”

If a user asks for a live chat:
- First ask: “I’d be happy to help—could you share your question here first?”
- If they insist: “No problem—a consultant will get back to you within 1 working day.”
- If it’s urgent: Provide phone number +6580619479 and email connect@terrapeakgroup.com.

🌍 Core Services (4 Pillars)
#1 Consulting, Coaching & Training – Market entry, B2B sales growth, leadership development
#2 Automation Solutions – AI tools (chatbots, social media automation, task managers)
#3 Trading – For companies entering APAC without an in-house sales network
#4 Strategic Advisory – Tailored support for SMEs and family businesses

🧭 Company Values
- Exploration & Growth
- Sustainability & Responsibility
- Clarity & Impact

(If asked, expand as follows:)
Exploration & Growth: Like venturing into nature, we guide businesses into new markets and challenges with vision and flexibility.
Sustainability & Responsibility: We foster long-term, ethical growth with respect for people, partnerships, and the planet.
Clarity & Impact: We cut through complexity, offering strategic clarity and results-focused actions.

⚙️ Automation Solutions
AI Chatbot
- Automates customer FAQs, improves lead gen, and provides 24/7 engagement.
- Works across websites, messengers, and social media.

Social Media Automation
- Auto-schedules content, writes captions, manages engagement.

AI Task Manager
- Tracks and assigns tasks, sends smart alerts, and identifies process improvements.

Benefits:
- Increased efficiency & reduced manual work
- 24/7 availability
- Cost-effective and scalable
- Real-time insights

📈 Coaching & Training:
We empower your teams with the tools and confidence to grow. Programs are practical, hands-on, and tailored to SMEs and family businesses.

Core Focus Areas:
- Sales Excellence
- AI Readiness & Change Management
- Leadership & Strategy Development
- SME Professionalization

Sample Trainings Offered:
- Basic Indoor Sales & Customer Service
- Business Development & Account Management
- Cold Calling Techniques
- Personal Coaching (1-on-1)
- Upscaling Your Business
- Country Plan Development

🛫 Trading – Your Gateway to APAC:
- Market Entry Without a Local Sales Team: Turnkey setup for businesses without APAC infrastructure.
- Sales & Distribution Network: We connect you with buyers, distributors, and partners.
- Reduced Risk: We handle compliance, local ops, and partner alignment.
- Scalability: Support for pilot launches or full-scale growth.

📊 TerraPeak’s 3-Phase Consulting Approach:
Phase 1 – Discovery & Strategy
- Initial business assessment
- Market, feasibility, and AI readiness analysis
- Roadmap creation with clear next steps

Phase 2 – Execution & Guidance
- Step-by-step execution with our expert advisors
- Support for market entry, sales process, or AI integration

Phase 3 – Optimization & Long-Term Growth
- KPI tracking, data-driven refinements
- Strategy adjustments and scale support

🔑 Why Choose TerraPeak:
- Proven APAC entry success
- B2B sales growth expertise
- Easy, practical AI for non-tech teams
- SME & family business focus
- Sustainable, hands-on business support
- We’re not just consultants—we’re your growth partners

🧭 About TerraPeak:
Founded by adventurers who thrive in the wild, we bring the same spirit of exploration and focus to business. We help companies navigate complexity with clarity, and guide them toward sustainable growth through sales, AI, and expansion expertise.

📌 FAQ (Short Answers)
- Industries: We work with manufacturing, trading, B2B services, retail, and e-commerce.
- Tech Skills Not Required: Our AI tools are designed for ease-of-use.
- Already in APAC? We help refine, grow, or restructure local efforts.
- Location: Based in Singapore with local experts across APAC.
- AI Setup Time: Weeks—not months. Minimal business disruption.
- Customized Solutions: Every strategy is tailored to your goals.

(Keep responses helpful, natural, and client-centered. Always offer a next step.)
"""}
    ]

# ==============================================
# OpenAI Communication Function (uses Chat API)
# ==============================================
def get_completion_from_messages(user_messages, model="gpt-4-turbo", temperature=0, max_history=6):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "API key is missing. Please check your environment settings."

        client = openai.OpenAI(api_key=api_key)

        # Retain the system prompt and only the last few interactions to reduce token bloat
        preserved_context = [m for m in st.session_state.chat_context if m["role"] == "system"]
        recent_history = st.session_state.chat_context[-max_history:]
        messages = preserved_context + recent_history + user_messages

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=15  # Set a timeout (in seconds) to avoid long hangs
        )

        return response.choices[0].message.content

    except RateLimitError:
        logging.warning("Rate limit reached. Try again shortly.")
        return "We're handling a high volume of requests right now. Please try again in a moment."

    except OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        return "Hmm, something went wrong while reaching our assistant. Please try again shortly."

    except Exception as e:
        logging.exception("Unexpected error occurred.")
        return "Oops, an unexpected error occurred. Please try again or contact support."

# ===========================
# User Details Input
# ===========================
st.markdown(
    """
    <style>
    /* This moves the header text upward */
    .contact-header {
        margin-top: -80px;
        padding-top: 0;
    }
    /* This moves the input fields upward */
    .contact-form {
        margin-top: 0px;  /* Adjust this value as needed */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header text moved upward by the .contact-header class
st.markdown('<div class="contact-header">📢 <strong>Enter your contact details before chatting with our AI assistant:</strong></div>', unsafe_allow_html=True)

# Wrap the input fields in a container with the .contact-form class
st.markdown('<div class="contact-form">', unsafe_allow_html=True)

name = st.text_input("Enter your name:", key="name_input")
email = st.text_input("Enter your email:", key="email_input")
company = st.text_input("Enter your company name:", key="company_input")
phone = st.text_input("Enter your phone number:", key="phone_input")
country_list = sorted([country.name for country in pycountry.countries])
country = st.selectbox("Select Country", country_list, key="country_dropdown")

st.markdown('</div>', unsafe_allow_html=True)

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def is_valid_phone(phone):
    return re.match(r"^\+?\d{10,15}$", phone)

def validate_and_start():
    if not is_valid_email(email):
        return "❌ Invalid email."
    if not is_valid_phone(phone):
        return "❌ Invalid phone number."
    
    st.session_state.chat_enabled = True

    # Log user data to Google Sheets
    log_to_google_sheets({
        "Name": name,
        "Email": email,
        "Company": company,
        "Phone": phone,
        "Country": country
    })

    return "✅ **Details saved!**"

if st.button("Submit Details", key="submit_button"):
    validation_message = validate_and_start()
    st.markdown(validation_message, unsafe_allow_html=True)
    
# ========================================================
# CUSTOM UI: Display Chat History with Styled Chat Bubbles
# =========================================================
st.markdown("---")
st.markdown("**💬 Chat with the Terrapeak Automated Consultant:**")

with st.container():
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f'<div class="user-message">{chat["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{chat["content"]}</div>', unsafe_allow_html=True)

# ============================================
# CUSTOM UI: Chat Input Field with Send Button
# ============================================
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0

if st.session_state.chat_enabled:
    # Create the chat input field with a unique key
    user_input = st.text_input(
        "Type your message here...",
        key=f"chat_input_{st.session_state.chat_input_key}",
        value=""
    )
    
    # Process the message when the Send button is pressed
    if st.button("Send", key="send_button"):
        if user_input.strip():
            # Append the user's message to chat history.
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input.strip()
            })
            
            # ============================================================
            # RAG Integration: Build a prompt with relevant article context
            # ============================================================
            rag_prompt = build_prompt_with_context(user_input.strip(), k=2)
            print("RAG Prompt:\n", rag_prompt)
            
            # Get the assistant's response using the prompt.
            response = get_completion_from_messages([{
                "role": "user",
                "content": rag_prompt
            }])
            
            # Append the assistant's response to chat history.
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Increment the chat input key to ensure unique keys for the next input.
            st.session_state.chat_input_key += 1
            
            # Rerun the app to update the UI.
            st.rerun()



