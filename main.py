import os
import openai
import streamlit as st
import re
from dotenv import load_dotenv, find_dotenv
import numpy as np
import faiss
import pycountry
import csv

# Load environment variables
_ = load_dotenv(find_dotenv())

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# For debugging purposes
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

# Hide Streamlit's default menu, header, and footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

    # ============================================================
# STEP 1: Define and Store Your Articles (RAG Source)
# ============================================================
# Your optimized TerraPeak launch article is stored here.
articles = [
    """
    TerraPeak Consulting Officially Launches to Guide Businesses Toward Market Expansion, Sales Growth, and AI Integration
    March 5, 2025 – Singapore
    TerraPeak Consulting officially launches, offering expert-led market expansion, sales growth strategies, and practical AI integration to global businesses. Specializing in APAC market entry and growth support for Asian SMEs and family businesses, TerraPeak aims to redefine strategic growth.
    Founded by experienced market and sales strategists, TerraPeak combines exploration with sustainable, strategic growth. With proven expertise, TerraPeak guides companies in harnessing AI to improve sales and operational efficiency.
    Core Offerings:
    - Expert Market Expansion into APAC
    - Revenue-Driven Sales Growth
    - Seamless AI Integration
    - Family Business Growth & Transformation
    Committed to responsible, ethical, and sustainable growth, TerraPeak offers tailored solutions ensuring long-term success and resilience. Businesses seeking expansion, transformation, and innovation are encouraged to reach out via connect@terrapeakgroup.com.
    """,
    """
    Unlocking Opportunities: Doing Business in Asia for Western Companies
    Asia's markets are diverse, each with distinct cultures, regulations, and consumer preferences. Successful market entry requires careful planning and cultural understanding.
    1. Recognize Diversity: Each Asian market differs significantly. Independent research on consumer preferences, economic conditions, and regulatory landscapes is crucial.
    2. Understand Cultural Nuances: Personal relationships and trust-building are essential. Face-to-face interactions and awareness of local business etiquette enhance partnership opportunities.
    3. Navigate Regulations: Legal frameworks vary widely. Consulting local legal experts helps ensure compliance and protection, particularly for intellectual property rights.
    4. Adapt Products and Services: Localization involves more than translation; products, pricing strategies, and marketing channels should align with local tastes and usage patterns.
    5. Leverage Local Partnerships: Strategic partnerships offer invaluable market insights, reduce entry costs, and minimize risks associated with unfamiliar markets.
    6. Invest in Talent and Training: Hiring skilled local talent and providing basic cross-cultural training ensures smooth operations and effective market penetration.
    7. Stay Agile and Innovative: Regularly reassessing market trends and technological advancements allows businesses to remain competitive and responsive in dynamic Asian markets.
    """,
    """
    AI & SMEs: Key Trends, Challenges, and Opportunities
    AI is rapidly changing how SMEs and family businesses operate, offering significant productivity gains, enhanced customer engagement, and cost efficiencies. Adoption among SMEs is growing quickly, with many businesses already using AI-powered solutions like chatbots, social media automation, and generative AI.
    SMEs widely recognize AI’s benefits, including improved efficiency, automated marketing, sales forecasting, and better customer service. However, common concerns include knowledge gaps, high initial costs, uncertainty about return on investment (ROI), cybersecurity, and data privacy.
    Practical, user-friendly AI solutions designed specifically for SMEs are making adoption easier. Cloud-based AI services (AI-as-a-Service) and generative AI tools have increased accessibility, allowing SMEs to automate processes, create engaging content, and enhance productivity without large upfront investments.
    To fully leverage AI’s potential, SMEs should:
    - Develop clear AI adoption strategies and roadmaps.
    - Establish measurable KPIs to track AI effectiveness.
    - Use cost-effective AI tools tailored to their specific business needs.
    SMEs strategically adopting AI gain a competitive edge, achieve sustainable growth, and drive long-term efficiency.
    """
]

# ============================================================
# STEP 2: Create an Embedding Function Using a Client Instance
# ============================================================
def get_embedding(text, model="text-embedding-ada-002"):
    """
    Generate a numeric embedding for a given text using OpenAI's API.
    Uses the new API style (openai.embeddings.create) and accesses the embedding 
    via attribute notation.
    """
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    # Use attribute notation to access the embedding
    embedding = response.data[0].embedding
    return np.array(embedding)

# ============================================================
# STEP 3: Generate Embeddings for the Articles and Build FAISS Index
# ============================================================
# Generate embeddings for each article
article_embeddings = [get_embedding(article) for article in articles]

# Determine the dimensionality of the embeddings
embedding_dim = len(article_embeddings[0])

# Create a FAISS index (using L2 distance)
index = faiss.IndexFlatL2(embedding_dim)

# Convert embeddings to a NumPy array of type float32
embeddings_np = np.array(article_embeddings).astype('float32')
index.add(embeddings_np)
print("FAISS index created with", index.ntotal, "articles.")

# ============================================================
# STEP 4: Create a Function to Retrieve Relevant Articles for a Query
# ============================================================
def retrieve_relevant_articles(query, k=2):
    """
    Retrieve the indices and distances of the k most relevant articles for the given query.
    """
    # Generate an embedding for the query text
    query_embedding = get_embedding(query).astype('float32')
    query_embedding = np.expand_dims(query_embedding, axis=0)  # FAISS requires a 2D array
    
    # Search the index for the top k similar articles
    distances, indices = index.search(query_embedding, k)
    return indices[0], distances[0]

# ============================================================
# STEP 5: Build a Prompt that Integrates the Retrieved Context
# ============================================================
def build_prompt_with_context(user_query, k=2):
    """
    Build a prompt that includes relevant context from retrieved articles along with the user query.
    """
    indices, _ = retrieve_relevant_articles(user_query, k)
    # Combine the texts of the retrieved articles into one context string
    context = "\n\n".join([articles[i] for i in indices])
    prompt = (
        f"Relevant Context:\n{context}\n\n"
        f"User Query: {user_query}\n\n"
        f"Answer:"
    )
    return prompt

# ============================================================
# EXISTING CHATBOT CODE BEGINS BELOW
# ============================================================

# ===========================
# CUSTOM UI: Inject custom CSS for styling using Terrapeak colors 
# ===========================
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

# ===========================
# Session State Initialization
# ===========================
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

If someone says “Hi”, “Hello”, “How are you?”, or anything casual—respond warmly and professionally, and offer to help. Examples:
- “Hi there! 👋 I’m Terra, your virtual assistant here at TerraPeak Consulting. How can I support your business today?”
- “Doing great—thanks for asking! What can I help you with today around market expansion, AI, or sales growth?”
- “Nice to meet you too! I can walk you through our services or connect you with a consultant if needed.”

If a user asks “What does TerraPeak do?”, explain:
“TerraPeak helps businesses grow through expert-led market expansion, revenue-focused sales strategies, and practical AI automation—especially for Western companies entering APAC or Asian SMEs scaling up.”

When asked for a live chat:
- First ask them to share their question with you.
- If they insist, explain a callback will be arranged within 1 working day.
- If urgent, share the phone number: +6580619479
- You may also share the email: connect@terrapeakgroup.com

TerraPeak Consulting specializes in:
- Helping Western companies expand into the APAC region
- Supporting Asian SMEs and family businesses to scale and professionalize
- Guiding businesses in adopting AI for automation (e.g., chatbots, social media, task management)
- Providing Trading support for companies entering APAC without a local sales network

Core Service Areas:
1. Consulting, Coaching & Training
2. Automation Solutions
3. Trading
4. Strategic Advisory

Company Values:
- Exploration & Growth
- Sustainability & Responsibility
- Clarity & Impact

(If asked, expand on the values using the optional clarifications)

---
🔹 AUTOMATION SOLUTIONS

AI Chatbot:
- Automates FAQs and customer support
- Captures leads and routes them to sales
- Works across websites, messaging apps, and social channels

Social Media Automation:
- Schedules content for best engagement times
- AI generates captions, content ideas, and hashtags
- Manages responses and enhances engagement

AI Task Manager:
- Automates task assignment & tracking
- Sends smart reminders & alerts
- Offers insights to improve workflows

Benefits:
- Increased efficiency & productivity
- 24/7 availability
- Cost savings through automation
- Scalable for business growth
- Real-time, data-driven insights

---
🔹 COACHING & TRAINING

Tailored support to develop your team’s skills:
- Sales Excellence Training
- AI Readiness Coaching
- Leadership & Strategy Development
- SME Professionalization

Courses we offer:
- Basic Indoor Sales & Customer Service
- Business Development & Account Management
- Cold Calling
- Personal Coaching (1-on-1)
- Upscaling Business for Next-Stage Growth
- Country Plan Development

---
🔹 TRADING

TerraPeak Trading – Gateway to APAC:
- Enter APAC markets without building a local sales team
- Use our network for partners, buyers, and distribution
- Reduce risk via guided compliance and execution
- Pilot or scale entry with flexibility

---
🔹 OUR 3-PHASE CONSULTING APPROACH

PHASE 1 – Discovery & Strategy
- Assess goals, strengths & challenges
- Market & feasibility analysis
- Create a roadmap for market entry, sales or AI adoption

PHASE 2 – Execution
- Guided implementation of plans
- Support for local partnerships, distribution, and tech adoption
- Hands-on guidance to reduce risk and maximize momentum

PHASE 3 – Evaluation
- Measure KPIs, performance, and results
- Optimize strategies with data insights
- Ensure long-term scalability and relevance

---
🔹 WHY CHOOSE TERRAPEAK

- Proven success in APAC entry and growth strategies
- Real-world sales & business expertise
- Practical, accessible AI tools for non-tech companies
- Personalized support for SMEs & family businesses
- Focus on sustainable, ethical business transformation
- We’re not just consultants—we’re your growth partner

---
🔹 ABOUT TERRAPEAK
We’re founded by nature-loving explorers who see growth as an adventure. We guide businesses through unknown terrain with vision, resilience, and a deep passion for sustainable business success. Whether helping you expand, optimize, or innovate, TerraPeak helps you reach peak performance—with clarity, direction, and hands-on expertise.

---
🔹 FAQ HIGHLIGHTS

- We work with manufacturing, trading, B2B services, retail, and e-commerce sectors.
- You don’t need technical skills to adopt AI—our tools are designed for ease and impact.
- Already active in APAC? We help optimize and expand your local success.
- Based in Singapore with local experts across APAC.
- AI solutions can be integrated in weeks with minimal disruption.
- Every project is tailored to your company’s goals and growth stage.
"""}
    ]

# ===========================
# OpenAI Communication Function (uses Chat API)
# ===========================
def get_completion_from_messages(user_messages, model="gpt-3.5-turbo", temperature=0):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = st.session_state.chat_context + user_messages
    response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return response.choices[0].message.content

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
    return "✅ **Details saved!**"

# -----------------------------
# Function to Append Customer Details to a CSV File
# -----------------------------
def append_to_csv(name, email, phone, country, 
                  file_path=r"C:\Users\ray\Terrapeak\Chatbot\Terrapeak_website_bot\terrapeak_chatbot\Chatbot Leads\customer_details.csv"):
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Check if the file exists (or is empty)
    file_exists = os.path.exists(file_path)
    
    # Open the file in append mode (it will create the file if it doesn't exist)
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # If the file does not exist or is empty, write the header row first
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writerow(["Name", "Email", "Phone", "Country"])
        # Append the new customer details
        writer.writerow([name, email, phone, country])


# Example usage when the submit button is clicked
if st.button("Submit Details", key="submit_button"):
    validation_message = validate_and_start()
    st.markdown(validation_message, unsafe_allow_html=True)
    # If validation is successful, update the CSV file
    if validation_message.startswith("✅"):
        file_path = r"C:\Users\ray\Terrapeak\Chatbot\Terrapeak_website_bot\terrapeak_chatbot\Chatbot Leads\customer_details.csv"
        append_to_csv(name, email, phone, country, file_path)
# ===========================
# CUSTOM UI: Display Chat History with Styled Chat Bubbles
# ===========================
st.markdown("---")
st.markdown("**💬 Chat with the Terrapeak Automated Consultant:**")

with st.container():
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f'<div class="user-message">{chat["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{chat["content"]}</div>', unsafe_allow_html=True)

# ===========================
# CUSTOM UI: Chat Input Field with Send Button
# ===========================
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0

if st.session_state.chat_enabled:
    user_input = st.text_input(
        "Type your message here...",
        key=f"chat_input_{st.session_state.chat_input_key}",
        value=""
    )

    if st.button("Send", key="send_button"):
        if user_input.strip():
            # Append the original user message to chat history.
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            
            # ============================================================
            # RAG Integration: Build a prompt with relevant article context
            # ============================================================
            rag_prompt = build_prompt_with_context(user_input.strip(), k=2)
            print("RAG Prompt:\n", rag_prompt)
            
            # Use the RAG prompt as the user message for the chat completion.
            response = get_completion_from_messages([{"role": "user", "content": rag_prompt}])
            
            # Append the bot's response to chat history.
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.session_state.chat_input_key += 1
            st.rerun()
