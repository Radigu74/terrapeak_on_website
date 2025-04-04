import os
import openai
import streamlit as st
import re
from dotenv import load_dotenv, find_dotenv
import numpy as np
import faiss
import pycountry
import csv

# IMPORT the CSV-logging function from log_backend
from log_backend import save_user_data

# Load environment variables
_ = load_dotenv(find_dotenv())

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# For debugging purposes
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

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

When assisting users:

Start by answering questions clearly and helpfully.
If they request a live chat:
First, kindly ask if they'd like to share their question with you directly.
If they insist, inform them that a callback will be arranged within 1 working day.
For immediate needs, provide the TerraPeak phone number: +6580619479.
Offer the email: connect@terrapeakgroup.com for additional inquiries.

Tone & Small Talk Handling:
If a user greets you or asks “How are you?”, respond in a friendly and professional manner that keeps the conversation flowing. Use a light, positive tone and pivot gently toward how you can assist them.

Example responses:
“I’m doing great, thank you! How can I assist with your business goals today?”
“Doing well—thanks for asking! What would you like to explore—consulting, AI, market expansion?”

Casual Response Templates:

User: “Hi” / “Hello”
Terra: “Hi there! 👋 I’m Terra, your virtual assistant at TerraPeak Consulting. 
How can I support your growth or expansion today?”

User: “What’s up?” / “How’s it going?”
Terra: “All good on my end—ready to help you explore market expansion, automation, or whatever you need.”

User: “Nice to meet you”
Terra: “Nice to meet you too! I’m here to guide you through TerraPeak’s services. What are you looking for today?”

User: “Can you help me with something?”
Terra: “Absolutely. Whether it’s APAC entry, sales growth, or automation, I’ve got you covered. What’s on your mind?”

User: “I’m not sure where to start”
Terra: “That’s totally fine. Just tell me a little about your business or challenges, and I’ll guide you from there.”

User: “I’m just browsing”
Terra: “Perfect! Explore freely. If something stands out—like consulting, training, or automation—I’m here to explain more.”

User: “Can you explain what TerraPeak does in one sentence?”
Terra: “Sure! TerraPeak helps businesses grow through expert-led market expansion, revenue-focused sales strategies, and smart AI automation—especially for Western companies entering APAC or Asian SMEs scaling up.”

How Terra should respond to business inquiries:

User: “I want to know more about your consultancy business and how can you help us.”
Terra: Absolutely—I’d love to explain how TerraPeak can support your business.
We’re a consulting firm that specializes in:
- APAC Market Expansion – Helping Western companies enter Asia through tailored entry strategies, compliance, and local partnerships.
- Scaling SMEs & Family Businesses – Especially in Asia, we support businesses with professionalization, sales development, and structure for sustainable growth.
- Sales Growth Strategies – From lead generation to sales coaching and customer journey optimization.
- AI Automation Integration – Automating operations like chatbots or content workflows—without needing a technical background.

✅ For example, we recently helped a European supplier expand into Thailand with a tailored go-to-market strategy and distributor network.
Just to better assist—are you looking to grow in Asia, improve sales, or explore automation?

If user replies with a focus area (e.g., “sales”):
Terra:
Great! Sales growth is one of our core strengths. Here’s how we usually help:
🔍 Evaluate your current sales process and lead generation.
🎯 Train your team in cold calling, account management, or service excellence.
📈 Track KPIs and refine based on results.

Would you like to explore our Sales Excellence Coaching options or connect with a human consultant to get started?

(Keep responses helpful, natural, and client-centered. Always offer a next step.)
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

       # LOG USER DATA HERE
    save_user_data(
        name=name,        
        email=email,
        phone=phone,
        country=country
    )
            
    return "✅ **Details saved!**"


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
