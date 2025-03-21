import streamlit as st
import streamlit.components.v1 as components
from database import UserDatabase
from dotenv import load_dotenv
import os
load_dotenv()
# Import exercises
from excerise_1 import exercise1
from excerise_2 import exercise2

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Set page config
st.set_page_config(
    page_title="AI BOOTCAMP Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
local_css("ai-bootcamp.css")

# Initialize the database
db = UserDatabase()

# Initialize session state variables
if "page" not in st.session_state:
    st.session_state.page = "home"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "show_login" not in st.session_state:
    st.session_state.show_login = False

def toggle_login_form():
    st.session_state.show_login = not st.session_state.show_login

def header_section():
    """Enhanced header with better visual structure"""
    st.markdown("""
    <div class="app-header">
        <h1>🤖 AI BOOTCAMP</h1>
        <p class="header-subtitle">Master AI concepts with hands-on exercises</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display login/guest buttons or logged in status
    if st.session_state.authenticated:
        st.markdown(f"""
        <div class="user-status">
            <span class="status-icon">👤</span> Logged in as: <strong>{st.session_state.username}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Logout", key="logout_btn"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Login", key="login_toggle_btn", use_container_width=True):
                toggle_login_form()
        with col2:
            if st.button("Continue as Guest", key="guest_btn", use_container_width=True):
                st.session_state.username = "guest"
                st.session_state.show_login = False
                st.rerun()

def login_form():
    """Enhanced login form"""
    if not st.session_state.authenticated and st.session_state.show_login:
        st.markdown("""
        <div class="login-header">
            <h2>Sign in to your account</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form", border=False):
            col1, col2 = st.columns(2)
            
            with col1:
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                
                submit_btn = st.form_submit_button("Sign In", use_container_width=True)
                
                if submit_btn:
                    if db.verify_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.show_login = False
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials!")
            
            with col2:
                st.markdown("""
                <div class="login-benefits">
                    <h3>Benefits of logging in:</h3>
                    <ul>
                        <li>Submit evaluation forms</li>
                        <li>Save your prompt and model responses</li>
                        <li>View overall results and analytics</li>
                        <li>Track your progress across modules</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

def footer_section():
    """Enhanced footer with better styling"""
    st.markdown("""
    <div class="app-footer">
        <p>© 2025 AI BOOTCAMP Lab. All rights reserved.</p>
        <div class="footer-links">
            <a href="#">Documentation</a> | 
            <a href="#">Support</a> | 
            <a href="#">Privacy Policy</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_module_card(title, description, icon, target_page):
    """Create an enhanced card for module or week with the entire card clickable"""
    
    # Create the card content
    card_content = f"""
    <div class="card-icon">{icon}</div>
    <h3>{title}</h3>
    <p>{description}</p>
    <div class="module-card-footer">
        <div class="card-action">Explore this module</div>
    </div>
    """
    
    # Use a simple clickable container instead of a form
    col1, col2 = st.columns([9, 1])  # 9:1 ratio to make most of the area clickable
    
    with col1:
        st.markdown(f'<div class="module-card">{card_content}</div>', unsafe_allow_html=True)
    
    with col2:
        if st.button(":arrow_right: ", key=f"btn_{target_page}", help="Navigate to this module"):
            st.session_state.page = target_page
            st.rerun()
            
    # Add JavaScript to make the card click the button
    st.markdown(f"""
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Find the module card and the corresponding button
            const cards = document.querySelectorAll('.module-card');
            cards.forEach(card => {{
                card.style.cursor = 'pointer';
                card.addEventListener('click', function() {{
                    // Find the closest button and click it
                    const buttonContainer = card.closest('.row-widget').nextElementSibling;
                    if (buttonContainer) {{
                        const button = buttonContainer.querySelector('button');
                        if (button) button.click();
                    }}
                }});
            }});
        }});
    </script>
    """, unsafe_allow_html=True)

def create_exercise_card(title, description, icon, target_page):
    """Create an enhanced card for exercises with the entire card clickable"""
    
    # Create the card content
    card_content = f"""
    <div class="card-icon">{icon}</div>
    <h3>{title}</h3>
    <p>{description}</p>
    <div class="exercise-card-footer">
        <div class="card-action">Start Exercise</div>
    </div>
    """
    
    # Use a simple clickable container instead of a form
    col1, col2 = st.columns([9, 1])  # 9:1 ratio to make most of the area clickable
    
    with col1:
        st.markdown(f'<div class="exercise-card">{card_content}</div>', unsafe_allow_html=True)
    
    with col2:
        if st.button(":arrow_right:", key=f"ex_btn_{target_page}", help="Start this exercise"):
            st.session_state.page = target_page
            st.rerun()
            
    # Add JavaScript to make the card click the button
    st.markdown(f"""
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Find the exercise card and the corresponding button
            const cards = document.querySelectorAll('.exercise-card');
            cards.forEach(card => {{
                card.style.cursor = 'pointer';
                card.addEventListener('click', function() {{
                    // Find the closest button and click it
                    const buttonContainer = card.closest('.row-widget').nextElementSibling;
                    if (buttonContainer) {{
                        const button = buttonContainer.querySelector('button');
                        if (button) button.click();
                    }}
                }});
            }});
        }});
    </script>
    """, unsafe_allow_html=True)

def back_button():
    """Enhanced back button"""
    st.markdown('<div class="back-button-container">', unsafe_allow_html=True)
    
    if st.button(":arrow_left: Home", key=f"back_home_{st.session_state.page}"):
        st.session_state.page = "home"
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def wrap_exercise_content(func):
    """Wrapper for exercise content with enhanced styling"""
    def wrapper(*args, **kwargs):
        # Display header and login components
        header_section()
        login_form()
        
        # Add back button
        back_button()
        
        # Add a divider
        st.markdown("<hr class='content-divider'>", unsafe_allow_html=True)
        # Use more specific matching for function names
        if func.__name__ == "exercise1" or func.__name__ == "styled_exercise1":
            exercise_title = "Exercise 1: Model Comparison"
        elif func.__name__ == "exercise2" or func.__name__ == "styled_exercise2":
            exercise_title = "Exercise 2: Prompt Engineering"
        else:
            exercise_title = func.__name__.replace("styled_exercise", "Exercise ").replace("exercise", "Exercise ")
        
        # Set the page title to remove "styled_" from browser tab
        st.markdown(f"<title>AI BOOTCAMP - {exercise_title}</title>", unsafe_allow_html=True)
        
        # Display the exercise title in the page content
        st.markdown(f"<h2 class='exercise-title'>{exercise_title}</h2>", unsafe_allow_html=True)
        
        # Execute the exercise content
        func(*args, **kwargs)
        
        # Add footer
        footer_section()
    return wrapper 

# Wrap exercise functions
@wrap_exercise_content
def styled_exercise1():
    exercise1()

@wrap_exercise_content
def styled_exercise2():
    exercise2()

def navigate():
    """Main navigation logic with enhanced UI"""
    if st.session_state.page == "home":
        # Home page
        header_section()
        login_form()
        
        st.markdown("<h2>Welcome to AI Bootcamp</h2>", unsafe_allow_html=True)
        st.write("Explore our interactive modules and hands-on exercises to master AI concepts.")
        
        # Week cards in a grid layout
        col1, col2 = st.columns(2)
        
        with col1:
            create_module_card(
                "Week 1: LLM Performance Comparison", 
                "Compare different large language models and evaluate their performance on various tasks. Learn about benchmarking methodologies and how to interpret model results.",
                "🔍",
                "week1"
            )
        
        with col2:
            create_module_card(
                "Week 2: Prompt Engineering Techniques", 
                "Learn advanced prompt engineering techniques to get the best results from AI models. Discover how to craft effective prompts for different use cases.",
                "✨",
                "week2"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            create_module_card(
                "Week 3: Models", 
                "Explore methods to fine-tune models for specific applications and domains. Learn how to prepare datasets and evaluate fine-tuned models.",
                "🧠",
                "week3"
            )
        
        with col4:
            create_module_card(
                "Week 4: Building AI Applications", 
                "Build end-to-end applications leveraging AI capabilities. Learn how to integrate AI models into practical applications and deploy them effectively.",
                "🛠️",
                "week4"
            )
        
        footer_section()
    
    elif st.session_state.page == "week1":
        # Week 1 page
        header_section()
        login_form()
        
        back_button()
        
        st.markdown("<h2>Week 1: LLM Performance Comparison</h2>", unsafe_allow_html=True)
        st.write("Select an exercise to start practicing:")
        
        # Exercise cards
        col1, col2 = st.columns(2)
        with col1:
            create_exercise_card(
                "Exercise 1: Model Comparison", 
                "Compare different language models on various benchmarks and understand their strengths and weaknesses.",
                "📊",
                "exercise1"
            )
        
        with col2:
            create_exercise_card(
                "Exercise 2: Performance Analysis", 
                "Analyze and visualize model performance metrics to gain insights into model behavior across different tasks.",
                "📈",
                "exercise2"
            )
        
        col3, col4 = st.columns(2)
        with col3:
            create_exercise_card(
                "Exercise 3: Coming Soon", 
                "This exercise will be unlocked in the future. Stay tuned!",
                "🔜",
                "exercise3"
            )
        
        with col4:
            create_exercise_card(
                "Exercise 4: Coming Soon", 
                "This exercise will be unlocked in the future. Stay tuned!",
                "🔜",
                "exercise4"
            )
        
        footer_section()
    
    elif st.session_state.page == "week2":
        # Week 2 page with placeholder content
        header_section()
        login_form()
        
        back_button()
        
        st.markdown("<h2>Week 2: Prompt Engineering Techniques</h2>", unsafe_allow_html=True)
        
        # Placeholder content with nice styling
        st.markdown("""
        <div class="module-card">
            <div class="card-icon">🚧</div>
            <h3>Coming Soon</h3>
            <p>We're currently developing this module. Check back soon for exciting content on prompt engineering!</p>
        </div>
        """, unsafe_allow_html=True)
        
        footer_section()
    
    elif st.session_state.page == "week3":
        # Week 3 page with placeholder content
        header_section()
        login_form()
        
        back_button()
        
        st.markdown("<h2>Week 3: Models</h2>", unsafe_allow_html=True)
        
        # Placeholder content with nice styling
        st.markdown("""
        <div class="module-card">
            <div class="card-icon">🚧</div>
            <h3>Coming Soon</h3>
            <p>We're currently developing this module. Check back soon for in-depth content on fine-tuning models!</p>
        </div>
        """, unsafe_allow_html=True)
        
        footer_section()
    
    elif st.session_state.page == "week4":
        # Week 4 page with placeholder content
        header_section()
        login_form()
        
        back_button()
        
        st.markdown("<h2>Week 4: Building AI Applications</h2>", unsafe_allow_html=True)
        
        # Placeholder content with nice styling
        st.markdown("""
        <div class="module-card">
            <div class="card-icon">🚧</div>
            <h3>Coming Soon</h3>
            <p>We're currently developing this module. Check back soon for practical content on building AI applications!</p>
        </div>
        """, unsafe_allow_html=True)
        
        footer_section()
    
    elif st.session_state.page == "exercise1":
        # Call the wrapped exercise1 function
        styled_exercise1()
    
    elif st.session_state.page == "exercise2":
        # Call the wrapped exercise2 function
        styled_exercise2()
        
    elif st.session_state.page == "exercise3":
        # Coming soon exercise placeholder
        header_section()
        login_form()
        
        back_button()
        
        st.markdown("<h2 class='exercise-title'>Exercise 3: Coming Soon</h2>", unsafe_allow_html=True)
        
        # Placeholder content with nice styling
        st.markdown("""
        <div class="module-card">
            <div class="card-icon">🔜</div>
            <h3>Under Development</h3>
            <p>We're currently developing this exercise content. Check back soon for new learning opportunities!</p>
        </div>
        """, unsafe_allow_html=True)
        
        footer_section()
    
    elif st.session_state.page == "exercise4":
        # Coming soon exercise placeholder
        header_section()
        login_form()
        
        back_button()
        
        st.markdown("<h2 class='exercise-title'>Exercise 4: Coming Soon</h2>", unsafe_allow_html=True)
        
        # Placeholder content with nice styling
        st.markdown("""
        <div class="module-card">
            <div class="card-icon">🔜</div>
            <h3>Under Development</h3>
            <p>We're currently developing this exercise content. Check back soon for new learning opportunities!</p>
        </div>
        """, unsafe_allow_html=True)
        
        footer_section()

# Start the app
navigate()