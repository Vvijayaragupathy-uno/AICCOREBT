from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from groq import Groq
import os
from database import UserDatabase
import json
import plotly.express as px
import plotly.graph_objects as go
import io
import PyPDF2
from dotenv import load_dotenv
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
github_key = os.getenv("GITHUB_API_KEY")
def exercise2():
    # Initialize the user database
    user_db = UserDatabase()
    
    # HEADER
    st.markdown("<h1 style='text-align: center; font-size: 20px;'>AI BOOTCAMP Lab - Exercise 2</h1>", unsafe_allow_html=True)
    
    # Define LLM models
    llm_models = [
        "llama3-70b-8192", "gemma2-9b-it", "mistral-saba-24b", "qwen-2.5-32b",
        "deepseek-r1-distill-qwen-32b", "allam-2-7b","llama-3.2-90b-vision-preview"
    ]
    prompt_techniques = {
        "Zero Shot": "Answer the following question without any examples.",
        "Few Shot": {
            "📱 Mobile Security Policy": [
                '''Here are some examples to guide your response: 
                Question: I have a Nokia 2017 that I want to use as a personal device and connect to university devices. Would using this device violate the mobile-device policy? 
                Answer: Yes.''',

                '''Question: My department plans to use personal mobile devices for accessing confidential university data. These devices are running outdated software and do not support the latest security updates. Would using these devices violate the mobile-device policy? 
                Answer: Yes.''',

                '''Question: I want to use my personal tablet to access university resources. The device has up-to-date security patches, strong passwords, and mobile-device management installed as per policy. Would using this device violate the mobile-device policy? 
                Answer: No.''',

                "Now, answer the following question."
            ],
            "📂 Data Ownership Scenario": [
                '''Here are some examples to guide your response: 
                Question 1: I'd like my department to retain confidential university documents on their personal devices for one year after projects end. Is this permitted? 
                Answer: No.''',

                '''Question 2: Our research team wants to store project data collected through university apps on their devices for six months. Would this comply with the policy? 
                Answer: No.''',

                '''Question 3: We need to keep a temporary copy of publicly available university policies on personal devices for reference during a conference. Would this violate the data-ownership policy? 
                Answer: No.''',

                "Now, answer the following question."
            ],
            "🔒 Security Breach Protocol": [
                '''Here are some examples to guide your response: 
                Question: An employee's mobile device starts exhibiting unusual behavior. The battery drains quickly even when not in use. The device connects repeatedly to IP addresses in countries where the company doesn't operate. Apps the employee never installed appear in the app drawer, including one that requests unusual permissions. Does this device show clear evidence of being compromised? 
                Answer: Yes.''',

                '''Question: An employee's mobile device starts exhibiting unusual behavior. The battery depletes unusually fast. Network logs show the device connecting to unrecognized servers with suspicious domain names. The security scan reveals hidden processes running in the background and unauthorized modifications to system files. Several unfamiliar applications with administrator privileges appear to be installed without the employee's knowledge. Does this device show clear evidence of being compromised? 
                Answer: Yes.''',

                '''Question: A user's mobile device suddenly runs slower, but no unexpected network activity, unauthorized apps, or security alerts have been observed. Could this be clear evidence of being compromised? 
                Answer: No.''',

                "Now, answer the following question."
            ]
        },
        "Role Based": "You are an expert in cybersecurity and mobile device management policies. You are a university IT compliance officer evaluating device requests. Please give me only yes or no as the answer to the following question.",
        "Chain of Thought": "Think step by step to solve the following problem. Break down your reasoning into clear logical steps."
    }
    
    # Define scenarios
    scenarios = {
            "📱 Mobile Security Policy": {
                "question": '''Question: I have a Nokia 2017 that I want to use as a personal device and connect to university devices. Would using this device violate the mobile-device policy? Please give me a yes or no answer.''',
            },
            "📂 Data Ownership Scenario": {
                "question": '''I want to allow my team to store work emails on their devices for up to three months to reference older communications as per Mobile device management policy. Can you please give yes or no answer?''',
            },
            "🔒 Security Breach Protocol": {
                "question": '''An employee's mobile device starts exhibiting. The device shows increased battery usage. It initiates several connections to various IP addresses. Some unfamiliar apps appear to be installed. Does this device show clear evidence of being compromised? Can you give the response as only no or yes?''',
            }
        }
    ANALYTICS_FOLDER = "Exercise/Exercise2"
    if not os.path.exists(ANALYTICS_FOLDER):
        os.makedirs(ANALYTICS_FOLDER)
    
    # Path to the analytics data file
    ANALYTICS_FILE = f"{ANALYTICS_FOLDER}/prompt_model_analytics.csv"
    
    # Function to save analytics data - MOVED UP from tab3
    def save_analytics_data():
        # Get data from both tabs
        tab1_data = []
        tab2_data = []
        
        # Path to the analytics data file
        ANALYTICS_FILE = f"{ANALYTICS_FOLDER}/prompt_model_analytics.csv"
        
        # Process tab1 data if available
        if hasattr(st.session_state, 'model_responses_tab1') and st.session_state.model_responses_tab1:
            # Get scenario name
            scenario_name = st.session_state.get('selected_scenario_name', 
                                            selected_scenario_name if 'selected_scenario_name' in locals() else "Unknown")
            
            base_data_tab1 = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'username': st.session_state.username if hasattr(st.session_state, 'username') else "anonymous",
                'scenario': scenario_name,
                'temperature': st.session_state.temperature,
                'max_tokens': st.session_state.max_tokens,
                'tab': 'Prompt Comparison',
                'model': st.session_state.selected_model,
                'prompt': st.session_state.get('selected_scenario', "Unknown prompt")  # Store actual prompt
            }
            
            for technique, response_data in st.session_state.model_responses_tab1.items():
                row_data = base_data_tab1.copy()
                row_data['prompt_technique'] = technique
                row_data['response'] = response_data["response"]
                row_data['token_count'] = response_data["token_count"]
                tab1_data.append(row_data)
        
        # Process tab2 data if available
        if hasattr(st.session_state, 'model_responses_tab2') and st.session_state.model_responses_tab2:
            # Get scenario name
            scenario_name = st.session_state.get('selected_scenario_name', 
                                            selected_scenario_name if 'selected_scenario_name' in locals() else "Unknown")
            
            base_data_tab2 = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'username': st.session_state.username if hasattr(st.session_state, 'username') else "anonymous",
                'scenario': scenario_name,
                'temperature': st.session_state.temperature,
                'max_tokens': st.session_state.max_tokens,
                'tab': 'Model Comparison',
                'prompt_technique': st.session_state.get('selected_prompt_technique', 
                                                        selected_prompt_technique if 'selected_prompt_technique' in locals() else "Unknown"),
                'prompt': st.session_state.get('selected_scenario', "Unknown prompt")  # Store actual prompt
            }
            
            for model, response_data in st.session_state.model_responses_tab2.items():
                row_data = base_data_tab2.copy()
                row_data['model'] = model
                row_data['response'] = response_data["response"]
                row_data['token_count'] = response_data["token_count"]
                tab2_data.append(row_data)
        
        # Combine data from both tabs
        combined_data = tab1_data + tab2_data
        
        if not combined_data:
            st.warning("No data to save. Please run a comparison first.")
            return
        
        # Create DataFrame from the combined data
        new_data = pd.DataFrame(combined_data)
        
        # Append to existing file or create new one
        if os.path.exists(ANALYTICS_FILE):
            try:
                existing_data = pd.read_csv(ANALYTICS_FILE)
                updated_data = pd.concat([existing_data, new_data], ignore_index=True)
                updated_data.to_csv(ANALYTICS_FILE, index=False)
                
            except Exception as e:
                st.error(f"Error updating analytics data: {e}")
                new_data.to_csv(ANALYTICS_FILE, index=False)
                
        else:
            new_data.to_csv(ANALYTICS_FILE, index=False)
           


    # Initialize session state variables
    if "chatHist" not in st.session_state:
        st.session_state.chatHist = {}
        # Initialize chatHist with nested dictionaries for each model
        for model in llm_models:
            st.session_state.chatHist[model] = {}

    # Helper function to ensure chat structure exists
    def ensure_valid_chat_structure(model, technique):
        """Ensure the chat history structure exists and is valid for a given model and technique."""
        # Ensure model exists as a dictionary
        if model not in st.session_state.chatHist:
            st.session_state.chatHist[model] = {}
        
        # Check if the model value is actually a dictionary
        if not isinstance(st.session_state.chatHist[model], dict):
            # If not, fix it by recreating it as a dictionary
            st.session_state.chatHist[model] = {}
        
        # Ensure technique exists for this model
        if technique not in st.session_state.chatHist[model]:
            st.session_state.chatHist[model][technique] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
    
    if "selected_scenario_name" not in st.session_state:
        st.session_state.selected_scenario_name = None
    if "selected_prompt_technique" not in st.session_state:
        st.session_state.selected_prompt_technique = None
    if "analytics_saved_ex2" not in st.session_state:
        st.session_state.analytics_saved_ex2 = False
    if "model_responses_tab1" not in st.session_state:
        st.session_state.model_responses_tab1 = {}
    if "model_responses_tab2" not in st.session_state:
        st.session_state.model_responses_tab2 = {}
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = llm_models[0]  # Default to the first model
    if "temperature" not in st.session_state:
        st.session_state.temperature = 1.0
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 300
    if "selected_scenario" not in st.session_state:
        st.session_state.selected_scenario = None
    if "uploaded_file_content" not in st.session_state:
        st.session_state.uploaded_file_content = None
    if "model_responses" not in st.session_state:
        st.session_state.model_responses = {}
    
    # Display login status in sidebar
    if st.session_state.username != "guest":
        st.sidebar.success(f"Logged in as: {st.session_state.username}")
    else:
        st.sidebar.info("Using as guest (responses won't be saved)")
 
   
    tab1, tab2, tab3, tab4 = st.tabs(["Prompt Selection", "Model selection", "Data Analysis and visualization","Voting Analysis"])

    with tab1:
        colHeader = st.columns([2,2])
        with colHeader[0]:
            st.header("Prompt Engineering Comparison")
        with colHeader[1]:
            st.markdown("Upload Document")
            uploaded_file = st.file_uploader("Choose a document", type=["txt", "pdf", "docx"], label_visibility="hidden")
            if uploaded_file is not None:
                st.write("Filename:", uploaded_file.name)
                
                # Check if it's a PDF file
                if uploaded_file.name.lower().endswith('.pdf'):
                    try:
                        # Create a BytesIO object from the uploaded file's content
                        pdf_bytes = io.BytesIO(uploaded_file.getvalue())
                        
                        # Create a PDF reader object
                        pdf_reader = PyPDF2.PdfReader(pdf_bytes)
                        
                        # Extract text from all pages
                        text_content = ""
                        total_pages = len(pdf_reader.pages)
                        
                        # Show progress to user
                        with st.spinner(f"Extracting text from {total_pages} pages..."):
                            for page_num in range(total_pages):
                                page = pdf_reader.pages[page_num]
                                text_content += page.extract_text() + "\n\n"
                        
                        # Apply a character limit to prevent token limit errors
                        MAX_CHARS = 4000
                        if len(text_content) > MAX_CHARS:
                            st.warning(f"PDF content is too large ({len(text_content)} characters). Truncating to {MAX_CHARS} characters.")
                            text_content = text_content[:MAX_CHARS] + "\n[Content truncated due to size limits]"
                        
                        st.session_state.uploaded_file_content = text_content
                        
                        # Show a preview of the extracted text
                        with st.expander("Preview Extracted Text"):
                            st.text(text_content[:500] + ("..." if len(text_content) > 500 else ""))
                            st.info(f"Successfully extracted {len(text_content)} characters from PDF.")
                            
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
                        # Set a basic placeholder to avoid None errors later
                        st.session_state.uploaded_file_content = f"Error extracting text from {uploaded_file.name}: {str(e)}"
                
                elif uploaded_file.name.lower().endswith('.docx'):
                    # For DOCX files, you could use python-docx library
                    # But for now, inform the user about limitations
                    st.warning("DOCX files are supported with basic text extraction only.")
                    try:
                        # Simple text decoding for DOCX
                        file_content = uploaded_file.read().decode("utf-8", errors="ignore")
                        
                        # Apply a character limit
                        MAX_CHARS = 4000
                        if len(file_content) > MAX_CHARS:
                            st.warning(f"File content is too large ({len(file_content)} characters). Truncating to {MAX_CHARS} characters.")
                            file_content = file_content[:MAX_CHARS] + "\n[Content truncated due to size limits]"
                        
                        st.session_state.uploaded_file_content = file_content
                    except Exception as e:
                        st.error(f"Error reading DOCX file: {str(e)}")
                        st.session_state.uploaded_file_content = f"Error reading {uploaded_file.name}: {str(e)}"
                
                else:
                    # For TXT and other files, read normally
                    try:
                        file_content = uploaded_file.read().decode("utf-8", errors="ignore")
                        
                        # Apply a character limit
                        MAX_CHARS = 4000
                        if len(file_content) > MAX_CHARS:
                            st.warning(f"File content is too large ({len(file_content)} characters). Truncating to {MAX_CHARS} characters.")
                            file_content = file_content[:MAX_CHARS] + "\n[Content truncated due to size limits]"
                        
                        st.session_state.uploaded_file_content = file_content
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
                        st.session_state.uploaded_file_content = f"Error reading {uploaded_file.name}: {str(e)}"

                # Save the uploaded file to a folder (unchanged)
                if not os.path.exists("Exercise/uploads"):
                    os.makedirs("Exercise/uploads")
                # Reset the file pointer position before saving
                uploaded_file.seek(0)
                with open(os.path.join("Exercise/uploads", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            else:
                st.write("No document uploaded. You can ask questions without a document.")

        
                
        # DROPDOWN FOR SINGLE MODEL SELECTION
        st.session_state.selected_model = st.selectbox(
            "Choose a model:",
            llm_models,
            index=llm_models.index(st.session_state.selected_model) if st.session_state.selected_model in llm_models else 0,
            key="model_selector"
        )
        
        # Initialize session state for tab1 selections
        if "previous_selection_tab1" not in st.session_state:
            st.session_state.previous_selection_tab1 = {
                "model": "",
                "techniques": [],
                "scenario": ""
            }
        if "is_comparing_tab1" not in st.session_state:
            st.session_state.is_comparing_tab1 = False
        if "comparison_complete_tab1" not in st.session_state:
            st.session_state.comparison_complete_tab1 = False
            
        # MULTI-SELECT FOR PROMPT TECHNIQUES
        selected_techniques = st.multiselect(
            "Choose prompt techniques to compare:",
            options=list(prompt_techniques.keys()),
            default=["Zero Shot"],  
            key="simple_prompt_selector"
        )

        # Ensure at least one technique is selected
        if not selected_techniques:
            st.warning("Please select at least one prompt technique.")
            selected_techniques = ["Zero Shot"]
            
        # Update session state
        st.session_state.prompt_techniques_tab1 = selected_techniques.copy()
        
        with colHeader[1]: # Model Settings in a popover
            with st.popover("⚙️ Model Settings"):
                # Temperature Slider
                st.session_state.temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.temperature,
                    step=0.1,
                    help="Controls the randomness of the model's responses. Lower values make the model more deterministic."
                )

                # Max Tokens Slider
                st.session_state.max_tokens = st.slider(
                    "Max Tokens",
                    min_value=200,
                    max_value=8192,
                    value=st.session_state.max_tokens,
                    step=100,
                    help="Controls the maximum number of tokens the model will generate."
                )

                # Maximum Prompt Techniques to Compare
                max_techniques_to_compare = st.slider(
                    "Maximum Prompt Techniques to Compare",
                    min_value=1,
                    max_value=4,
                    value=4,
                    step=1,
                    help="Limit the number of prompt techniques that can be compared for better viewing experience."
                )

        # Check if too many techniques are selected
        if len(selected_techniques) > max_techniques_to_compare:
            st.warning(f"You can compare a maximum of {max_techniques_to_compare} prompt techniques. Please deselect some techniques.")
            # Truncate the selection for UI display only
            prompt_techniques_to_use = selected_techniques[:max_techniques_to_compare]
        else:
            prompt_techniques_to_use = selected_techniques

        # Scenario Selection Header
        st.write("📌 Select the Scenario:")

        # Radio button for scenario selection
        scenario_options = list(scenarios.keys())
        selected_scenario_name = st.radio(
            "Select a scenario:", 
            scenario_options,
            index=0 if st.session_state.selected_scenario else 0  # Default to first scenario
        )
        
        if selected_scenario_name:
            # Display the selected scenario details
            st.markdown(f"**Question:** {scenarios[selected_scenario_name]['question']}")
            
            # Store the selected scenario in session state
            st.session_state.selected_scenario = scenarios[selected_scenario_name]["question"]
        else:
            st.session_state.selected_scenario = None

        # Check if selection has changed
        current_selection = {
            "model": st.session_state.selected_model,
            "techniques": st.session_state.prompt_techniques_tab1,
            "scenario": selected_scenario_name
        }
        
        selection_changed = (
            current_selection["model"] != st.session_state.previous_selection_tab1["model"] or
            current_selection["techniques"] != st.session_state.previous_selection_tab1["techniques"] or
            current_selection["scenario"] != st.session_state.previous_selection_tab1["scenario"]
        )
        
        # Comparison section
        st.subheader("Compare Prompt Techniques Side by Side")
        
        # Check if we have techniques and a scenario selected
        if not prompt_techniques_to_use:
            st.info("👈 Please select at least one prompt technique to use")
        elif not st.session_state.selected_scenario:
            st.info("👈 Please select a scenario first")
        else:
            # Trigger automatic comparison if selection changed
            if selection_changed and prompt_techniques_to_use and st.session_state.selected_scenario:
                st.session_state.is_comparing_tab1 = True
                st.session_state.comparison_complete_tab1 = False
                if "model_responses_tab1" not in st.session_state:
                    st.session_state.model_responses_tab1 = {}
                else:
                    st.session_state.model_responses_tab1 = {}  # Reset responses
                
                st.session_state.previous_selection_tab1 = current_selection.copy()  # Update previous selection
                
                # Show a message that comparison is starting automatically
                st.info("Selection changed. Starting comparison automatically...")
                st.rerun()
            
            # Add a manual refresh button
            
                
            # Create columns for each prompt technique
            cols = st.columns(len(prompt_techniques_to_use))
            
            # Display technique responses or process comparisons
            for i, technique in enumerate(prompt_techniques_to_use):
                with cols[i]:  # Assign each technique to a column
                    st.markdown(f"### {technique}")  # Technique Name
                    
                    # If we're still comparing and this technique doesn't have a response yet
                    if (st.session_state.is_comparing_tab1 and 
                        (not hasattr(st.session_state, 'model_responses_tab1') or 
                         technique not in st.session_state.model_responses_tab1)):
                        
                        with st.spinner(f"Getting {technique} response..."):
                            try:
                                # Initialize the Groq client
                                client = Groq(api_key=groq_key)

                                # Create a complete prompt based on the technique
                                full_prompt = ""
                                
                                # Add uploaded content if available
                                if st.session_state.uploaded_file_content:
                                    full_prompt = st.session_state.uploaded_file_content + "\n\n"
                                
                                # Add technique-specific instructions
                                if technique == "Few Shot":
                                    # Check if the selected scenario matches a key in the Few Shot dictionary
                                    if selected_scenario_name in prompt_techniques["Few Shot"]:
                                        # Get the examples for the selected scenario
                                        examples = prompt_techniques["Few Shot"][selected_scenario_name]
                                        full_prompt += "\n".join(examples) + "\n\n"
                                    else:
                                        st.warning(f"No examples found for the selected scenario: {selected_scenario_name}")
                                else:
                                    full_prompt += prompt_techniques[technique] + "\n\n"
                                
                                # Add the scenario question
                                if st.session_state.selected_scenario:
                                    full_prompt += st.session_state.selected_scenario
                                
                                # Make sure the technique key exists in the chat history for this model
                                model = st.session_state.selected_model
                                
                                # Store selected scenario name in session state
                                st.session_state.selected_scenario_name = selected_scenario_name
                                
                                # Use the helper function to ensure the structure exists
                                ensure_valid_chat_structure(model, technique)

                                # Add the user message to chat history
                                st.session_state.chatHist[model][technique].append(
                                    {"role": "user", "content": full_prompt}
                                )
                                
                                # Fetch response from Groq API
                                response = client.chat.completions.create(
                                    model=model,
                                    messages=st.session_state.chatHist[model][technique],
                                    max_tokens=st.session_state.max_tokens,
                                    temperature=st.session_state.temperature
                                )
                                assistant_response = response.choices[0].message.content
                                
                                # Calculate token count (approximate)
                                token_count = len(assistant_response.split())

                                # Store the response
                                if not hasattr(st.session_state, 'model_responses_tab1'):
                                    st.session_state.model_responses_tab1 = {}
                                    
                                st.session_state.model_responses_tab1[technique] = {
                                    "response": assistant_response,
                                    "token_count": token_count,
                                    "prompt": full_prompt  # Store the full prompt used
                                }
                                
                                # Append the response to the chat history
                                st.session_state.chatHist[model][technique].append(
                                    {"role": "assistant", "content": assistant_response}
                                )
                                
                                # Update session state
                                if len(st.session_state.model_responses_tab1) == len(prompt_techniques_to_use):
                                    st.session_state.is_comparing_tab1 = False
                                    st.session_state.comparison_complete_tab1 = True
                                    
                                    # Call save_analytics_data without condition
                                    save_analytics_data()
                                    
                                # Rerun to update UI
                                st.rerun()
                                    
                            except Exception as e:
                                st.error(f"Error: {e}", icon="🚨")
                                if not hasattr(st.session_state, 'model_responses_tab1'):
                                    st.session_state.model_responses_tab1 = {}
                                    
                                st.session_state.model_responses_tab1[technique] = {
                                    "response": f"Error: {str(e)}",
                                    "token_count": 0,
                                    "prompt": full_prompt if 'full_prompt' in locals() else "Error occurred before creating prompt"
                                }
                                
                                if len(st.session_state.model_responses_tab1) == len(prompt_techniques_to_use):
                                    st.session_state.is_comparing_tab1 = False
                                    st.session_state.comparison_complete_tab1 = True
                                                
                    # Display response if available
                    if hasattr(st.session_state, 'model_responses_tab1') and technique in st.session_state.model_responses_tab1:
                        response_data = st.session_state.model_responses_tab1[technique]
                        response = response_data["response"]
                        token_count = response_data["token_count"]
                        
                        # Display metrics
                        st.metric(
                            label="Token Count",
                            value=token_count
                        )
                        
                        # Display the response in a scrollable container
                        st.markdown(f'<div style="max-height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; border-radius: 5px;">{response}</div>', unsafe_allow_html=True)
                    else:
                        # Show loading spinner if we're still waiting for this technique's response
                        with st.spinner(f"Getting {technique} response..."):
                            st.info("Loading response...")
                            
            # Save responses to CSV file if the user is authenticated and not a guest
            if len(st.session_state.model_responses_tab1) == len(prompt_techniques_to_use):
                st.session_state.is_comparing_tab1 = False
                st.session_state.comparison_complete_tab1 = True
                save_analytics_data()
            if st.button("🔄 Refresh Comparison", key="refresh_comparison_tab1"):
                st.session_state.is_comparing_tab1 = True
                st.session_state.comparison_complete_tab1 = False
                st.session_state.model_responses_tab1 = {}
                st.rerun()

    #tab2
    #tab2
    with tab2:
        st.header("Model Comparison with Single Prompt Technique")   

        # Model Settings in a popover
        with st.popover("⚙️ Model Settings"):
            # Temperature Slider with a unique key
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Controls the randomness of the model's responses. Lower values make the model more deterministic.",
                key="temperature_slider_tab2"  # Unique key for Tab 2
            )

            # Max Tokens Slider with a unique key
            st.session_state.max_tokens = st.slider(
                "Max Tokens",
                min_value=200,
                max_value=8192,
                value=st.session_state.max_tokens,
                step=100,
                help="Controls the maximum number of tokens the model will generate.",
                key="max_tokens_slider_tab2"  # Unique key for Tab 2
            )
            
        # Initialize session state for tab2 if not already done
        if "selected_models" not in st.session_state:
            st.session_state.selected_models = [llm_models[0]]
        if "previous_selection_tab2" not in st.session_state:
            st.session_state.previous_selection_tab2 = {
                "models": [],
                "technique": "",
                "scenario": ""
            }
        if "comparison_complete" not in st.session_state:
            st.session_state.comparison_complete = False
            
        # Allow the user to select up to four models
        selected_models = st.multiselect(
            "Choose up to four models to compare:",
            llm_models,
            default=st.session_state.selected_models if st.session_state.selected_models else [llm_models[0]],
            key="model_selector_tab2"
        )
        
        # Update session state with selected models
        st.session_state.selected_models = selected_models

        # Limit the number of selected models to four
        if len(selected_models) > 4:
            st.warning("You can compare a maximum of four models. Please deselect some models.")
            st.session_state.selected_models = selected_models[:4]

        # Allow the user to select one prompt technique
        selected_prompt_technique = st.selectbox(
            "Choose a prompt technique:",
            list(prompt_techniques.keys()),
            key="prompt_technique_selector_tab2"
        )

        # Allow the user to select a scenario
        selected_scenario_name = st.radio(
            "Select a scenario:",
            list(scenarios.keys()),
            key="scenario_selector_tab2"
        )

        # Display the selected scenario question
        if selected_scenario_name:
            st.subheader("Selected Scenario")
            st.write(scenarios[selected_scenario_name]["question"])
            st.session_state.selected_scenario = scenarios[selected_scenario_name]["question"]
        else:
            st.session_state.selected_scenario = None

        # Check if selection has changed
        current_selection = {
            "models": st.session_state.selected_models,
            "technique": selected_prompt_technique,
            "scenario": selected_scenario_name
        }
        
        selection_changed = (
            current_selection["models"] != st.session_state.previous_selection_tab2["models"] or
            current_selection["technique"] != st.session_state.previous_selection_tab2["technique"] or
            current_selection["scenario"] != st.session_state.previous_selection_tab2["scenario"]
        )
        
        # Check if we have models and a scenario selected
        if not st.session_state.selected_models:
            st.info("👈 Please select models to compare first")
        elif not st.session_state.selected_scenario:
            st.info("👈 Please select a scenario first")
        else:
            # Trigger automatic comparison if selection changed
            if selection_changed and st.session_state.selected_models and st.session_state.selected_scenario:
                st.session_state.is_comparing = True
                st.session_state.comparison_complete = False
                st.session_state.model_responses_tab2 = {}  # Reset responses
                st.session_state.previous_selection_tab2 = current_selection.copy()  # Update previous selection
                
                # Show a message that comparison is starting automatically
                st.info("Selection changed. Starting comparison automatically...")
                st.rerun()
            
            # Create columns for each model
            if st.session_state.is_comparing or st.session_state.comparison_complete:
                # Create columns for each model
                cols = st.columns(len(st.session_state.selected_models))
                
                # Display model responses or loading indicators
                for i, model in enumerate(st.session_state.selected_models):
                    with cols[i]:
                        st.subheader(model)
                        
                        # If we're still comparing and this model doesn't have a response yet
                        if st.session_state.is_comparing and (not hasattr(st.session_state, 'model_responses_tab2') or model not in st.session_state.model_responses_tab2):
                    
                            try:
                                # Initialize Groq client
                                client = Groq(api_key=groq_key)
                                
                                # Create a complete prompt based on the selected technique
                                full_prompt = ""
                                
                                # Add technique-specific instructions
                                if selected_prompt_technique == "Few Shot":
                                    # Check if the selected scenario matches a key in the Few Shot dictionary
                                    if selected_scenario_name in prompt_techniques["Few Shot"]:
                                        # Get the examples for the selected scenario
                                        examples = prompt_techniques["Few Shot"][selected_scenario_name]
                                        full_prompt += "\n".join(examples) + "\n\n"
                                    else:
                                        st.warning(f"No examples found for the selected scenario: {selected_scenario_name}")
                                else:
                                    full_prompt += prompt_techniques[selected_prompt_technique] + "\n\n"
                                
                                # Add the scenario question
                                full_prompt += st.session_state.selected_scenario
                                
                                # Store selected info in session state for saving later
                                st.session_state.selected_scenario_name = selected_scenario_name
                                st.session_state.selected_prompt_technique = selected_prompt_technique
                                
                                # Prepare messages
                                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                                
                                # Add uploaded document if available
                                if hasattr(st.session_state, 'uploaded_file_content') and st.session_state.uploaded_file_content:
                                    messages.append({"role": "user", "content": f"Please use this information as context: {st.session_state.uploaded_file_content}"})
                                
                                # Add the prompt
                                messages.append({"role": "user", "content": full_prompt})
                                
                                # Make API call
                                response = client.chat.completions.create(
                                    model=model,
                                    messages=messages,
                                    max_tokens=st.session_state.max_tokens,
                                    temperature=st.session_state.temperature
                                )
                                
                                # Get response
                                assistant_response = response.choices[0].message.content
                                
                                # Calculate token count (approximate)
                                token_count = len(assistant_response.split())
                                
                                # Store the response
                                if not hasattr(st.session_state, 'model_responses_tab2'):
                                    st.session_state.model_responses_tab2 = {}
                                    
                                st.session_state.model_responses_tab2[model] = {
                                    "response": assistant_response,
                                    "token_count": token_count,
                                    "prompt": full_prompt  # Store the full prompt used
                                }
                                
                                # Update session state
                                if len(st.session_state.model_responses_tab2) == len(st.session_state.selected_models):
                                    st.session_state.is_comparing = False
                                    st.session_state.comparison_complete = True
                                    
                                    # Call save_analytics_data without condition
                                    save_analytics_data()
                                
                                # Rerun to update UI
                                st.rerun()
                                
                            except Exception as e:
                                if not hasattr(st.session_state, 'model_responses_tab2'):
                                    st.session_state.model_responses_tab2 = {}
                                    
                                st.session_state.model_responses_tab2[model] = {
                                    "response": f"Error: {str(e)}",
                                    "token_count": 0,
                                    "prompt": full_prompt if 'full_prompt' in locals() else "Error occurred before creating prompt"
                                }
                                
                                if len(st.session_state.model_responses_tab2) == len(st.session_state.selected_models):
                                    st.session_state.is_comparing = False
                                    st.session_state.comparison_complete = True
                        
                        # Display response if available
                        if hasattr(st.session_state, 'model_responses_tab2') and model in st.session_state.model_responses_tab2:
                            response_data = st.session_state.model_responses_tab2[model]
                            response = response_data["response"]
                            token_count = response_data["token_count"]
                            
                            # Display metrics
                            st.metric(
                                label="Token Count",
                                value=token_count
                            )
                            
                            # Display the response in a scrollable container
                            st.markdown(f'<div style="max-height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; border-radius: 5px;">{response}</div>', unsafe_allow_html=True)
                        else:
                            # Show loading spinner if we're still waiting for this model's response
                            with st.spinner(f"Getting response from {model}..."):
                                st.info("Loading response...")
                
                # Save responses to database if all responses are collected and user is authenticated
                if len(st.session_state.model_responses_tab2) == len(st.session_state.selected_models):
                    st.session_state.is_comparing = False
                    st.session_state.comparison_complete = True
                    save_analytics_data()
                
                # Add a manual refresh button
                if st.button("🔄 Refresh Comparison", key="refresh_comparison"):
                    st.session_state.is_comparing = True
                    st.session_state.comparison_complete = False
                    st.session_state.model_responses_tab2 = {}
                    if hasattr(st.session_state, 'results_saved'):
                        delattr(st.session_state, 'results_saved')
                    st.rerun()




    #tab3 
    with tab3:
        st.header("Prompt Engineering & Model Analysis")
        
        # Function to load analytics data
        def load_analytics_data():
            if os.path.exists(ANALYTICS_FILE):
                try:
                    data = pd.read_csv(ANALYTICS_FILE)
                    # Convert timestamp strings to datetime objects
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    return data
                except Exception as e:
                    st.error(f"Error loading analytics data: {e}")
                    return pd.DataFrame()
            else:
                return pd.DataFrame()
        
        # Save current analytics data if available
        save_analytics_data()
        
        # Load all analytics data
        analytics_data = load_analytics_data()
        
        if analytics_data.empty:
            st.info("No analytics data available yet. Complete prompt technique or model comparisons to generate data.")
        else:
            # Brief introduction about data analytics
            with st.expander("📊 Why Prompt Engineering & Model Analysis Matter", expanded=True):
                st.markdown("""
                ### The Impact of Prompt Engineering & Model Selection
                
                Analyzing different prompt techniques and model performances helps you:
                
                - **Optimize Prompt Strategies**: Discover which prompt techniques deliver the best results for specific scenarios
                - **Compare Model Performance**: Identify which models excel at different types of questions
                - **Improve Response Quality**: Find the optimal combination of model, prompt technique, and parameters
                - **Reduce Token Usage**: Balance response quality with computational efficiency
                - **Enhance User Experience**: Tailor AI interactions based on data-driven insights
                
                The visualizations below provide actionable insights into your prompt engineering and model selection strategies.
                """)
            
            # Create two columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Prompt technique effectiveness chart
                st.subheader("Prompt Technique Effectiveness")
                technique_data = analytics_data[analytics_data['prompt_technique'].notna()]
                technique_counts = technique_data['prompt_technique'].value_counts().reset_index()
                technique_counts.columns = ['Technique', 'Count']
                
                fig = px.bar(technique_counts, x='Technique', y='Count', 
                            color='Count', color_continuous_scale='Viridis',
                            title="Prompt Techniques by Usage Frequency")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Model usage frequency chart
                st.subheader("Model Usage Frequency")
                model_counts = analytics_data['model'].value_counts().reset_index()
                model_counts.columns = ['Model', 'Count']
                
                fig = px.bar(model_counts, x='Model', y='Count', 
                            color='Count', color_continuous_scale='Plasma',
                            title="Models Used by Frequency")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Token usage by prompt technique
                st.subheader("Token Usage by Prompt Technique")
                token_by_technique = technique_data.groupby('prompt_technique')['token_count'].mean().reset_index()
                token_by_technique.columns = ['Technique', 'Avg Token Count']
                token_by_technique = token_by_technique.sort_values('Avg Token Count', ascending=False)
                
                fig = px.bar(token_by_technique, x='Technique', y='Avg Token Count',
                            color='Avg Token Count', color_continuous_scale='Viridis',
                            title="Average Token Usage by Prompt Technique")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Token usage by model
                st.subheader("Token Usage by Model")
                token_by_model = analytics_data.groupby('model')['token_count'].mean().reset_index()
                token_by_model.columns = ['Model', 'Avg Token Count']
                token_by_model = token_by_model.sort_values('Avg Token Count', ascending=False)
                
                fig = px.bar(token_by_model, x='Model', y='Avg Token Count',
                            color='Avg Token Count', color_continuous_scale='Plasma',
                            title="Average Token Usage by Model")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Scenario analysis (full width)
            st.subheader("Scenario Analysis")
            
            scenario_data = analytics_data.groupby(['scenario', 'prompt_technique'])['token_count'].mean().reset_index()
            scenario_data.columns = ['Scenario', 'Technique', 'Avg Token Count']
            
            fig = px.bar(scenario_data, x='Scenario', y='Avg Token Count', color='Technique', barmode='group',
                    title="Token Usage by Scenario and Prompt Technique")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Usage trends over time (full width)
            st.subheader("Usage Trends Over Time")
            
            # Prepare daily aggregated data
            analytics_data['date'] = analytics_data['timestamp'].dt.date
            
            # Trend by prompt technique
            daily_technique_usage = analytics_data[analytics_data['prompt_technique'].notna()].groupby(['date', 'prompt_technique']).size().reset_index(name='count')
            
            fig = px.line(daily_technique_usage, x='date', y='count', color='prompt_technique',
                        title="Daily Prompt Technique Usage Trends")
            fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Number of Queries")
            st.plotly_chart(fig, use_container_width=True)
            
            # Trend by model
            daily_model_usage = analytics_data.groupby(['date', 'model']).size().reset_index(name='count')
            
            fig = px.line(daily_model_usage, x='date', y='count', color='model',
                        title="Daily Model Usage Trends")
            fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Number of Queries")
            st.plotly_chart(fig, use_container_width=True)
            
            # Advanced Analytics section
            st.header("Advanced Analytics")
            
            # Create two tabs for different analysis views
            advanced_tab1, advanced_tab2 = st.tabs(["Prompt Engineering Analysis", "Model Performance Analysis"])
            
            with advanced_tab1:
                # Technique effectiveness by scenario
                st.subheader("Technique Effectiveness by Scenario")
                
                # For this analysis, we'll use token count as a proxy for response quality/complexity
                scenario_technique = analytics_data.groupby(['scenario', 'prompt_technique'])['token_count'].agg(['mean', 'count']).reset_index()
                scenario_technique.columns = ['Scenario', 'Technique', 'Avg Token Count', 'Count']
                
                # Only include scenarios with sufficient data
                popular_scenarios = analytics_data['scenario'].value_counts().nlargest(5).index
                scenario_technique = scenario_technique[scenario_technique['Scenario'].isin(popular_scenarios)]
                
                fig = px.scatter(scenario_technique, x='Scenario', y='Avg Token Count', 
                            size='Count', color='Technique',
                            title="Prompt Technique Effectiveness by Scenario")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Compare Zero Shot vs Few Shot effectiveness
                if 'Zero Shot' in analytics_data['prompt_technique'].values and 'Few Shot' in analytics_data['prompt_technique'].values:
                    st.subheader("Zero Shot vs Few Shot Comparison")
                    
                    zero_few_data = analytics_data[analytics_data['prompt_technique'].isin(['Zero Shot', 'Few Shot'])]
                    zero_few_summary = zero_few_data.groupby(['scenario', 'prompt_technique'])['token_count'].mean().reset_index()
                    zero_few_summary = zero_few_summary.pivot(index='scenario', columns='prompt_technique', values='token_count').reset_index()
                    zero_few_summary['token_difference'] = zero_few_summary['Few Shot'] - zero_few_summary['Zero Shot']
                    zero_few_summary['percentage_difference'] = (zero_few_summary['token_difference'] / zero_few_summary['Zero Shot'] * 100).round(1)
                    
                    # Create a comparison visualization
                    fig = px.bar(zero_few_summary, x='scenario', y='percentage_difference',
                            title="Few Shot vs Zero Shot: Token Count Percentage Difference",
                            color='percentage_difference', color_continuous_scale='RdBu',
                            labels={'percentage_difference': '% Difference (Few Shot - Zero Shot)'})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show the comparison data as a table
                    st.dataframe(zero_few_summary, use_container_width=True)
                
                # Temperature impact on prompt techniques
                st.subheader("Temperature Impact on Prompt Techniques")
                
                technique_temp = analytics_data.groupby(['prompt_technique', 'temperature'])['token_count'].mean().reset_index()
                technique_temp.columns = ['Technique', 'Temperature', 'Avg Token Count']
                
                fig = px.line(technique_temp, x='Temperature', y='Avg Token Count', color='Technique',
                        markers=True, title="How Temperature Affects Different Prompt Techniques")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with advanced_tab2:
                # Model comparison across scenarios
                st.subheader("Model Performance Across Scenarios")
                
                model_scenario = analytics_data.groupby(['model', 'scenario'])['token_count'].mean().reset_index()
                model_scenario.columns = ['Model', 'Scenario', 'Avg Token Count']
                
                # Only include popular models
                popular_models = analytics_data['model'].value_counts().nlargest(5).index
                model_scenario = model_scenario[model_scenario['Model'].isin(popular_models)]
                
                fig = px.bar(model_scenario, x='Scenario', y='Avg Token Count', color='Model', barmode='group',
                        title="Model Performance by Scenario")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Model effectiveness with different prompt techniques
                st.subheader("Model Performance with Different Prompt Techniques")
                
                model_technique = analytics_data.groupby(['model', 'prompt_technique'])['token_count'].mean().reset_index()
                model_technique.columns = ['Model', 'Technique', 'Avg Token Count']
                model_technique = model_technique[model_technique['Model'].isin(popular_models)]
                
                fig = px.bar(model_technique, x='Model', y='Avg Token Count', color='Technique', barmode='group',
                        title="Model Performance by Prompt Technique")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Temperature vs. Token Count Analysis by Model
                st.subheader("Temperature vs. Token Usage by Model")
                
                # Create a scatter plot
                fig = px.scatter(analytics_data, x='temperature', y='token_count', color='model',
                            size='token_count', hover_data=['prompt_technique', 'scenario'],
                            title="Effect of Temperature on Token Usage by Model")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Max Token Limit Analysis
                st.subheader("Max Token Limit Analysis")
                
                token_limit_data = analytics_data.groupby(['model', 'max_tokens'])['token_count'].agg(['mean', 'count']).reset_index()
                token_limit_data.columns = ['Model', 'Max Token Limit', 'Average Tokens Used', 'Count']
                token_limit_data = token_limit_data[token_limit_data['Model'].isin(popular_models)]
                
                fig = px.scatter(token_limit_data, x='Max Token Limit', y='Average Tokens Used',
                            size='Count', color='Model',
                            title="Relationship Between Max Token Limit and Actual Token Usage by Model")
                fig.update_layout(height=500)
                fig.add_trace(
                    go.Scatter(x=[0, max(token_limit_data['Max Token Limit'])], 
                            y=[0, max(token_limit_data['Max Token Limit'])], 
                            mode='lines', name='1:1 Line',
                            line=dict(color='red', dash='dash'))
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Token efficiency table
                token_limit_data['Efficiency'] = (token_limit_data['Average Tokens Used'] / token_limit_data['Max Token Limit'] * 100).round(1)
                
                st.subheader("Token Utilization Efficiency by Model")
                st.dataframe(token_limit_data.sort_values(['Model', 'Max Token Limit']), use_container_width=True)
            
            # Prompt Technique and Model Combined Analysis
            st.header("Combined Prompt-Model Analysis")
            
            # Create a heatmap of model + prompt technique effectiveness
            model_prompt_effectiveness = analytics_data.groupby(['model', 'prompt_technique'])['token_count'].mean().reset_index()
            model_prompt_pivot = model_prompt_effectiveness.pivot(index='model', columns='prompt_technique', values='token_count')
            
            fig = px.imshow(model_prompt_pivot, 
                        labels=dict(x="Prompt Technique", y="Model", color="Avg Token Count"),
                        title="Model-Prompt Technique Effectiveness Heatmap",
                        color_continuous_scale='Viridis')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Export data section
            st.header("Export Analytics Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = analytics_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Full Analytics Data (CSV)",
                    data=csv,
                    file_name=f"prompt_model_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
            
            with col2:
                # Create summary data
                summary_data = {
                    "total_queries": len(analytics_data),
                    "unique_users": analytics_data['username'].nunique(),
                    "unique_models": analytics_data['model'].nunique(),
                    "unique_techniques": analytics_data['prompt_technique'].nunique(),
                    "most_used_model": analytics_data['model'].value_counts().index[0],
                    "most_used_technique": analytics_data['prompt_technique'].value_counts().index[0],
                    "avg_tokens_per_response": int(analytics_data['token_count'].mean()),
                    "most_common_temperature": float(analytics_data['temperature'].mode()[0]),
                    "data_collection_started": analytics_data['timestamp'].min().strftime("%Y-%m-%d"),
                    "data_collection_latest": analytics_data['timestamp'].max().strftime("%Y-%m-%d"),
                }
                
                json_summary = json.dumps(summary_data, indent=2).encode('utf-8')
                st.download_button(
                    "📥 Download Summary Report (JSON)",
                    data=json_summary,
                    file_name=f"prompt_model_analytics_summary_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                )
    #tab4  
    with tab4:
        st.header("Model Voting & Results Dashboard")
        
        # Define columns for layout
        left_col, right_col = st.columns([2, 3])
        
        # Define the evaluation folder path
        EVAL_FOLDER = "Exercise/evaluations"
        if not os.path.exists(EVAL_FOLDER):
            os.makedirs(EVAL_FOLDER)
        
        # Define the evaluation file path
        EVAL_FILE = f"{EVAL_FOLDER}/model_votes.csv"
        
        # Left column - Model Voting Form
        with left_col:
            st.subheader("Vote for the Best Models")
            
            # Style the form with custom CSS
            st.markdown("""
            <style>
            .voting-card {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .winner-badge {
                display: inline-block;
                padding: 5px 10px;
                background-color: #ffd700;
                color: #333;
                border-radius: 15px;
                font-weight: bold;
                margin-right: 10px;
            }
            .criteria-card {
                background-color: #e9ecef;
                border-left: 4px solid #1f77b4;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 0 4px 4px 0;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Display voting criteria
            st.markdown("""
            <div class="voting-card">
                <h3>Voting Guidelines</h3>
                <p>Consider the following criteria when voting for the best model:</p>
            </div>
            """, unsafe_allow_html=True)

            # Display each criterion in a separate card using st.markdown instead of raw HTML
            st.markdown('<div class="criteria-card"><strong>Response Quality:</strong> Which model provided the most accurate and helpful answers?</div>', unsafe_allow_html=True)
            st.markdown('<div class="criteria-card"><strong>Reasoning Ability:</strong> Which model showed the best logical reasoning and step-by-step thinking?</div>', unsafe_allow_html=True)
            st.markdown('<div class="criteria-card"><strong>Clarity:</strong> Which model communicated most clearly and was easiest to understand?</div>', unsafe_allow_html=True)
            st.markdown('<div class="criteria-card"><strong>Completeness:</strong> Which model provided the most comprehensive responses?</div>', unsafe_allow_html=True)
                        
            with st.form("model_voting_form"):
                # Get all available models for voting
                def get_all_available_models():
                    # Get models from predefined list
                    predefined_models = set(llm_models)
                    
                    # Check for model responses in session state
                    models_from_responses = set()
                    if hasattr(st.session_state, 'model_responses'):
                        models_from_responses.update(st.session_state.model_responses.keys())
                        
                    if hasattr(st.session_state, 'model_responses_tab2'):
                        models_from_responses.update(st.session_state.model_responses_tab2.keys())
                    
                    # Combine all models
                    all_models = predefined_models.union(models_from_responses)
                    
                    # Return as sorted list
                    return sorted(list(all_models))
                
                all_models = get_all_available_models()
                
                # Scenario selection
                scenarios = ["📱 Mobile Security Policy", "📂 Data Ownership Scenario", "🔒 Security Breach Protocol"]
                selected_scenario = st.selectbox("Select scenario:", scenarios)
                
                # Model selection
                st.markdown("### Choose Your Top Model")
                top_model = st.selectbox(
                    "Which model performed the best for the selected scenario?",
                    options=[""] + all_models,
                    key="top_model"
                )
                
                # Why this model is the best
                if top_model:
                    top_model_reason = st.text_area(
                        "Why is this model the best?",
                        placeholder="Explain what made this model stand out...",
                        key="top_model_reason"
                    )
                
                # General feedback (optional)
                st.markdown("### Additional Feedback (Optional)")
                general_feedback = st.text_area(
                    "Share any additional observations about the models' performance",
                    placeholder="Other thoughts or comparisons between models...",
                    key="general_feedback"
                )
                
                # Submit button with styling
                submitted = st.form_submit_button(
                    "Cast Your Vote",
                    use_container_width=True,
                    type="primary"
                )
                
                # Process form submission
                if submitted:
                    if not top_model:
                        st.error("Please select a model to vote for.")
                    else:
                        # Add timestamp
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Prepare data for the DataFrame
                        new_vote = {
                            'timestamp': timestamp,
                            'username': st.session_state.username,
                            'scenario': selected_scenario,
                            'voted_model': top_model,
                            'reason': top_model_reason if 'top_model_reason' in locals() else "",
                            'feedback': general_feedback
                        }
                        
                        # Load existing data or create new DataFrame
                        if os.path.exists(EVAL_FILE):
                            try:
                                votes_df = pd.read_csv(EVAL_FILE)
                                # Append new vote
                                votes_df = pd.concat([votes_df, pd.DataFrame([new_vote])], ignore_index=True)
                            except Exception as e:
                                st.error(f"Error reading existing votes file: {str(e)}")
                                # Create new DataFrame with just this vote
                                votes_df = pd.DataFrame([new_vote])
                        else:
                            # Create new DataFrame with just this vote
                            votes_df = pd.DataFrame([new_vote])
                        
                        # Save to CSV
                        try:
                            votes_df.to_csv(EVAL_FILE, index=False)
                            st.success("✅ Your vote has been recorded!")
                            
                            # Refresh the page to update visualizations
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error saving vote: {str(e)}")
        
        # Right column - Results Visualization
        with right_col:
            st.subheader("Voting Results & Analytics")
            
            # Load voting data
            if os.path.exists(EVAL_FILE):
                try:
                    votes_df = pd.read_csv(EVAL_FILE)
                    
                    # Add visualization options
                    viz_tabs = st.tabs(["Overall Results", "Scenario Analysis", "Time Trends", "Feedback Analysis"])
                    
                    # Overall Results tab
                    with viz_tabs[0]:
                        st.markdown("### Overall Voting Results")
                        
                        # Count votes for each model
                        vote_counts = votes_df['voted_model'].value_counts().reset_index()
                        vote_counts.columns = ['Model', 'Votes']
                        vote_counts = vote_counts.sort_values('Votes', ascending=False)
                        
                        # Display the winner and top models
                        if not vote_counts.empty:
                            # Create metrics for top 3 models
                            col1, col2, col3 = st.columns(3)
                            
                            # Winner (1st place)
                            with col1:
                                winner_model = vote_counts.iloc[0]['Model']
                                winner_votes = vote_counts.iloc[0]['Votes']
                                total_votes = vote_counts['Votes'].sum()
                                winner_percentage = (winner_votes / total_votes * 100) if total_votes > 0 else 0
                                
                                st.markdown(f"""
                                <div style="text-align: center">
                                    <span class="winner-badge">WINNER</span>
                                </div>
                                """, unsafe_allow_html=True)
                                st.metric(
                                    winner_model, 
                                    f"{winner_votes} votes", 
                                    f"{winner_percentage:.1f}% of total"
                                )
                                
                            # 2nd place
                            with col2:
                                if len(vote_counts) >= 2:
                                    second_model = vote_counts.iloc[1]['Model']
                                    second_votes = vote_counts.iloc[1]['Votes']
                                    second_percentage = (second_votes / total_votes * 100) if total_votes > 0 else 0
                                    
                                    st.markdown(f"""
                                    <div style="text-align: center">
                                        <span style="color: #666; font-weight: bold;">SECOND PLACE</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.metric(
                                        second_model, 
                                        f"{second_votes} votes", 
                                        f"{second_percentage:.1f}% of total"
                                    )
                                    
                            # 3rd place
                            with col3:
                                if len(vote_counts) >= 3:
                                    third_model = vote_counts.iloc[2]['Model']
                                    third_votes = vote_counts.iloc[2]['Votes']
                                    third_percentage = (third_votes / total_votes * 100) if total_votes > 0 else 0
                                    
                                    st.markdown(f"""
                                    <div style="text-align: center">
                                        <span style="color: #666; font-weight: bold;">THIRD PLACE</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.metric(
                                        third_model, 
                                        f"{third_votes} votes", 
                                        f"{third_percentage:.1f}% of total"
                                    )
                            
                            # Display vote distribution chart
                            st.markdown("### Vote Distribution")
                            st.bar_chart(vote_counts.set_index('Model'))
                            
                            # Display summary statistics
                            st.markdown("### Voting Summary")
                            st.markdown(f"**Total Votes Cast:** {total_votes}")
                            st.markdown(f"**Number of Models Voted For:** {len(vote_counts)}")
                            st.markdown(f"**Most Popular Model:** {winner_model} with {winner_votes} votes ({winner_percentage:.1f}%)")
                            
                            # Display winner feedback
                            st.markdown("### Why People Voted for the Winner")
                            winner_feedback = votes_df[votes_df['voted_model'] == winner_model]['reason'].dropna()
                            
                            if not winner_feedback.empty:
                                # Display random sample of feedback (up to 3)
                                feedback_sample = winner_feedback.sample(min(3, len(winner_feedback)))
                                for i, feedback in enumerate(feedback_sample):
                                    if feedback and str(feedback) != 'nan':
                                        st.markdown(f"""
                                        <div style="border-left: 4px solid #ffd700; padding-left: 10px; margin-bottom: 10px;">
                                            "{feedback}"
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.info(f"No specific feedback available for {winner_model}")
                        else:
                            st.info("No votes have been cast yet. Be the first to vote!")
                    
                    # Scenario Analysis tab
                    with viz_tabs[1]:
                        st.markdown("### Results by Scenario")
                        
                        # Scenario filter
                        scenario_options = ["All"] + list(votes_df['scenario'].unique())
                        selected_scenario_filter = st.selectbox(
                            "Filter by scenario:",
                            scenario_options,
                            key="scenario_filter"
                        )
                        
                        # Filter data by scenario if needed
                        scenario_data = votes_df
                        if selected_scenario_filter != "All":
                            scenario_data = votes_df[votes_df['scenario'] == selected_scenario_filter]
                        
                        if not scenario_data.empty:
                            # Count votes for each model in this scenario
                            scenario_votes = scenario_data['voted_model'].value_counts().reset_index()
                            scenario_votes.columns = ['Model', 'Votes']
                            scenario_votes = scenario_votes.sort_values('Votes', ascending=False)
                            
                            # Display the top model for this scenario
                            st.markdown(f"### Top Model for {selected_scenario_filter if selected_scenario_filter != 'All' else 'All Scenarios'}")
                            
                            if not scenario_votes.empty:
                                winner_model = scenario_votes.iloc[0]['Model']
                                winner_votes = scenario_votes.iloc[0]['Votes']
                                total_votes = scenario_votes['Votes'].sum()
                                winner_percentage = (winner_votes / total_votes * 100) if total_votes > 0 else 0
                                
                                st.markdown(f"""
                                <div style="text-align: center; margin-bottom: 20px;">
                                    <span class="winner-badge">WINNER</span>
                                    <h2>{winner_model}</h2>
                                    <p>{winner_votes} votes ({winner_percentage:.1f}% of total)</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display vote distribution chart for this scenario
                                st.markdown("### Vote Distribution")
                                st.bar_chart(scenario_votes.set_index('Model'))
                                
                                # Display feedback for winner in this scenario
                                st.markdown("### Why People Voted for the Winner")
                                scenario_winner_feedback = scenario_data[scenario_data['voted_model'] == winner_model]['reason'].dropna()
                                
                                if not scenario_winner_feedback.empty:
                                    # Display random sample of feedback (up to 3)
                                    s_feedback_sample = scenario_winner_feedback.sample(min(3, len(scenario_winner_feedback)))
                                    for feedback in s_feedback_sample:
                                        if feedback and str(feedback) != 'nan':
                                            st.markdown(f"""
                                            <div style="border-left: 4px solid #ffd700; padding-left: 10px; margin-bottom: 10px;">
                                                "{feedback}"
                                            </div>
                                            """, unsafe_allow_html=True)
                                else:
                                    st.info(f"No specific feedback available for {winner_model} in this scenario")
                        else:
                            st.info(f"No votes available for {selected_scenario_filter}")
                    
                    # Time Trends tab
                    with viz_tabs[2]:
                        st.markdown("### Voting Trends Over Time")
                        
                        # Convert timestamp to datetime
                        votes_df['timestamp'] = pd.to_datetime(votes_df['timestamp'])
                        
                        # Add time filter
                        time_filter_options = [
                            "All Time",
                            "Last Week",
                            "Last Month",
                            "Last 3 Months"
                        ]
                        selected_time_filter = st.selectbox(
                            "Filter by time period:",
                            time_filter_options,
                            key="time_filter"
                        )
                        
                        # Apply time filter
                        filtered_df = votes_df.copy()
                        current_time = datetime.now()
                        
                        if selected_time_filter == "Last Week":
                            one_week_ago = current_time - timedelta(days=7)
                            filtered_df = votes_df[votes_df['timestamp'] >= one_week_ago]
                        elif selected_time_filter == "Last Month":
                            one_month_ago = current_time - timedelta(days=30)
                            filtered_df = votes_df[votes_df['timestamp'] >= one_month_ago]
                        elif selected_time_filter == "Last 3 Months":
                            three_months_ago = current_time - timedelta(days=90)
                            filtered_df = votes_df[votes_df['timestamp'] >= three_months_ago]
                        
                        # Create time-based visualizations
                        if not filtered_df.empty:
                            # Add a day column for grouping
                            filtered_df['day'] = filtered_df['timestamp'].dt.date
                            
                            # Group by day and count votes
                            daily_counts = filtered_df.groupby('day').size().reset_index(name='count')
                            
                            # Display daily vote counts
                            st.markdown("### Daily Voting Activity")
                            st.line_chart(daily_counts.set_index('day'))
                            
                            # Display top models over time
                            st.markdown("### Top Models Over Time")
                            
                            # Get top 3 models across the entire filtered period
                            top_models = filtered_df['voted_model'].value_counts().nlargest(3).index.tolist()
                            
                            if top_models:
                                # Add filter for specific models
                                model_filter_options = ["All Top Models"] + top_models
                                selected_model_filter = st.radio(
                                    "Filter by model:",
                                    model_filter_options,
                                    horizontal=True
                                )
                                
                                # Group by day and model, then count votes
                                model_time_data = filtered_df.groupby(['day', 'voted_model']).size().reset_index(name='votes')
                                
                                # Filter for selected models
                                if selected_model_filter != "All Top Models":
                                    model_time_data = model_time_data[model_time_data['voted_model'] == selected_model_filter]
                                else:
                                    model_time_data = model_time_data[model_time_data['voted_model'].isin(top_models)]
                                
                                # Create pivot table for visualization
                                model_pivot = model_time_data.pivot(index='day', columns='voted_model', values='votes').fillna(0)
                                
                                # Display trend chart
                                st.line_chart(model_pivot)
                            else:
                                st.info("Not enough data to display model trends")
                        else:
                            st.info(f"No data available for the selected time period: {selected_time_filter}")
                    
                    # Feedback Analysis tab
                    with viz_tabs[3]:
                        st.markdown("### User Feedback Analysis")
                        
                        # Model selector for feedback
                        feedback_models = ["All Models"] + list(votes_df['voted_model'].unique())
                        selected_feedback_model = st.selectbox(
                            "View feedback for:",
                            feedback_models,
                            key="feedback_model_selector"
                        )
                        
                        # Filter feedback by model
                        if selected_feedback_model != "All Models":
                            feedback_data = votes_df[votes_df['voted_model'] == selected_feedback_model]
                        else:
                            feedback_data = votes_df
                        
                        # Display feedback
                        if not feedback_data.empty:
                            # Get model-specific reasons
                            reasons = feedback_data['reason'].dropna()
                            
                            if not reasons.empty:
                                st.markdown(f"### Why Users Voted for {selected_feedback_model if selected_feedback_model != 'All Models' else 'Their Preferred Models'}")
                                
                                # Display all reasons with username and timestamp
                                for _, row in feedback_data.iterrows():
                                    if row['reason'] and str(row['reason']) != 'nan':
                                        st.markdown(f"""
                                        <div style="border-left: 4px solid #1f77b4; padding-left: 10px; margin-bottom: 15px;">
                                            <small>{row['timestamp']} - <strong>{row['username']}</strong> voted for <strong>{row['voted_model']}</strong></small><br>
                                            "{row['reason']}"
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.info(f"No specific reasons provided for {selected_feedback_model if selected_feedback_model != 'All Models' else 'any models'}")
                            
                            # General feedback
                            general_feedback = feedback_data['feedback'].dropna()
                            
                            if not general_feedback.empty:
                                st.markdown("### General Feedback")
                                
                                # Display all general feedback
                                for _, row in feedback_data.iterrows():
                                    if row['feedback'] and str(row['feedback']) != 'nan':
                                        st.markdown(f"""
                                        <div style="border-left: 4px solid #4c9be8; padding-left: 10px; margin-bottom: 15px;">
                                            <small>{row['timestamp']} - <strong>{row['username']}</strong></small><br>
                                            "{row['feedback']}"
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.info("No general feedback provided")
                            
                            # Download options
                            st.markdown("### Download Feedback Data")
                            
                            # Prepare filtered data for download
                            csv = feedback_data.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Feedback as CSV",
                                data=csv,
                                file_name=f"model_feedback_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                            )
                        else:
                            st.info("No feedback available for the selected model")
                except Exception as e:
                    st.error(f"Error loading voting data: {str(e)}")
            else:
                st.info("No votes have been cast yet. Be the first to vote using the form!")