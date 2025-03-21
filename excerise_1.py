import pandas as pd
import streamlit as st
from groq import Groq
import os
from database import UserDatabase
import re
import time
import plotly.express as px
import plotly.graph_objects as go
import json
import re
from datetime import datetime, timedelta    
import  io
import PyPDF2
from dotenv import load_dotenv
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
github_key = os.getenv("GITHUB_API_KEY")
def exercise1():
    # Initialize the user database
    user_db = UserDatabase()
    
    # Initialize session state variables
    if "chatHist" not in st.session_state:
        st.session_state.chatHist = {}
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
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
    if "is_comparing" not in st.session_state:
        st.session_state.is_comparing = False
    if "comparison_complete" not in st.session_state:
        st.session_state.comparison_complete = False
    if "expanded_scenario" not in st.session_state:
        st.session_state.expanded_scenario = None
    
    st.write("compare the lllm models")
    # Available LLM models
    llm_models = [
        "llama3-70b-8192", "gemma2-9b-it", "llama-3.2-90b-vision-preview", "qwen-2.5-32b",
        "deepseek-r1-distill-qwen-32b", "allam-2-7b","mistral-saba-24b"
    ]
    
    # Initialize model chat histories
    for model in llm_models:
        if model not in st.session_state.chatHist:
            st.session_state.chatHist[model] = [{"role": "system", "content": "You are a helpful assistant."}]
    
    # Create two tabs
    tab1, tab2, tab3 = st.tabs(["📋 Configuration", "🏆 Comparison Arena", "🔍 Data Analysis and Visualization"])
    
    with tab1:
        # Left sidebar for settings, right for scenario selection
        col_settings, col_scenarios = st.columns([2, 3], gap="medium")
        
        with col_settings:
            # Model Selection Panel
            st.subheader("📊 Model Selection")
            
            # Model multiselect
            selected_models = st.multiselect(
                "Select models to compare:",
                llm_models,
                default=st.session_state.selected_models[:1] if st.session_state.selected_models else llm_models[:1],
                help="Choose up to 4 models to compare side by side"
            )
            
            # Update session state with selected models
            st.session_state.selected_models = selected_models[:4]  # Limit to 4 models
            
            # Show warning if more than 4 models selected
            if len(selected_models) > 4:
                st.warning("Maximum 4 models can be compared at once. Only the first 4 will be used.")
            
            # Parameters Panel
            st.subheader("⚙️ Model Parameters")
            
            # Temperature slider
            temp_val = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Controls randomness: lower values are more deterministic"
            )
            st.session_state.temperature = temp_val
            
            # Display temperature description based on value
            if temp_val < 0.5:
                st.write("Focused: More consistent responses")
            elif temp_val < 1.2:
                st.write("Balanced: Good mix of creativity and consistency")
            else:
                st.write("Creative: More diverse responses")
            
            # Max tokens slider
            st.session_state.max_tokens = st.slider(
                "Max Tokens",
                min_value=200,
                max_value=4000,
                value=st.session_state.max_tokens,
                step=100,
                help="Maximum length of model response"
            )
            
            # Document Upload Panel
            st.subheader("📄 Document Upload (Optional)")
            
            uploaded_file = st.file_uploader(
                "Upload a document for context:",
                type=["txt", "pdf", "docx"],
                help="Models will use this document as context for answering"
            )

            if uploaded_file:
                # Check if the file is a PDF
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
                        st.success(f"✅ Successfully uploaded and processed PDF: {uploaded_file.name}")
                        
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
                        # Set a basic placeholder to avoid None errors later
                        st.session_state.uploaded_file_content = f"Error extracting text from {uploaded_file.name}: {str(e)}"
                
                # For non-PDF files
                else:
                    try:
                        file_content = uploaded_file.read().decode("utf-8", errors="ignore")
                        
                        # Apply a character limit
                        MAX_CHARS = 4000
                        if len(file_content) > MAX_CHARS:
                            st.warning(f"File content is too large ({len(file_content)} characters). Truncating to {MAX_CHARS} characters.")
                            file_content = file_content[:MAX_CHARS] + "\n[Content truncated due to size limits]"
                        
                        st.session_state.uploaded_file_content = file_content
                        st.success(f"✅ Successfully uploaded: {uploaded_file.name}")
                        
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
                        st.session_state.uploaded_file_content = f"Error reading {uploaded_file.name}: {str(e)}"
                
                # Create uploads directory if it doesn't exist (unchanged)
                if not os.path.exists("Exercise/uploads"):
                    os.makedirs("Exercise/uploads")
                
                # Save the file (reset the file pointer position before saving)
                uploaded_file.seek(0)
                with open(os.path.join("Exercise/uploads", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
        
        with col_scenarios:
            st.subheader("🎯 Select Scenario")
            st.write("Choose one scenario to evaluate model performance:")

            # Scenarios dictionary
            scenarios = {
                "📱 Mobile Security Policy": {
                    "question": '''A university IT administrator discovers that a faculty member has installed a custom operating system on their university-provided smartphone. According to best practices aligned with WPI's Mobile Device Management Policy, what should happen next?''',
                    "options": [
                        "The device should be remotely wiped immediately",
                        "The device should be confiscated and recycled",
                        "The OS must be restored to manufacturer specifications",
                        "The faculty member should be required to purchase the device"
                    ]
                },
                "📂 Data Ownership Scenario": {
                    "question": '''A research professor who used their personal smartphone to access university email for 5 years is leaving for a position at another institution. The phone contains important research communications. Based on standard MDM policies similar to WPI's, who has rights to the email data on the device?''',
                    "options": [
                        "The professor retains full ownership of all data",
                        "The emails belong to the university but the device belongs to the professor",
                        "The university has rights to wipe university-related data only",
                        "Ownership is determined by the department chair"
                    ]
                },
                "🔒 Security Breach Protocol": {
                    "question": '''A university employee reports their smartphone containing university email was stolen while traveling. According to principles in policies like WPI's, what is the correct sequence of actions?''',
                    "options": [
                        "Report to local police, then IT security, wait for instructions",
                        "Contact IT security immediately, remote wipe will be initiated",
                        "Purchase a replacement device, restore from backup, then report",
                        "Change passwords for all accounts, then report to department head"
                    ]
                }
            }

            # Display predefined scenarios as radio buttons
            scenario_options = list(scenarios.keys())
            selected_scenario_name = st.radio(
                "Select a scenario:", 
                scenario_options,
                index=0 if st.session_state.selected_scenario else None
            )

            if selected_scenario_name:
                # Display the selected scenario details
                st.write(f"**Question:** {scenarios[selected_scenario_name]['question']}")
                
                # Display options as bullet points
                st.write("**Options:**")
                for option in scenarios[selected_scenario_name]["options"]:
                    st.write(f"- {option}")
                
                # Store the selected scenario in session state
                st.session_state.selected_scenario = scenarios[selected_scenario_name]["question"]
            else:
                st.session_state.selected_scenario = None

            # Custom scenario input
            custom_scenario = st.text_area(
                "Or create your own scenario/question:",
                placeholder="Enter your custom scenario or question here...",
                help="Write a specific question to test the models"
            )

            # If custom scenario is provided, use it
            if custom_scenario.strip():
                st.session_state.selected_scenario = custom_scenario
                st.success("✅ Using your custom scenario")
    
    with tab2:
        if not st.session_state.selected_models:
            st.info("👈 Please select models and a scenario in the Configuration tab first")
        elif not st.session_state.selected_scenario:
            st.info("👈 Please select a scenario in the Configuration tab first")
        else:
            # Display the selected scenario at the top
            st.subheader("Selected Scenario")
            st.write(st.session_state.selected_scenario)
            
            # Create the comparison button
            if not st.session_state.comparison_complete and not st.session_state.is_comparing:
                if st.button("🚀 Start Comparison", key="compare_button", help="Compare selected models on the chosen scenario"):
                    st.session_state.is_comparing = True
                    st.session_state.model_responses = {}  # Reset responses
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
                        if st.session_state.is_comparing and model not in st.session_state.model_responses:
                            #
                            try:
                                # Determine which API to use based on the model
                                if model == "gemini-2.0-flash":
                                    # Initialize Google Genai client
                                    google_client = genai.Client(api_key=gemini_key)
                                    
                                    # Prepare content
                                    prompt_content = ""
                                    
                                    # Add uploaded document if available
                                    if st.session_state.uploaded_file_content:
                                        prompt_content += f"Please use this information as context: {st.session_state.uploaded_file_content}\n\n"
                                    
                                    # Add the scenario/question
                                    prompt_content += st.session_state.selected_scenario
                                    
                                    # Make API call to Google's Gemini
                                    generation_config = {
                                        "max_output_tokens": st.session_state.max_tokens,
                                        "temperature": st.session_state.temperature
                                    }
                                    
                                    response = google_client.generate_content(
                                        model="gemini-2.0-flash",
                                        contents=prompt_content,
                                        generation_config=generation_config
                                    )
                                    
                                    # Store the response
                                    st.session_state.model_responses[model] = response.text
                                else:
                                    # Use Groq for other models
                                    client = Groq(api_key=groq_key)
                                    
                                    # Prepare messages
                                    messages = [{"role": "system", "content": "You are a helpful assistant."}]
                                    
                                    # Add uploaded document if available
                                    if st.session_state.uploaded_file_content:
                                        messages.append({"role": "user", "content": f"Please use this information as context: {st.session_state.uploaded_file_content}"})
                                    
                                    # Add the scenario/question
                                    messages.append({"role": "user", "content": st.session_state.selected_scenario})
                                    
                                    # Make API call
                                    response = client.chat.completions.create(
                                        model=model,
                                        messages=messages,
                                        max_tokens=st.session_state.max_tokens,
                                        temperature=st.session_state.temperature
                                    )
                                    
                                    # Store the response
                                    st.session_state.model_responses[model] = response.choices[0].message.content
                                
                                # Update session state
                                if len(st.session_state.model_responses) == len(st.session_state.selected_models):
                                    st.session_state.is_comparing = False
                                    st.session_state.comparison_complete = True
                                
                            except Exception as e:
                                st.session_state.model_responses[model] = f"Error: {str(e)}"
                                if len(st.session_state.model_responses) == len(st.session_state.selected_models):
                                    st.session_state.is_comparing = False
                                    st.session_state.comparison_complete = True
                                                    
                        # Display response if available
                        if model in st.session_state.model_responses:
                            response = st.session_state.model_responses[model]
                            
                            # Calculate metrics
                            word_count = len(response.split())
                            
                            
                            # Display metrics
                            metrics_cols = st.columns(1)
                            with metrics_cols[0]:
                                st.write("token ", word_count)
                        
                            
                            # Display the response
                            st.text_area("Response", response, height=350, key=f"response_{model}")
                
                # Save responses to database if all responses are collected and user is authenticated
                if st.session_state.comparison_complete and st.session_state.username != "guest" and len(st.session_state.model_responses) > 0:
                    saved_file = user_db.save_user_interaction(
                        st.session_state.username,
                        st.session_state.selected_scenario,
                        st.session_state.model_responses
                    )
                    
                    if saved_file and not hasattr(st.session_state, 'results_saved'):
                        st.sidebar.success(f"✅ Responses saved to database")
                        st.session_state.results_saved = True
                
                # Reset button after comparison is complete
                if st.session_state.comparison_complete:
                    if st.button("🔄 Start New Comparison", key="reset_comparison"):
                        st.session_state.is_comparing = False
                        st.session_state.comparison_complete = False
                        if hasattr(st.session_state, 'results_saved'):
                            delattr(st.session_state, 'results_saved')
                        st.rerun()
    # Data Analysis and Visualization Tab
    with tab3:
    # Create folder for storing analytics data
        ANALYTICS_FOLDER = "Exercise/ExerciseOne"
        if not os.path.exists(ANALYTICS_FOLDER):
            os.makedirs(ANALYTICS_FOLDER)
        
        # Path to the analytics data file
        ANALYTICS_FILE = f"{ANALYTICS_FOLDER}/model_usage_analytics.csv"
        
        st.header("Model Usage Analytics")
        
        # Function to save model usage data
        def save_model_usage_data():
            # Only process if there are model responses and a selected scenario
            if (not hasattr(st.session_state, 'model_responses') or 
                not st.session_state.model_responses or 
                not st.session_state.selected_scenario):
                return
                
            # Check if we've already saved this session's data
            if hasattr(st.session_state, 'analytics_saved') and st.session_state.analytics_saved:
                return
                
            # Prepare base data for each model response
            base_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'username': st.session_state.username if hasattr(st.session_state, 'username') else "anonymous",
                'prompt': st.session_state.selected_scenario,
                'temperature': st.session_state.temperature,
                'max_tokens': st.session_state.max_tokens
            }
            
            # Prepare data for all models
            rows = []
            
            for model, response in st.session_state.model_responses.items():
                # Create a copy of base data
                row_data = base_data.copy()
                
                # Add model-specific information
                row_data['model'] = model
                row_data['response'] = response
                
                # Estimate token count - this is approximate
                # A better approach would be to use a proper tokenizer if available
                token_count = len(response.split())
                row_data['token_count'] = token_count
                
                rows.append(row_data)
            
            # Create DataFrame from the rows
            new_data = pd.DataFrame(rows)
            
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
            
            # Mark that we've saved the data for this session
            st.session_state.analytics_saved = True
        
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
        
        # Save current model usage data if available
        save_model_usage_data()
        
        # Load all analytics data
        analytics_data = load_analytics_data()
        
        if analytics_data.empty:
            st.info("No analytics data available yet. Complete model comparisons to generate data.")
        else:
            # Brief introduction about data analytics
            with st.expander("📊 Why Data Analysis is Important", expanded=True):
                st.markdown("""
                ### The Power of LLM Analytics

                Tracking and analyzing language model performance is critical for:

                - **Model Selection**: Identify which models perform best for specific types of questions
                - **Resource Optimization**: Balance token usage with response quality
                - **Parameter Tuning**: Discover optimal temperature and token limit settings
                - **User Behavior**: Understand how different users interact with AI models
                - **Continuous Improvement**: Track performance metrics over time to guide future enhancements

                The visualizations below will help you extract actionable insights from your interactions with these models.
                """)
            
            # Create two columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Model usage frequency chart
                st.subheader("Model Usage Frequency")
                model_counts = analytics_data['model'].value_counts().reset_index()
                model_counts.columns = ['Model', 'Count']
                
                fig = px.bar(model_counts, x='Model', y='Count', 
                            color='Count', color_continuous_scale='Viridis',
                            title="Models Used by Frequency")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Temperature distribution
                st.subheader("Temperature Settings Distribution")
                temp_counts = analytics_data['temperature'].value_counts().reset_index()
                temp_counts.columns = ['Temperature', 'Count']
                temp_counts = temp_counts.sort_values('Temperature')
                
                fig = px.bar(temp_counts, x='Temperature', y='Count',
                            color='Temperature', color_continuous_scale='Thermal',
                            title="Frequency of Temperature Settings")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
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
                
                # User activity
                st.subheader("User Activity")
                user_counts = analytics_data['username'].value_counts().reset_index()
                user_counts.columns = ['User', 'Query Count']
                user_counts = user_counts.sort_values('Query Count', ascending=False).head(10)
                
                fig = px.bar(user_counts, x='User', y='Query Count',
                            color='Query Count', color_continuous_scale='Viridis',
                            title="Most Active Users (Top 10)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Usage trends over time (full width)
            st.subheader("Model Usage Trends Over Time")
            
            # Prepare daily aggregated data
            analytics_data['date'] = analytics_data['timestamp'].dt.date
            daily_usage = analytics_data.groupby(['date', 'model']).size().reset_index(name='count')
            
            # Create line chart
            fig = px.line(daily_usage, x='date', y='count', color='model',
                        title="Daily Model Usage Trends")
            fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Number of Queries")
            st.plotly_chart(fig, use_container_width=True)
            
            # Advanced Analytics section
            st.header("Advanced Analytics")
            
            # Create two tabs for different analysis views
            advanced_tab1, advanced_tab2 = st.tabs(["Model Performance", "Parameter Analysis"])
            
            with advanced_tab1:
                # Token usage distribution
                st.subheader("Token Usage Distribution")
                
                # Create token usage bins
                analytics_data['token_bin'] = pd.cut(analytics_data['token_count'], 
                                                bins=[0, 100, 200, 300, 500, 1000, 2000, 4000],
                                                labels=['0-100', '101-200', '201-300', '301-500', 
                                                        '501-1000', '1001-2000', '2001-4000'])
                
                token_dist = analytics_data.groupby(['model', 'token_bin']).size().reset_index(name='count')
                
                # Only include models with sufficient data
                popular_models = analytics_data['model'].value_counts().nlargest(5).index
                token_dist = token_dist[token_dist['model'].isin(popular_models)]
                
                fig = px.bar(token_dist, x='token_bin', y='count', color='model', barmode='group',
                            title="Token Usage Distribution by Model")
                fig.update_layout(height=500, xaxis_title="Token Count Range", yaxis_title="Number of Responses")
                st.plotly_chart(fig, use_container_width=True)
                
                # Model usage by prompt type (using keyword analysis)
                st.subheader("Model Usage by Topic Category")
                
                # Define keywords for different topics
                topic_keywords = {
                    "Security": ["security", "breach", "protection", "hack", "vulnerability", "policy", "compliance"],
                    "Data": ["data", "information", "database", "storage", "file", "record", "document"],
                    "Policy": ["policy", "procedure", "guideline", "regulation", "compliance", "rule", "standard"],
                    "Technical": ["technical", "system", "network", "device", "hardware", "software", "configuration"],
                    "Mobile": ["mobile", "phone", "device", "smartphone", "tablet", "cellular", "wireless"]
                }
                
                # Function to categorize prompts
                def categorize_prompt(prompt):
                    prompt_lower = prompt.lower()
                    categories = []
                    
                    for category, keywords in topic_keywords.items():
                        for keyword in keywords:
                            if keyword.lower() in prompt_lower:
                                categories.append(category)
                                break
                                
                    if not categories:
                        return ["Other"]
                    return categories
                
                # Apply categorization
                analytics_data['topics'] = analytics_data['prompt'].apply(categorize_prompt)
                
                # Explode the topics list to create one row per topic
                topic_data = analytics_data.explode('topics')
                
                # Count by model and topic
                topic_counts = topic_data.groupby(['model', 'topics']).size().reset_index(name='count')
                
                # Filter to most common models
                topic_counts = topic_counts[topic_counts['model'].isin(popular_models)]
                
                fig = px.bar(topic_counts, x='topics', y='count', color='model', barmode='group',
                            title="Model Usage by Topic Category")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with advanced_tab2:
                # Temperature vs. Token Count Analysis
                st.subheader("Temperature vs. Token Usage")
                
                # Create a scatter plot
                fig = px.scatter(analytics_data, x='temperature', y='token_count', color='model',
                                size='token_count', hover_data=['timestamp', 'username'],
                                title="Effect of Temperature on Token Usage")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Max Token Limit Analysis
                st.subheader("Max Token Limit Analysis")
                token_limit_data = analytics_data.groupby('max_tokens')['token_count'].agg(['mean', 'count']).reset_index()
                token_limit_data.columns = ['Max Token Limit', 'Average Tokens Used', 'Count']
                
                fig = px.scatter(token_limit_data, x='Max Token Limit', y='Average Tokens Used',
                                size='Count', color='Average Tokens Used',
                                color_continuous_scale='Viridis',
                                title="Relationship Between Max Token Limit and Actual Token Usage")
                fig.update_layout(height=500)
                fig.add_trace(
                    go.Scatter(x=[0, 4000], y=[0, 4000], mode='lines', name='1:1 Line',
                            line=dict(color='red', dash='dash'))
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Efficiency calculation
                token_limit_data['Efficiency'] = (token_limit_data['Average Tokens Used'] / token_limit_data['Max Token Limit'] * 100).round(1)
                
                # Format as a table
                st.subheader("Token Utilization Efficiency")
                st.dataframe(token_limit_data.sort_values('Max Token Limit'), use_container_width=True)
            
            # Export data section
            st.header("Export Analytics Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = analytics_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Full Analytics Data (CSV)",
                    data=csv,
                    file_name=f"model_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
            
            with col2:
                # Create summary data
                summary_data = {
                    "total_queries": len(analytics_data),
                    "unique_users": analytics_data['username'].nunique(),
                    "unique_models": analytics_data['model'].nunique(),
                    "most_used_model": analytics_data['model'].value_counts().index[0],
                    "avg_tokens_per_response": int(analytics_data['token_count'].mean()),
                    "most_common_temperature": float(analytics_data['temperature'].mode()[0]),
                    "data_collection_started": analytics_data['timestamp'].min().strftime("%Y-%m-%d"),
                    "data_collection_latest": analytics_data['timestamp'].max().strftime("%Y-%m-%d"),
                }
                
                json_summary = json.dumps(summary_data, indent=2).encode('utf-8')
                st.download_button(
                    "📥 Download Summary Report (JSON)",
                    data=json_summary,
                    file_name=f"model_analytics_summary_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                )