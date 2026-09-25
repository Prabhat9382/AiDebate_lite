import streamlit as st
import os
import time
from litellm import completion

# 1. Load Keys from Streamlit Secrets
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

SPOKE_MODELS = {
    "Gemini": "gemini/gemini-3.5-flash-lite",                 
    "Groq_Fast": "groq/openai/gpt-oss-20b",                        
    "Groq_Large": "groq/openai/gpt-oss-120b"          
}
MODERATOR_MODEL = "gemini/gemini-3.5-flash-lite"

def query_model(agent_name, model_id, prompt, max_retries=5):
    messages = [
        {"role": "system", "content": "You are a professional debater in an AI roundtable. Provide your final public argument immediately. DO NOT output any internal scratchpads or chain-of-thought."},
        {"role": "user", "content": prompt}
    ]
    for attempt in range(max_retries):
        try:
            response = completion(model=model_id, messages=messages, timeout=45)
            msg = response.choices[0].message
            content = msg.content or getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None)
            return agent_name, content or "[Empty response]"
        except Exception as e:
            if any(err in str(e).lower() for err in ["503", "429", "timeout", "connection"]):
                time.sleep(5 + (attempt * 5))
            else:
                return agent_name, f"[API Error: {str(e)}]"
    return agent_name, "[API Error: Failed]"

def run_sequential_spokes(prompt):
    results = {}
    for name, model in SPOKE_MODELS.items():
        _, content = query_model(name, model, prompt)
        results[name] = content
        time.sleep(3)
    return results

# UI Setup
st.set_page_config(page_title="AI Roundtable", page_icon="🤖", layout="centered")
st.title("🤖 Multi-Model AI Roundtable")

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "final_verdict" not in st.session_state:
    st.session_state.final_verdict = ""

# Display Chat History
for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Main Logic
if not st.session_state.topic:
    topic = st.chat_input("Enter the debate topic to start...")
    if topic:
        st.session_state.topic = topic
        st.session_state.history.append({"role": "user", "content": f"**Topic:** {topic}"})
        st.rerun()

elif not st.session_state.final_verdict:
    with st.spinner("Generating Round 1 (Initial Takes)..."):
        round1_prompt = f"Topic: {st.session_state.topic}\nTask: State your initial stance and primary argument on this topic. Be concise and persuasive."
        r1_results = run_sequential_spokes(round1_prompt)
        
    with st.spinner("Generating Round 2 (Cross-Critique)..."):
        round2_prompt = f"Topic: {st.session_state.topic}\nHere are the initial stances:\n"
        for n, a in r1_results.items(): round2_prompt += f"- {n}: {a}\n"
        round2_prompt += "\nTask: Critique the weakest point in your peers' arguments and defend your own. Be direct."
        r2_results = run_sequential_spokes(round2_prompt)
        
    with st.spinner("Synthesizing Collective Verdict..."):
        synthesis_prompt = f"Topic: {st.session_state.topic}\nRound 2 Arguments:\n"
        for n, a in r2_results.items(): synthesis_prompt += f"- {n}: {a}\n"
        synthesis_prompt += "\nTask: Act as an objective synthesis engine. DO NOT declare a 'winner'. Output a structured summary: 1. Core Agreements 2. Unresolved Disagreements 3. Conditional Synthesis 4. Unverified Variables."
        _, verdict = query_model("Moderator", MODERATOR_MODEL, synthesis_prompt)
        
    st.session_state.final_verdict = verdict
    st.session_state.history.append({"role": "assistant", "content": f"### ⚖️ Collective Verdict\n{verdict}"})
    st.rerun()

else:
    user_input = st.chat_input("Enter your opinion/counter-argument...")
    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        
        with st.spinner("Models evaluating your feedback..."):
            feedback_prompt = f"Topic: {st.session_state.topic}\nPrevious Verdict: {st.session_state.final_verdict}\n\nHuman counter-argument:\n'{user_input}'\n\nTask: Update your stance addressing the human's point. Do you agree or disagree, and why?"
            feedback_results = run_sequential_spokes(feedback_prompt)
            
        with st.spinner("Synthesizing Updated Verdict..."):
            new_synth_prompt = f"Topic: {st.session_state.topic}\nHuman's Argument: {user_input}\nModels' Responses:\n"
            for n, a in feedback_results.items(): new_synth_prompt += f"- {n}: {a}\n"
            new_synth_prompt += "\nTask: Act as an objective synthesis engine. Output an updated collective verdict that incorporates the human's input."
            _, new_verdict = query_model("Moderator", MODERATOR_MODEL, new_synth_prompt)
            
        st.session_state.final_verdict = new_verdict
        st.session_state.history.append({"role": "assistant", "content": f"### ⚖️ Updated Collective Verdict\n{new_verdict}"})
        st.rerun()
