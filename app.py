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

def query_model(agent_name, model_id, prompt, is_moderator=False, max_retries=5):
    if is_moderator:
        sys_content = "You are the final, objective Fact-Checking Judge. Evaluate the evidence strictly. Discard any fabricated specs or flawed math. Do not declare a winner. Synthesize only verified facts."
    else:
        sys_content = "You are a professional debater. DO NOT output any internal scratchpads. CRITICAL RULE: If you cite a specific number, price, wattage, or date, you MUST explicitly state whether it is a known fact or a theoretical estimate. Do not present estimates as facts."

    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": prompt}
    ]
    
    kwargs = {"model": model_id, "messages": messages, "timeout": 90}
    if is_moderator and "gemini" in model_id.lower():
        kwargs["tools"] = [{"googleSearch": {}}]
        
    for attempt in range(max_retries):
        try:
            response = completion(**kwargs)
            msg = response.choices[0].message
            content = msg.content or getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None)
            return agent_name, content or "[Empty response]"
        except Exception as e:
            if any(err in str(e).lower() for err in ["503", "429", "timeout", "connection"]):
                time.sleep(5 + (attempt * 5))
            else:
                return agent_name, f"[API Error: {str(e)}]"
    return agent_name, "[API Error: Failed]"

# UI Setup
st.set_page_config(page_title="AI Roundtable", page_icon="🤖", layout="centered")
st.title("🤖 Multi-Model AI Roundtable")
st.caption("Asymmetric Verification Edition: Live Debate Feed + Fact-Checking Judge")

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

# Helper function to run rounds and show live output
def run_live_round(prompt, round_title):
    results = {}
    combined_output = f"### {round_title}\n\n"
    
    with st.chat_message("assistant"):
        st.markdown(f"### {round_title}")
        for name, model in SPOKE_MODELS.items():
            with st.spinner(f"⏳ {name} is typing..."):
                _, content = query_model(name, model, prompt, is_moderator=False)
                results[name] = content
                st.markdown(f"**[{name}]**\n\n{content}")
                st.divider()
                combined_output += f"**[{name}]**\n{content}\n\n---\n\n"
                time.sleep(2)
                
    st.session_state.history.append({"role": "assistant", "content": combined_output})
    return results

# Main Logic
if not st.session_state.topic:
    topic = st.chat_input("Enter the debate topic to start...")
    if topic:
        st.session_state.topic = topic
        st.session_state.history.append({"role": "user", "content": f"**Topic:** {topic}"})
        st.rerun()

elif not st.session_state.final_verdict:
    round1_prompt = f"Topic: {st.session_state.topic}\nTask: State your initial stance and primary argument on this topic. Be concise and persuasive."
    r1_results = run_live_round(round1_prompt, "🎙️ Round 1: Initial Takes")
    
    round2_prompt = f"Topic: {st.session_state.topic}\nHere are the initial stances:\n"
    for n, a in r1_results.items(): round2_prompt += f"- {n}: {a}\n"
    round2_prompt += "\nTask: Critique the weakest point in your peers' arguments and defend your own. Expose unverified assumptions. Be direct."
    r2_results = run_live_round(round2_prompt, "⚔️ Round 2: Cross-Critique")
        
    with st.chat_message("assistant"):
        with st.spinner("⚖️ Moderator is fact-checking & synthesizing..."):
            synthesis_prompt = f"Topic: {st.session_state.topic}\nRound 2 Arguments:\n"
            stance_labels = ["Stance A", "Stance B", "Stance C"]
            for i, (model_name, arg) in enumerate(r2_results.items()):
                synthesis_prompt += f"- {stance_labels[i]}: {arg}\n\n"
                
            synthesis_prompt += "\nTask: Act as an objective synthesis engine. Evaluate the stances blindly based purely on evidence. Output a structured summary: 1. Core Agreements 2. Fact-Check & Discrepancies (Call out false claims) 3. Conditional Synthesis 4. Unverified Variables."
            _, verdict = query_model("Moderator", MODERATOR_MODEL, synthesis_prompt, is_moderator=True)
            
            st.markdown(f"### ⚖️ Collective Verdict\n{verdict}")
            st.session_state.history.append({"role": "assistant", "content": f"### ⚖️ Collective Verdict\n{verdict}"})
            
    st.session_state.final_verdict = verdict
    st.rerun()

else:
    user_input = st.chat_input("Enter your opinion/counter-argument...")
    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        
        feedback_prompt = f"Topic: {st.session_state.topic}\nPrevious Verdict: {st.session_state.final_verdict}\n\nHuman counter-argument:\n'{user_input}'\n\nTask: Update your stance addressing the human's point. Do you agree or disagree, and why?"
        feedback_results = run_live_round(feedback_prompt, "🔄 Models Evaluating Feedback")
            
        with st.chat_message("assistant"):
            with st.spinner("⚖️ Moderator is fact-checking & synthesizing updated verdict..."):
                new_synth_prompt = f"Topic: {st.session_state.topic}\nHuman's Argument: {user_input}\nModels' Responses:\n"
                stance_labels = ["Stance A", "Stance B", "Stance C"]
                for i, (model_name, arg) in enumerate(feedback_results.items()):
                    new_synth_prompt += f"- {stance_labels[i]}: {arg}\n\n"
                    
                new_synth_prompt += "\nTask: Act as an objective synthesis engine. Output an updated collective verdict that incorporates the verified facts from the human's input."
                _, new_verdict = query_model("Moderator", MODERATOR_MODEL, new_synth_prompt, is_moderator=True)
                
                st.markdown(f"### ⚖️ Updated Collective Verdict\n{new_verdict}")
                st.session_state.history.append({"role": "assistant", "content": f"### ⚖️ Updated Collective Verdict\n{new_verdict}"})
                
        st.session_state.final_verdict = new_verdict
        st.rerun()
