import streamlit as st
import os
import time
import json
from litellm import completion

# 1. Load Keys from Streamlit Secrets
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

SPOKE_MODELS = {
    "Groq_Fast": "groq/openai/gpt-oss-20b",
    "Groq_Large": "groq/openai/gpt-oss-120b",
    "Gemini": "gemini/gemini-3.5-flash-lite"
}
MODERATOR_MODEL = "groq/openai/gpt-oss-120b"

def fetch_web_facts(query):
    """Safely fetch live web snippets. Now returns actual error text if blocked."""
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(query, max_results=4)
        if results:
            return "\n".join([f"- {res.get('body', '')}" for res in results])
    except Exception as e:
        return f"Search Error: {str(e)}"
    return "No live web data available."

def generate_search_query(user_topic, human_input=""):
    """Uses JSON Mode to mathematically force the LLM to output one clean search string."""
    prompt = (
        f"Topic: '{user_topic}'\nHuman input: '{human_input}'\n\n"
        "Extract the core product names and technical keywords to fact-check this. "
        "Output ONLY a valid JSON object with a single key 'query'. "
        "Example: {\"query\": \"Sony HT-S40R vs Sony HT-S20R specifications\"}"
    )
    try:
        # JSON output guarantees no extra text or glued queries
        response = completion(
            model="groq/openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            timeout=15
        )
        return json.loads(response.choices[0].message.content)["query"]
    except Exception:
        # Safe string fallback
        clean_fallback = user_topic.replace("?", "").replace("!", "")[:40].strip()
        return f"{clean_fallback} specifications"

def query_model(agent_name, model_id, prompt, is_moderator=False, max_retries=5):
    if is_moderator:
        sys_content = (
            "You are the final, objective Fact-Checking Judge. Evaluate the evidence strictly against the provided Live Web Facts. "
            "CRITICAL: If Live Web Facts say 'No live web data available' or lack exact technical numbers, DO NOT invent or guess specs. "
            "Acknowledge that verified specs were unavailable rather than assuming them. Synthesize only verified facts."
        )
    else:
        sys_content = "You are a professional debater. DO NOT output any internal scratchpads. CRITICAL RULE: If you cite a specific number, price, wattage, or date, you MUST explicitly state whether it is a known fact or a theoretical estimate. Do not present estimates as facts."

    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": prompt}
    ]
    
    kwargs = {"model": model_id, "messages": messages, "timeout": 90}
    last_error_msg = ""
    
    for attempt in range(max_retries):
        try:
            response = completion(**kwargs)
            msg = response.choices[0].message
            content = msg.content or getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None)
            return agent_name, content or "[Empty response]", kwargs["model"]
        except Exception as e:
            last_error_msg = str(e)
            
            # FIXED FALLBACK: Actually switch to a different model if rate-limited
            if "429" in last_error_msg and is_moderator:
                fallback_kwargs = kwargs.copy()
                fallback_kwargs["model"] = "groq/llama3-8b-8192" 
                try:
                    fallback_response = completion(**fallback_kwargs)
                    fallback_msg = fallback_response.choices[0].message
                    content = fallback_msg.content or getattr(fallback_msg, 'reasoning_content', None) or getattr(fallback_msg, 'reasoning', None)
                    if content:
                        return agent_name, content + "\n\n*(Note: Fact-checking downgraded due to quota limits)*", fallback_kwargs["model"]
                except Exception:
                    pass

            if any(err in last_error_msg.lower() for err in ["503", "429", "timeout", "connection"]):
                time.sleep(5 + (attempt * 5))
            else:
                return agent_name, f"[API Error: {last_error_msg}]", model_id
                
    return agent_name, f"[API Error: Failed after 5 retries. Last error: {last_error_msg}]", model_id

# UI Setup
st.set_page_config(page_title="AI Roundtable", page_icon="🤖", layout="centered")
st.title("🤖 Multi-Model AI Roundtable")
st.caption("Agentic Search Edition: Live Debate Feed + Fact-Checking Judge")

if "history" not in st.session_state:
    st.session_state.history = []
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "final_verdict" not in st.session_state:
    st.session_state.final_verdict = ""

for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

def run_live_round(prompt, round_title):
    results = {}
    combined_output = f"### {round_title}\n\n"
    
    with st.chat_message("assistant"):
        st.markdown(f"### {round_title}")
        for name, model in SPOKE_MODELS.items():
            with st.spinner(f"⏳ {name} is typing..."):
                _, content, _ = query_model(name, model, prompt, is_moderator=False)
                results[name] = content
                st.markdown(f"**[{name}]**\n\n{content}")
                st.divider()
                combined_output += f"**[{name}]**\n{content}\n\n---\n\n"
                time.sleep(2)
                
    st.session_state.history.append({"role": "assistant", "content": combined_output})
    return results

def get_author_label(model_id):
    if "gpt-oss-120b" in model_id.lower():
        return "⚡ Groq GPT-OSS-120B (with Agentic Web Search)"
    elif "llama3" in model_id.lower():
        return "⚡ Groq Llama 3 8B (Fallback)"
    elif "gemini" in model_id.lower():
        return "🤖 Gemini 3.5 Flash Lite"
    else:
        return f"⚡ {model_id}"

# Debate Execution
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
    for n, a in r1_results.items():
        round2_prompt += f"- {n}: {a}\n"
    round2_prompt += "\nTask: Critique the weakest point in your peers' arguments and defend your own. Expose unverified assumptions. Be direct."
    r2_results = run_live_round(round2_prompt, "⚔️ Round 2: Cross-Critique")
        
    with st.chat_message("assistant"):
        with st.spinner("🔍 Agent is extracting clean search keywords..."):
            smart_query = generate_search_query(st.session_state.topic)
            
        with st.spinner(f"⚖️ Moderator is fact-checking '{smart_query}' & synthesizing..."):
            live_facts = fetch_web_facts(smart_query)
            
            synthesis_prompt = f"Topic: {st.session_state.topic}\nLive Web Facts:\n{live_facts}\n\nRound 2 Arguments:\n"
            stance_labels = ["Stance A", "Stance B", "Stance C"]
            for i, (model_name, arg) in enumerate(r2_results.items()):
                synthesis_prompt += f"- {stance_labels[i]}: {arg}\n\n"
                
            synthesis_prompt += "\nTask: Act as an objective synthesis engine. Cross-check the stances against the Live Web Facts. Discard fabricated specs. Output a structured summary: 1. Core Agreements 2. Fact-Check & Discrepancies 3. Conditional Synthesis 4. Unverified Variables."
            _, verdict, actual_model = query_model("Moderator", MODERATOR_MODEL, synthesis_prompt, is_moderator=True)
            
            author_label = get_author_label(actual_model)
            final_display = f"### ⚖️ Collective Verdict\n*(Written by {author_label})*\n\n**🔍 Search Query Used:** `{smart_query}`\n\n{verdict}"
            
            st.markdown(final_display)
            st.session_state.history.append({"role": "assistant", "content": final_display})
            
    st.session_state.final_verdict = final_display
    st.rerun()

else:
    user_input = st.chat_input("Enter your opinion/counter-argument...")
    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        feedback_prompt = f"Topic: {st.session_state.topic}\nPrevious Verdict: {st.session_state.final_verdict}\n\nHuman counter-argument:\n'{user_input}'\n\nTask: Update your stance addressing the human's point. Do you agree or disagree, and why?"
        feedback_results = run_live_round(feedback_prompt, "🔄 Models Evaluating Feedback")
            
        with st.chat_message("assistant"):
            with st.spinner("🔍 Agent is extracting clean search keywords..."):
                smart_query = generate_search_query(st.session_state.topic, user_input)
                
            with st.spinner(f"⚖️ Moderator is searching web & updating verdict..."):
                live_facts = fetch_web_facts(smart_query)
                
                new_synth_prompt = f"Topic: {st.session_state.topic}\nHuman's Argument: {user_input}\nLive Web Facts:\n{live_facts}\n\nModels' Responses:\n"
                stance_labels = ["Stance A", "Stance B", "Stance C"]
                for i, (model_name, arg) in enumerate(feedback_results.items()):
                    new_synth_prompt += f"- {stance_labels[i]}: {arg}\n\n"
                    
                new_synth_prompt += "\nTask: Act as an objective synthesis engine. Output an updated collective verdict that incorporates verified facts from the web and human input."
                _, new_verdict, actual_model = query_model("Moderator", MODERATOR_MODEL, new_synth_prompt, is_moderator=True)
                
                author_label = get_author_label(actual_model)
                updated_display = f"### ⚖️ Updated Collective Verdict\n*(Written by {author_label})*\n\n**🔍 Search Query Used:** `{smart_query}`\n\n{new_verdict}"
                
                st.markdown(updated_display)
                st.session_state.history.append({"role": "assistant", "content": updated_display})
                
        st.session_state.final_verdict = updated_display
        st.rerun()
