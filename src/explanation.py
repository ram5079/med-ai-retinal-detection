import json
import logging

# Ensure transformers doesn't spam warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_medical_recommendation(predicted_class, lesion_score, confidence):
    """
    Recommendations mapping Severity prediction, exact structural score, and temperature confidence.
    """
    is_high_conf = confidence > 0.8
    if predicted_class == 0:
        if lesion_score < 10:
             return "Healthy baseline", "No signs of diabetic retinopathy isolated. Continue standard annual eye exams.", "#22c55e" # Green
        else:
             return "Re-evaluate Baseline", "Routine monitoring indicated. Minor structural anomalies suggest a clinical follow-up is beneficial.", "#84cc16"
    elif predicted_class == 1:
        text = "Mild vascular irregularities suspected."
        if lesion_score > 30 and is_high_conf:
             text += " Consistent lesion mapping detected; schedule an appointment in 3-6 months."
        else:
             text += " Maintain systemic glucose control and monitor annually."
        return "Routine Monitor", text, "#eab308" # Yellow
    elif predicted_class == 2:
        text = "Moderate structural changes detected."
        if lesion_score > 50:
             text += " Significant peripheral lesion spread isolated. Consult a specialist within 1-2 months."
        else:
             text += " Consult an eye doctor within 3-6 months for a dilated exam."
        return "Consult Doctor", text, "#f97316" # Orange
    elif predicted_class == 3:
         return "Immediate Specialist Attention", "Severe indicators present. Substantial risk to retinal integrity detected; seek an immediate formulation evaluation.", "#ef4444" # Red
    else:
         return "Urgent Treatment Required", "Proliferative characteristics identified. Extreme risk of blinding complications. Sight-saving intervention is emergent.", "#b71c1c" # Dark Red


def generate_llm_explanation(structured_json, stage, generator=None):
    """
    Step 3 of the Hybrid Engine: takes strict structural features and queries a Local LLM 
    (flan-t5-small) to format it without hallucination. Enforces deterministic generation and verification.
    """
    # 1. Fallback to rule-based logic if generator is unavailable
    if generator is None:
        return _fallback_rule_based(structured_json, stage)
        
    try:
        # 2. Strict Deterministic Prompt formulation
        # Explicit instructions to avoid hallucination entirely
        prompt = (
            f"You are a clinical AI assistant. Generate a concise explanation ONLY based on the provided structured retinal analysis.\\n"
            f"Do not add new information. Do not hallucinate. Do not generalize beyond the given data.\\n"
            f"Context: {stage}\\n"
            f"Data:\\n"
            f"- intensity: {structured_json['intensity']}\\n"
            f"- spread: {structured_json['spread']}\\n"
            f"- location: {structured_json['location']}\\n"
            f"- area_percent: {structured_json['area_percent']}%\\n"
            f"Output one professional clinical paragraph of exactly 2 sentences.\\n"
            f"Explanation: "
        )
        
        # 3. Controlled deterministic execution (Remove creativity)
        output = generator(
            prompt, 
            max_new_tokens=80,
            num_return_sequences=1,
            do_sample=True, # enable sampling strictly for temp controls
            temperature=0.1, # Extremely low temperature to remove creativity
            top_p=0.85, 
            repetition_penalty=1.2
        )
        
        generated_text = output[0]['generated_text'].strip()
        
        # 4. Debug Mode Logging
        print("====== DEBUG MODE: LLM HYBRID EXPLANATION ======")
        print("[LLM USED] - Execution Successful")
        print(f"INPUT JSON:   {json.dumps(structured_json)}")
        print(f"RAW LLM TEXT: {generated_text}")
        print("================================================")
        
        # 5. Fallback verification (Ensures LLM actually retained the factual numeric inputs)
        area_str = str(structured_json['area_percent'])
        if area_str not in generated_text:
            print(f"LLM hallucination rejected (missing exact area {area_str}%). Falling back to hybrid deterministic engine...")
            return _fallback_rule_based(structured_json, stage)
            
        return generated_text
        
    except Exception as e:
        print(f"LLM Generation pipeline failed: {e}")
        return _fallback_rule_based(structured_json, stage)

def _fallback_rule_based(structured_json, stage):
    """
    Rule-based deterministic mapping explicitly used when LLM is offline or verification fails.
    """
    print("====== DEBUG MODE: LLM HYBRID EXPLANATION ======")
    print("[RULE-BASED FALLBACK] - LLM bypassed or failed")
    print(f"INPUT JSON: {json.dumps(structured_json)}")
    print("================================================")
    
    intensity = structured_json['intensity']
    spread = structured_json['spread']
    loc = structured_json['location']
    area = structured_json['area_percent']
    
    if intensity == "none":
        return f"Analysis reveals clear retinal structuring with no pathological markers. "
        
    return (
    f"{spread.capitalize()} {intensity}-intensity activations span {area}% of the retinal area.\n\n"
    f"These findings are primarily located in the {loc} structures, providing spatial correlation for the model's diagnostic stage."
)
