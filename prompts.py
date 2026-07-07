# Prompts for the Video Captioning Agent

VIDEO_UNDERSTANDING_PROMPT = """
You are an expert video analysis assistant.
Below is a sequence of keyframes extracted from a video at even intervals, ordered chronologically.

Please analyze the frames carefully and provide a rich, detailed, objective, chronological description of what occurs in the video.
Focus on:
1. The setting, environment, and background.
2. The main subjects (people, animals, objects) and their physical attributes.
3. The actions, movements, interactions, and events unfolding over time.
4. Any visible text, UI elements, or graphics.
5. Overall mood, color palette, lighting, and transitions.

Provide your description as a coherent, detailed narrative. Do not interpret or inject subjective opinions; describe only what is visibly present in the sequence.
"""

STYLE_GENERATION_PROMPT = """
You are an expert creative writer and copywriter.
Below is a detailed, objective description of a video clip:

---
VIDEO DESCRIPTION:
{video_description}
---

Your task is to generate exactly four short captions (typically 1 to 3 sentences long each) in the following four styles. You must strictly adhere to the definition of each style:

1. **formal**:
   - Tone: Professional, objective, factual, and neutral.
   - Analogy: Like a documentary narrator or a serious news report.
   - Guideline: Describe the scene clearly and formally.

2. **sarcastic**:
   - Tone: Dry, ironic, lightly mocking with subtle wit.
   - Guideline: Highlight the mundane, make a dry observation, or mock the situation gently without being overly mean.

3. **humorous_tech**:
   - Tone: Funny and witty, referencing programming, software engineering, hardware, computer science, tech culture, bugs, or AI.
   - Guideline: Relate the events/subjects in the video to tech culture or tech stack jokes (e.g., "when production crashes on Friday", "running a model with 0.1 epoch", "compiling C++ code").

4. **humorous_non_tech**:
   - Tone: Light-hearted, everyday humor.
   - Guideline: Use regular everyday humor, observational comedy, or relatable jokes. Absolutely DO NOT use any technical jargon or programming concepts.

Ensure that each caption is tailored specifically to the events described in the video.

Please return your response in the following JSON format. Make sure the output is raw JSON, with no markdown code blocks around it, and it must contain these exact keys:
{{
  "formal": "Your formal caption here",
  "sarcastic": "Your sarcastic caption here",
  "humorous_tech": "Your humorous_tech caption here",
  "humorous_non_tech": "Your humorous_non_tech caption here"
}}
"""

REFINEMENT_PROMPT = """
You are an expert editor.
Review the following draft captions generated for a video clip against the original detailed objective description.

---
ORIGINAL VIDEO DESCRIPTION:
{video_description}

---
DRAFT CAPTIONS (JSON format):
{draft_captions}
---

Your task is to refine and correct these captions.
Specifically check:
1. **Factual Accuracy**: Do any of the captions contain hallucinations, factual errors, or details that contradict the original video description? If so, correct them.
2. **Style Alignment**: Does each caption strictly conform to its target style (`formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`)? Ensure `humorous_non_tech` contains absolutely NO programming/tech references or jargon.
3. **Format Integrity**: Return the corrected captions in the exact same JSON format with keys "formal", "sarcastic", "humorous_tech", "humorous_non_tech".

Please return your response in the following JSON format. Make sure the output is raw JSON, with no markdown code blocks around it:
{{
  "formal": "Refined formal caption here",
  "sarcastic": "Refined sarcastic caption here",
  "humorous_tech": "Refined humorous_tech caption here",
  "humorous_non_tech": "Refined humorous_non_tech caption here"
}}
"""

