from textwrap import dedent

INFERENCE_FIND_PROMPT = dedent("""
Find the object_id that exactly matches the following description: "{}".
Only select object IDs from the provided visible set: {}. 
Do not invent or output IDs that are not included in this set.

Follow this procedure:
1) Enumerate the green numeric IDs visible in the images (target candidates).
2) Identify the target object_id based on visual and spatial attributes.
3) If exactly one green candidate fully satisfies, return that single id.
4) If no green candidates satisfy:
   - If the target class is clearly not visible near any red reference, return [].
   - If the target may exist but is likely occluded or poorly visible near a red reference, you may return that single red ID instead as a signal for further exploration.
5) If neither green nor red satisfy, return [].

Return a JSON object with:
    - "object_ids": the id of the selected object (one-element list), or [] if none match.
    - "reason": a short explanation of why the object_id was selected (e.g., based on color, shape, position, or fallback exploration reasoning).
""").strip()

INFERENCE_COUNT_PROMPT = dedent("""
Count all objects that exactly match the following description: "{}".
Candidate object_ids: {}. 
Identify every instance that satisfies all visual and spatial attributes.
Return a JSON object with:
    - "object_ids": the ids of matching objects. (If no such object exists, return [])
    - "count": the total number of matching objects. It must be same as the length of "object_ids".
    - "reason": a short explanation of how the objects were identified (e.g., based on color, shape, position).
""").strip()

INFERENCE_FOLLOW_BETWEEN_PROMPT = dedent("""
Find the directly related objects to find the path that satisfies the following description: "{}".
Only select object IDs from the provided visible set: {}. 
Do not invent or output IDs that are not included in this set.

Follow this procedure:
1) Prefer selecting a pair of green IDs that clearly satisfy the 'between' or relational condition.
2) If no valid green pair exists:
   - If the relation is uncertain due to occlusion or poor visibility near a red reference, you may return a **single red ID** (tuple with one element) as a signal for further exploration.
3) If neither a valid green pair nor a red exploration candidate exists, return [].

Return a JSON object with:
    - "object_ids": [<id_a>, <id_b>] (two elements) when a valid green pair is found,
                    OR [<id>] (single element) when returning a red bbox exploration signal,
                    OR [] if no match.
    - "reason": a short explanation of why the object_ids were selected (e.g., based on color, shape, position, relation, or exploration reasoning).
""").strip()

VALIDATE_PROMPT = dedent("""
Validate whether the current annotations fully satisfy the following description: "{}".

Follow this procedure:
1) For each annotated ID, verify it satisfies all visual/spatial attributes in the description
   (e.g., color, shape, size, position, relations like "on the left", "next to"), using multi-view consistency.
2) If any annotated object violates the description, the annotation is incorrect.
3) If the description implies multiple instances (e.g., plurals like "books") and the provided set does not cover them,
   treat it as incomplete.
4) If all annotated objects satisfy the description and there is no missing required instance within the provided candidates,
   the annotation is fully satisfied.

Always return a strict JSON object with:
    - "confidence": 1 if fully satisfied; otherwise 0
    - "reason": a short explanation (≤ 2 sentences) describing the key evidence for the decision
""").strip()

VALIDATE_FIND_PROMPT = dedent("""
Validate whether the single annotated object in the images fully satisfies the following description (FIND task): "{}".
Exactly one annotated green numeric ID is provided: {}. Do not invent IDs or consider non-annotated objects.

Success criteria (FIND):
- The single annotated ID must satisfy all visual/spatial attributes in the description.
- If it violates any attribute, the annotation is not fully satisfied.

Procedure:
1) Verify the single annotated ID against color/shape/size/position/relations (e.g., "on the left", "next to") with multi-view consistency.
2) Return confidence 1 if the single ID fully satisfies the description; otherwise return confidence 0.

Always return a strict JSON object with:
    - "confidence": 1 if fully satisfied; otherwise 0
    - "reason": a short explanation (≤ 2 sentences) describing the key evidence
""").strip()

VALIDATE_COUNT_PROMPT = dedent("""
Validate whether the current annotations fully satisfy the following description (COUNT task): "{}".
Only consider these annotated green numeric IDs in the images: {}. Do not invent IDs.

Success criteria (COUNT):
- All and only the objects that satisfy the description must be annotated (complete and precise).
- If any annotated ID violates the description (false positive) or any required instance is missing (false negative), it is not fully satisfied.

Procedure:
1) For each annotated ID, verify the description via color/shape/size/position/relations with multi-view consistency.
2) Ensure there are no violating IDs and no missing instances implied by the description (e.g., plurals like "books").

Always return a strict JSON object with:
    - "confidence": 1 if fully satisfied; otherwise 0
    - "reason": a short explanation (≤ 2 sentences) describing the key evidence
""").strip()

VALIDATE_FOLLOW_BETWEEN_PROMPT = dedent("""
Validate whether the current annotations fully satisfy the following description (FOLLOW_BETWEEN task): "{}".
Only consider these annotated green numeric IDs in the images: {}. Do not invent IDs.

Success criteria (FOLLOW_BETWEEN):
- Exactly two distinct annotated IDs must define the endpoints referenced by the description (e.g., "between the TV and the tea table").
- Both endpoints must be correct in class/identity and spatial arrangement; extras or wrong pairs invalidate the result.

Procedure:
1) Verify that exactly two annotated IDs are intended as the endpoints.
2) Check both endpoints satisfy the described classes/attributes and the spatial relation ("between") under multi-view consistency.

Always return a strict JSON object with:
    - "confidence": 1 if fully satisfied; otherwise 0
    - "reason": a short explanation (≤ 2 sentences) describing the key evidence
""").strip()


PATH_GENERATION_PROMPT = dedent("""
Generate a path that satisfies the following description: "{}".
Green numbers: {}.
Red numbers: {}.
Blue numbers: {}.

Return a JSON object with:
    - "selected_numbers": list[int], // the ids of navigable positions
    - "reason": string // short explanation
""").strip()

PATH_EVALUATION_PROMPT = dedent("""
Evaluate the robot's mission based on the following description: "{}".
Return a JSON object with:
    - "mission_status": 1 if the mission is complete; otherwise 0
    - "reason": string // short explanation
""").strip()

PROMPT = {
    "inference_find": INFERENCE_FIND_PROMPT,
    "inference_count": INFERENCE_COUNT_PROMPT,
    "inference_follow_between": INFERENCE_FOLLOW_BETWEEN_PROMPT,
    "validate_find": VALIDATE_FIND_PROMPT,
    "validate_count": VALIDATE_COUNT_PROMPT,
    "validate_follow_between": VALIDATE_FOLLOW_BETWEEN_PROMPT,
    "path_generation": PATH_GENERATION_PROMPT,
    "path_evaluation": PATH_EVALUATION_PROMPT,
}